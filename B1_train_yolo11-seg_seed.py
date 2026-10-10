"""
B1_train_yolo11-seg_v2.py
===============================================================
方案B的第2步：訓練一個「獨立的」牙齒實例分割(instance segmentation)模型。

這支腳本跟 B1_train_yolo11-pose_v2.py 是平行關係，不是取代關係：
    B1  → pose 模型 → 預測 A(切端)/B(根尖) 關鍵點   [主線，不動]
    B1s → seg  模型 → 預測牙齒輪廓 mask             [輔助，本檔]
    B3  → 讀 B2 的關鍵點 CSV + 本模型的 mask，做防呆/投影/對照

刻意沿用 B1 的超參數(IMG_SIZE、BATCH、LR0、AMP、augmentation 強度、
SPLIT_SEED、VAL_RATIO)。小資料集上這些值是踩過坑調出來的，
這次實驗要驗證的是「分割有沒有用」，不是「超參數能不能再調」，
一次只動一個變因。

*** v2 變更：多個訓練 seed ***
舊版的 RANDOM_SEED=42 只用在 train/val 切分，model.train() 沒傳 seed，
實際訓練 seed 是 Ultralytics 預設的 0。v2 把兩者拆開：
  - SPLIT_SEED = 42      → train/val 切分，固定不動(所有 seed 切到同一組 val)
  - TRAIN_SEEDS = [0..4] → 只改訓練隨機性(head 初始化、batch 洗牌、augmentation)
每個 seed 輸出到 {WORK_DIR}_seed{s}/，最後寫一份跨 seed 彙整。
切分改用獨立的 random.Random(SPLIT_SEED)：迴圈中全域 random 會被前一輪
(與 Ultralytics 的 init_seeds)推進，用全域狀態會讓每個 seed 切到不同 val。
main() 會檢查所有 seed 的 val 名單完全一致，不一致直接報錯。

舊版 yolo11n_seg_run_padded/ 的結果 = seed 0。若不想重訓，可把該資料夾
複製成 yolo11n_seg_run_padded_seed0/，腳本偵測到 weights_ready.pt 會跳過訓練。
(重訓 seed 0 也可以，順便檢查可重現性；GPU 上可能有極小差異，屬正常。)

*** 標註格式需求 ***
SEG_DATA_DIR 底下是 Roboflow 匯出的 YOLOv11/YOLOv8 Instance Segmentation 格式：
  - images/、labels/(或 train/valid/test 三個子資料夾各自有 images/、labels/)
  - 每個 labels/*.txt 一行一個 instance：
        "class x1 y1 x2 y2 x3 y3 ... xn yn"
    座標是正規化到 0~1 的多邊形頂點，至少 3 個點(6 個數字)。
    注意這跟 pose 的 "class cx cy w h kx ky v ..." 格式完全不同。

*** 關於「要標幾顆牙」***
兩種都可行，差在學長的時間成本，預設走 A 案：

  A 案(預設)：每張圖只標「目標牙」那一顆
      成本最低(1 顆/張)。缺點是其他外觀相似的牙齒被當成背景，
      監督訊號互相衝突，模型的 confidence 會偏低、可能漏偵測。
      但方案B不在乎模型認不認得「哪顆是目標牙」——B3 是用
      pose 預測的 A/B 點去挑 mask 的。所以只要目標牙那顆的
      「輪廓品質」夠好就達成目的。SEG_CONF 記得調低。

  B 案：每張圖標所有可見的牙齒，全部同一個 class
      監督訊號乾淨，模型表現會明顯好；成本約 10 倍。
      如果 A 案跑出來漏偵測嚴重(B3 的「seg無輸出」很多)，再升級。

*** 座標空間 ***
seg 專案與 pose 專案必須用「同一批圖 + Roboflow Generate 完全相同的
Resize 設定」，否則 mask 與 keypoint 會落在兩個座標系。

安裝需求：
    pip install ultralytics pandas pyyaml openpyxl
"""

import random
import shutil
from pathlib import Path

import pandas as pd
import yaml
from ultralytics import YOLO

# ---------------- 設定區 ----------------
SEG_DATA_DIR = "A5_train_padded"        # 👈 Roboflow Instance Segmentation 匯出資料夾
WORK_DIR = "yolo11n_seg_run_padded"      # 實際輸出到 {WORK_DIR}_seed{s}/

BASE_MODEL = "yolo11n-seg.pt"                     # 👈 跟 pose 版唯一的模型差異
EPOCHS = 150

# 以下刻意與 B1_train_yolo11-pose_v2.py 對齊，不要單獨調整
IMG_SIZE = 640
VAL_RATIO = 0.2
SPLIT_SEED = 42             # train/val 切分(原 RANDOM_SEED)，固定不動
CLASS_NAMES = ["tooth"]     # A 案與 B 案都只用單一 class

# 本次實驗唯一的變因
TRAIN_SEEDS = [0, 1, 2, 3, 4]   # 0 = 舊版實際用的 seed(Ultralytics 預設)
SEED_SUMMARY_XLSX = f"{WORK_DIR}_seed彙整.xlsx"

PATIENCE = 0                # early stopping 關閉
LR0 = 0.001                 # 小資料集調低初始學習率
AMP = False                 # 關閉混合精度，排除 NaN 造成 fitness collapse
BATCH = 4


# ============================================================
# 第1部分：資料集收集與切分
# ============================================================

def collect_seg_dataset():
    """相容 Roboflow 標準 train/valid/test 匯出或扁平 images/+labels/ 結構，
    全部 pool 起來，由這支腳本自己重新切一次 train/val。"""
    pool = []
    seen_ids = {}
    for split_name in ("train", "valid", "test", "images"):
        if split_name == "images":
            img_dir = Path(SEG_DATA_DIR) / "images"
            lbl_dir = Path(SEG_DATA_DIR) / "labels"
        else:
            img_dir = Path(SEG_DATA_DIR) / split_name / "images"
            lbl_dir = Path(SEG_DATA_DIR) / split_name / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"⚠️ {img_path.name} 沒有對應的標註檔，跳過")
                continue
            if img_path.stem in seen_ids:
                print(f"⚠️ 圖片ID重複：'{img_path.stem}'，已出現在 "
                      f"{seen_ids[img_path.stem]}，這次({split_name})跳過")
                continue
            seen_ids[img_path.stem] = split_name
            pool.append({"img_id": img_path.stem, "img_path": img_path,
                         "label_path": label_path})

    print(f"\n✅ 分割資料 pool 完成，總共 {len(pool)} 張X光片")
    if len(pool) < 20:
        print(f"⚠️ 只有 {len(pool)} 張圖有標註，資料量偏少，訓練結果可能不穩定。")
    return pool


def split_train_val(pool, val_ratio):
    """用獨立的 RNG 切分：不受全域 random 狀態影響，每個 seed 都切到同一組。
    random.Random(42).shuffle 與舊版 random.seed(42)+random.shuffle 結果相同，
    所以 val 名單與舊版一致。"""
    shuffled = pool[:]
    random.Random(SPLIT_SEED).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_items, train_items = shuffled[:n_val], shuffled[n_val:]
    print(f"📊 train/val 切分：train {len(train_items)} 張 / val {len(val_items)} 張"
          f"(val_ratio={val_ratio}, split_seed={SPLIT_SEED})")
    return train_items, val_items


def copy_items(items, dst_img_dir, dst_lbl_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        shutil.copy2(item["img_path"], dst_img_dir / item["img_path"].name)
        shutil.copy2(item["label_path"], dst_lbl_dir / item["label_path"].name)


# ============================================================
# 第2部分：標註品質統計
# ============================================================

def count_polygons_per_image(items):
    """統計每張圖有幾個 instance、多邊形頂點數，並抓出格式異常的行。

    頂點數是分割標註品質的直接指標：Roboflow 的 Smart Polygon 產出的
    點數通常較多，手繪的較少。點數過少(例如 <8)代表輪廓被簡化得太粗，
    牙根尖端的形狀可能已經被抹平——而那正是你唯一在乎的區域。
    """
    records = []
    for item in items:
        n_inst, verts, n_bad = 0, [], 0
        with open(item["label_path"], "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for line in lines:
            parts = line.split()
            coords = parts[1:]
            # 合法的多邊形：偶數個座標，且至少 3 個點
            if len(coords) < 6 or len(coords) % 2 != 0:
                n_bad += 1
                continue
            n_inst += 1
            verts.append(len(coords) // 2)
        records.append({
            "檔名": item["img_path"].name,
            "instance數": n_inst,
            "最少頂點數": min(verts) if verts else 0,
            "平均頂點數": round(sum(verts) / len(verts), 1) if verts else 0,
            "格式異常行數": n_bad,
        })
    return records


def report_annotation_quality(df: pd.DataFrame):
    if df.empty:
        return
    n_bad = int(df["格式異常行數"].sum())
    if n_bad:
        print(f"❌ 發現 {n_bad} 行格式異常的標註(座標數不是偶數或少於3點)。")
        print(f"   最可能的原因：匯出成了 pose/bbox 格式而不是 Segmentation 格式。")

    mean_inst = df["instance數"].mean()
    if mean_inst < 1.5:
        print(f"ℹ️  每張圖平均 {mean_inst:.1f} 個 instance → 判定為「A 案：只標目標牙」")
        print(f"   這是預期的。模型 confidence 會偏低，B3 的 SEG_CONF 建議設 0.10~0.15。")
    else:
        print(f"ℹ️  每張圖平均 {mean_inst:.1f} 個 instance → 判定為「B 案：標多顆牙」")

    thin = df[df["最少頂點數"] < 8]
    if not thin.empty:
        print(f"⚠️ 有 {len(thin)} 張圖的多邊形頂點數少於 8，輪廓可能過度簡化：")
        print(f"   {', '.join(thin['檔名'].head(5).tolist())}")
        print(f"   → 牙根尖端的形狀是這個方案的重點，頂點太少會把它抹平，建議補標。")


# ============================================================
# 第3部分：訓練
# ============================================================

def seed_work_dir(train_seed):
    return Path(f"{WORK_DIR}_seed{train_seed}")


def write_data_yaml(work_dir):
    """注意：seg 的 data.yaml 不需要也不能有 kpt_shape。"""
    data_yaml = {
        "path": str(work_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out_path = work_dir / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True)
    return out_path


def extract_seg_map(trainer):
    """撈 Box(B) 與 Mask(M) 兩種 mAP。

    seg 的 mask mAP 不像 pose 的 OKS 那樣容易灌水(mask IoU 是實打實的
    像素重疊)，所以這個數字比 pose mAP 可信。但它衡量的是「整顆牙的
    輪廓重疊度」，而牙根尖端只佔全牙面積的極小比例——mask mAP 很高
    不代表根尖畫得準。最終還是要看 B3 的長度誤差。
    """
    out = {}
    try:
        raw = getattr(trainer, "metrics", None) or {}
        for key, value in raw.items():
            if not key.startswith("metrics/"):
                continue
            name = key.replace("metrics/", "")
            if name.endswith("(B)"):
                out[f"Box_{name[:-3]}"] = round(float(value), 4)
            elif name.endswith("(M)"):
                out[f"Mask_{name[:-3]}"] = round(float(value), 4)
    except Exception as e:
        print(f"   ⚠️ 讀不到驗證指標({e})")
    return out


def train_seg_model(train_seed):
    """回傳 (權重路徑, val 檔名集合)。切分一律先做，供 main 檢查跨 seed 一致性。"""
    work_dir = seed_work_dir(train_seed)
    ready_path = work_dir / "weights_ready.pt"

    pool = collect_seg_dataset()
    if not pool:
        raise FileNotFoundError(f"在 {SEG_DATA_DIR} 裡找不到任何已標註的資料")
    train_items, val_items = split_train_val(pool, VAL_RATIO)
    val_ids = {it["img_id"] for it in val_items}

    if ready_path.exists():
        print(f"⚠️ 偵測到已訓練的模型 {ready_path}，直接使用。"
              f"若要重新訓練請先刪除 {work_dir}")
        return ready_path, val_ids

    poly_df = pd.DataFrame(count_polygons_per_image(train_items))
    print("\n=== 標註品質 ===")
    report_annotation_quality(poly_df)

    data_root = work_dir / "data"
    copy_items(train_items, data_root / "train" / "images", data_root / "train" / "labels")
    copy_items(val_items, data_root / "val" / "images", data_root / "val" / "labels")
    data_yaml = write_data_yaml(data_root)

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        optimizer="AdamW",
        project=str(work_dir),
        name="tooth_seg",
        val=True,
        patience=PATIENCE,
        lr0=LR0,
        amp=AMP,
        batch=BATCH,
        seed=train_seed,        # v2：訓練隨機性，本次唯一變因
        deterministic=True,     # v2：同 seed 可重現(Ultralytics 預設即 True，明寫避免誤改)
        # 以下 augmentation 強度與 B1 pose 版完全一致，刻意不調整
        degrees=10,
        translate=0.05,
        scale=0.1,
        fliplr=0.0,     # 牙齒有生理方向性，且要與 pose 版保持一致，不做左右翻轉
        mosaic=0.3,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"找不到 {best_weights}，訓練可能中途失敗，請檢查 log")

    shutil.copy2(best_weights, ready_path)
    print(f"\n✅ seed {train_seed} 分割模型訓練完成，模型存成: {ready_path}")

    total_inst = int(poly_df["instance數"].sum()) if not poly_df.empty else 0
    val_metrics = extract_seg_map(model.trainer)
    summary_df = pd.DataFrame([{
        "train_seed": train_seed,
        "split_seed": SPLIT_SEED,
        "train張數": len(train_items),
        "val張數": len(val_items),
        "train_instance總數": total_inst,
        "每張平均instance數": round(poly_df["instance數"].mean(), 2) if not poly_df.empty else 0,
        "每張平均頂點數": round(poly_df["平均頂點數"].mean(), 1) if not poly_df.empty else 0,
        **val_metrics,
    }])

    summary_path = work_dir / "訓練摘要.xlsx"
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="訓練摘要", index=False)
        poly_df.to_excel(writer, sheet_name="train每張標註統計", index=False)
        pd.DataFrame({"val檔名": sorted(val_ids)}).to_excel(
            writer, sheet_name="val名單", index=False)

    print(f"📍 訓練摘要：{summary_path}")
    if val_metrics:
        print(f"   {val_metrics}")
    else:
        print(f"   ⚠️ 讀不到 val 集 Box/Mask mAP。")

    return ready_path, val_ids


def summarize_seeds(seeds):
    """讀各 seed 的 訓練摘要.xlsx，彙整 val 指標的 mean ± SD。"""
    rows = []
    for s in seeds:
        p = seed_work_dir(s) / "訓練摘要.xlsx"
        if not p.exists():
            print(f"⚠️ seed {s} 沒有 {p}(可能是從舊資料夾複製來的)，彙整略過")
            continue
        r = pd.read_excel(p, sheet_name="訓練摘要").iloc[0].to_dict()
        r["train_seed"] = s
        rows.append(r)
    if not rows:
        return
    sdf = pd.DataFrame(rows)
    metric_cols = [c for c in sdf.columns if c.startswith(("Box_", "Mask_"))]
    stat = pd.DataFrame({"mean": sdf[metric_cols].mean(),
                         "SD": sdf[metric_cols].std(ddof=1),
                         "n_seed": sdf[metric_cols].count()})
    with pd.ExcelWriter(SEED_SUMMARY_XLSX, engine="openpyxl") as w:
        sdf.to_excel(w, sheet_name="各seed", index=False)
        stat.to_excel(w, sheet_name="mean±SD")

    print(f"\n===== 跨 {len(sdf)} 個 seed(val 集) =====")
    for c in [c for c in metric_cols if c.startswith("Mask_")]:
        print(f"   {c}: {sdf[c].mean():.4f} ± {sdf[c].std(ddof=1):.4f}")
    print(f"📄 {SEED_SUMMARY_XLSX}")


def main():
    val_sets = {}
    for s in TRAIN_SEEDS:
        print(f"\n################ TRAIN_SEED = {s} ################")
        weights_path, val_ids = train_seg_model(s)
        val_sets[s] = val_ids
        print(f"📍 seed {s} 權重: {weights_path}")

    # 單一變因檢查：所有 seed 的 val 名單必須完全相同
    ref = val_sets[TRAIN_SEEDS[0]]
    bad = [s for s, v in val_sets.items() if v != ref]
    if bad:
        raise RuntimeError(f"❌ seed {bad} 的 val 名單與 seed {TRAIN_SEEDS[0]} 不同，"
                           f"切分不一致，結果不可比！")
    print(f"\n✅ {len(TRAIN_SEEDS)} 個 seed 的 val 名單完全一致({len(ref)} 張)")

    summarize_seeds(TRAIN_SEEDS)

    print(f"\n=======================================================")
    print(f"✨ YOLO11-seg 多 seed 訓練完成！權重在 {WORK_DIR}_seed*/weights_ready.pt")
    print(f"=======================================================")
    print(f"⚠️ val mAP 只能看訓練有沒有學起來；Mask mAP 高 ≠ 根尖畫得準。")
    print(f"   真正的指標是 B3→C1 的長度誤差，每個 seed 都要各跑一次。")
    print(f"👉 接下來：跑 B2_seg_inference_v4.py(會自動逐 seed 推論)。")


if __name__ == "__main__":
    main()
