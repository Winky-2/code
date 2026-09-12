"""
B1s_train_yolo11-seg.py
===============================================================
方案B的第2步：訓練一個「獨立的」牙齒實例分割(instance segmentation)模型。

這支腳本跟 B1_train_yolo11-pose_v2.py 是平行關係，不是取代關係：
    B1  → pose 模型 → 預測 A(切端)/B(根尖) 關鍵點   [主線，不動]
    B1s → seg  模型 → 預測牙齒輪廓 mask             [輔助，本檔]
    B3  → 讀 B2 的關鍵點 CSV + 本模型的 mask，做防呆/投影/對照

刻意沿用 B1 的超參數(IMG_SIZE、BATCH、LR0、AMP、augmentation 強度、
RANDOM_SEED、VAL_RATIO)。小資料集上這些值是踩過坑調出來的，
這次實驗要驗證的是「分割有沒有用」，不是「超參數能不能再調」，
一次只動一個變因。

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
SEG_DATA_DIR = "train_segment"        # 👈 Roboflow Instance Segmentation 匯出資料夾
WORK_DIR = "yolo11_seg_run"      # 訓練用資料/權重輸出的工作資料夾

BASE_MODEL = "yolo11n-seg.pt"                     # 👈 跟 pose 版唯一的模型差異
EPOCHS = 150

# 以下刻意與 B1_train_yolo11-pose_v2.py 對齊，不要單獨調整
IMG_SIZE = 640
VAL_RATIO = 0.2
RANDOM_SEED = 42            # 同 seed + 同檔名池 → train/val 切分會與 pose 版一致
CLASS_NAMES = ["tooth"]     # A 案與 B 案都只用單一 class

PATIENCE = 0                # early stopping 關閉
LR0 = 0.001                 # 小資料集調低初始學習率
AMP = False                 # 關閉混合精度，排除 NaN 造成 fitness collapse
BATCH = 4

random.seed(RANDOM_SEED)


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
    shuffled = pool[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_items, train_items = shuffled[:n_val], shuffled[n_val:]
    print(f"📊 train/val 切分：train {len(train_items)} 張 / val {len(val_items)} 張"
          f"(val_ratio={val_ratio}, seed={RANDOM_SEED})")
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


def train_seg_model():
    work_dir = Path(WORK_DIR)
    ready_path = work_dir / "weights_ready.pt"
    if ready_path.exists():
        print(f"⚠️ 偵測到已訓練的模型 {ready_path}，直接使用。"
              f"若要重新訓練請先刪除 {work_dir}")
        return ready_path

    pool = collect_seg_dataset()
    if not pool:
        raise FileNotFoundError(f"在 {SEG_DATA_DIR} 裡找不到任何已標註的資料")

    train_items, val_items = split_train_val(pool, VAL_RATIO)

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
        project=WORK_DIR,
        name="tooth_seg",
        val=True,
        patience=PATIENCE,
        lr0=LR0,
        amp=AMP,
        batch=BATCH,
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
    print(f"\n✅ 分割模型訓練完成，模型存成: {ready_path}")

    total_inst = int(poly_df["instance數"].sum()) if not poly_df.empty else 0
    val_metrics = extract_seg_map(model.trainer)
    summary_df = pd.DataFrame([{
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

    print(f"📍 訓練摘要：{summary_path}")
    if val_metrics:
        print(f"   {val_metrics}")
        print(f"   ⚠️ 這是 val 集(約 {len(val_items)} 張)的表現，只能看訓練有沒有學起來。")
        print(f"      Mask mAP 高 ≠ 根尖畫得準(根尖只佔全牙面積極小比例)，")
        print(f"      真正的指標是 B3 對 33 張 test 圖算出來的長度誤差。")
    else:
        print(f"   ⚠️ 讀不到 val 集 Box/Mask mAP。")

    return ready_path


def main():
    weights_path = train_seg_model()

    print(f"\n=======================================================")
    print(f"✨ YOLO11-seg 牙齒分割模型訓練完成！")
    print(f"📍 權重路徑: {weights_path}")
    print(f"=======================================================")
    print(f"\n👉 接下來：把這個路徑填進 B3_mask_assist_and_compare.py 的 SEG_WEIGHTS，")
    print(f"   確認 B3 的 IMAGE_DIR 與 B2 的 IMAGE_DIR 是同一個資料夾，然後執行 B3。")
    print(f"   先打開 B3_mask視覺化/ 用肉眼確認 mask 輪廓有沒有套在正確的那顆牙上，")
    print(f"   再看「三法比較」工作表決定要不要升級成方案A。")


if __name__ == "__main__":
    main()
