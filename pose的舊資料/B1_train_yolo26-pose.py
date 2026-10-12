"""
對應新流程圖階段：（單階段版）直接在「完整X光片」上同時訓練
YOLO11-pose，一次輸出「每顆牙的bbox」+「A(切端)/B(根尖)關鍵點」，
取代原本 stage1(全牙齒偵測+裁切) + stage2(單顆牙pose) 的兩階段設計。

*** 單一class版本(配合Roboflow實際標註方式) ***
Roboflow標註只有一個class(tooth)，bbox+keypoint都在同一個class底下標，
沒有另外開target_tooth這個class。所以「目標牙 vs 一般牙」完全靠
keypoint visibility區分：
    - 目標牙(加星號那顆)：A、B兩點都標實際座標，v=2
    - 一般牙：keypoint留空/填0，v=0(未標註，不計入keypoint loss)

*** 這個做法有一個必須先知道的限制 ***
YOLO-pose的keypoint loss會用visibility把v=0的點從loss裡排除，但
「排除」只影響訓練時的梯度，不代表模型學到了「這個instance不該有
keypoint」這種語意——模型架構上，keypoint head是共用的，對每一個
偵測到的框都會吐出一組keypoint座標與信心度，包括那些訓練時v=0的
一般牙。也就是說：
    - 一般牙的keypoint預測基本上是「沒被監督過」的雜訊，數值不能用，
      但模型不會主動告訴你「這組沒意義」。
    - 因為沒有class訊號可以直接篩「這是不是目標牙」，2c推論端必須
      改用IoU比對舊標註bbox來選框(細節見2c_test_pose_and_measure_
      fullimage.py的select_target_instance())，這件事在單一class
      設計下是必要步驟，不是保險而已。
    - 這也代表：未來如果要接完全沒有舊標註的全新病例，IoU比對這條路
      會失效，屆時需要另一種認牙方式(例如醫師在畫面上點一下星號位置，
      或另外訓練一個「這是不是目標牙」的分類器)。這是單一class設計
      比兩個class設計多欠的一筆帳，先讓你們知道，之後上線前要補。

*** 標註格式需求 ***
POSE_DATA_DIR 底下需要是Roboflow匯出的YOLO-pose格式(完整X光片)：
  - images/、labels/(或train/valid/test三個子資料夾各自有images/、labels/)
  - 每個labels/*.txt「一張圖可以有很多行」，每行代表一顆牙：
        "class cx cy w h kx1 ky1 v1 kx2 ky2 v2"
    只有一個class(id=0, tooth)。
    目標牙(加星號)那行：A、B要標實際座標，v=2。
    一般牙那行：kx1 ky1 v1 kx2 ky2 v2填 "0 0 0 0 0 0"(v=0=未標註)。
    A、B標註順序全資料集要一致(永遠A(切端)在前、B(根尖)在後)。

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
POSE_DATA_DIR = "132-train-second-model_v2"    # 👈 Roboflow匯出：完整X光片，單一class + keypoint visibility標目標牙
WORK_DIR = "yolo26_pose_run-batch4"              # 訓練用資料/權重輸出的工作資料夾

BASE_MODEL = "yolo26n-pose.pt"
EPOCHS = 150

#    全片解析度通常比裁切後單顆牙圖大很多，如果沿用640，目標牙在畫面裡
#    會被壓縮到很小，關鍵點定位精度可能明顯下降。建議先確認你們X光片
#    的原始解析度，抓一個能讓目標牙細節不至於糊掉的imgsz(常見會抓
#    960或1280)，這裡先預設1280，訓練前務必依實際情況調整。
IMG_SIZE = 640

VAL_RATIO = 0.2                               # 一般train/val切分比例(不是5-fold)，比照stage1的做法
CLASS_NAMES = ["tooth"]                       # 單一class，目標牙/一般牙靠keypoint visibility區分
KPT_SHAPE = [2, 3]                            # 每個instance固定輸出2個keypoint slot(A, B)，
                                              # 目標牙v=2有實際座標，一般牙v=0未標註
KPT_NAMES = ["A_incisal", "B_apex"]           # 只是方便自己看的註記，實際不影響訓練
RANDOM_SEED = 42

CONF_THRESHOLD = 0.15                         # 之後拿這個模型做推論時預設用的信心度門檻

# 小資料集訓練穩定性設定(跟stage1診斷fitness collapse時學到的教訓一致)
PATIENCE = 0           # early stopping耐心值，0代表關閉
LR0 = 0.001             # 初始學習率調低，避免小資料集訓練發散
AMP = False             # 關閉混合精度，排除AMP造成NaN導致fitness collapse的可能性
BATCH = 4               # 資料量小，batch別開太大

random.seed(RANDOM_SEED)


def collect_pose_dataset():
    """相容Roboflow標準train/valid/test匯出，或扁平images/+labels/結構，
    全部pool起來，由這支腳本自己重新切一次train/val。全片模式下一張圖
    檔名本身就唯一，不需要像裁切圖那樣去雜湊還原ID。"""
    pool = []
    seen_ids = {}
    for split_name in ("train", "valid", "test", "images"):
        if split_name == "images":
            img_dir = Path(POSE_DATA_DIR) / "images"
            lbl_dir = Path(POSE_DATA_DIR) / "labels"
        else:
            img_dir = Path(POSE_DATA_DIR) / split_name / "images"
            lbl_dir = Path(POSE_DATA_DIR) / split_name / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"⚠️ {img_path.name} 沒有對應的標註檔，跳過")
                continue
            img_id = img_path.stem
            if img_id in seen_ids:
                print(f"⚠️ 圖片ID重複：'{img_id}'，已出現在 {seen_ids[img_id]}，這次({split_name})跳過")
                continue
            seen_ids[img_id] = split_name
            pool.append({"img_id": img_id, "img_path": img_path, "label_path": label_path})

    print(f"✅ 全片訓練資料pool完成，總共 {len(pool)} 張X光片")
    if len(pool) < 20:
        print(f"⚠️ 目前只有 {len(pool)} 張圖有標註，資料量偏少，訓練出來的模型")
        print(f"   可能不夠穩定，建議持續請學長姐/醫師標註更多X光片。")
    return pool


def split_train_val(pool, val_ratio):
    shuffled = pool[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    print(f"📊 train/val切分：train {len(train_items)} 張 / val {len(val_items)} 張"
          f"(val_ratio={val_ratio})")
    return train_items, val_items


def copy_items(items, dst_img_dir, dst_lbl_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        shutil.copy2(item["img_path"], dst_img_dir / item["img_path"].name)
        shutil.copy2(item["label_path"], dst_lbl_dir / item["label_path"].name)


def write_data_yaml(work_dir):
    data_yaml = {
        "path": str(work_dir.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
        "kpt_shape": KPT_SHAPE,
    }
    out_path = work_dir / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True)
    return out_path


def extract_pose_map(trainer):
    """撈Box跟Pose兩種mAP。注意：這個mAP是「所有牙齒(目標牙+一般牙)
    混在一起算」的整體表現，一般牙的keypoint訓練時v=0不計入loss，
    OKS計算時也會被忽略，所以這個mAP幾乎完全反映目標牙的表現沒錯——
    但兩階段版遇過的「OKS對非COCO關鍵點設定容易灌水」的問題依然存在
    (uniform sigma、bbox偏大會墊高OKS)，數字異常漂亮時不要照單全收，
    還是要以2c算出來的px/mm誤差為準。"""
    out = {}
    try:
        raw = getattr(trainer, "metrics", None) or {}
        for key, value in raw.items():
            if not key.startswith("metrics/"):
                continue
            name = key.replace("metrics/", "")
            if name.endswith("(B)"):
                out[f"Box_{name[:-3]}"] = round(float(value), 4)
            elif name.endswith("(P)"):
                out[f"Pose_{name[:-3]}"] = round(float(value), 4)
    except Exception as e:
        print(f"   ⚠️ 讀不到驗證指標({e})")
    return out


def train_pose_model():
    work_dir = Path(WORK_DIR)
    ready_path = work_dir / "weights_ready.pt"
    if ready_path.exists():
        print(f"⚠️ 偵測到已經訓練過的模型 {ready_path}，直接使用，若要重新訓練請先刪除 {work_dir}")
        return ready_path

    pool = collect_pose_dataset()
    if not pool:
        raise FileNotFoundError(f"在 {POSE_DATA_DIR} 裡找不到任何已標註的資料")

    train_items, val_items = split_train_val(pool, VAL_RATIO)

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
        name="ab_keypoint_pose",
        val=True,
        patience=PATIENCE,
        lr0=LR0,
        amp=AMP,
        batch=BATCH,
        degrees=10,
        translate=0.1,
        scale=0.2,
        fliplr=0.0,     # A/B點是切端/根尖，跟牙齒方向有生理意義，維持不做左右翻轉
        mosaic=0.3,     # 全片模式下比較不會像裁切模式那樣容易把目標牙keypoint裁到畫面外，
                        # 但先沿用偏保守的強度，訓練穩定後再視情況調高
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"找不到 {best_weights}，訓練可能中途失敗，請檢查log")

    shutil.copy2(best_weights, ready_path)
    print(f"✅ 全片單階段pose模型訓練完成，模型存成: {ready_path}")

    val_metrics = extract_pose_map(model.trainer)
    if val_metrics:
        summary_df = pd.DataFrame([{
            "train張數": len(train_items),
            "val張數": len(val_items),
            **val_metrics,
        }])
        summary_path = work_dir / "訓練摘要.xlsx"
        summary_df.to_excel(summary_path, index=False)
        print(f"📍 訓練摘要(val集上的Box/Pose mAP)：{summary_path}")
        print(f"   {val_metrics}")
        print(f"   ⚠️ 這是val集(約{len(val_items)}張)上的表現，只能看訓練有沒有基本學起來，")
        print(f"      不是正式要報告的指標——正式指標要看這個模型對33張test圖的推論結果(2c)。")

    return ready_path


def main():
    weights_path = train_pose_model()

    print(f"\n=======================================================")
    print(f"✨ 全片單階段YOLO11-pose(單一class，bbox+A/B關鍵點)模型訓練完成！")
    print(f"📍 權重路徑: {weights_path}")
    print(f"=======================================================")
    print(f"\n👉 接下來：拿這個模型直接對「33張完整X光片test圖」跑推論")
    print(f"   (不用再先裁切、也不用letterbox換算)。因為只有單一class，")
    print(f"   推論端要靠IoU比對舊標註bbox來挑出哪一個偵測框才是目標牙，")
    print(f"   細節見 2c_test_pose_and_measure.py。")


if __name__ == "__main__":
    main()
