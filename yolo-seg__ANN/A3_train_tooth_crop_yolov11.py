"""
A2a_train_tooth_detector_yolov11.py
============================
（由原 A2_train_tooth_crop_yolov11.py 拆出：只負責「訓練」）
流程：全牙齒偵測資料 pool -> 切一次 train/val -> 訓練 YOLOv11 偵測器
輸出：WORK_DIR/weights_ready.pt  ← A2b 會讀這個

單一模型、不做5-fold：這是「一般牙齒長什麼樣子」的偵測器，不是算根管長度的模型，
用一般 train/val 挑 best.pt 即可。

安裝需求：
    pip install ultralytics pyyaml
"""

import random
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

# ---------------- 設定區 ----------------
ALL_TEETH_EXPORT_DIR = "rename_train"          # 👈 Roboflow「全牙齒偵測」匯出資料夾
WORK_DIR = "train_crop_yolov11_run"                # 訓練資料/權重輸出(A2b 讀 WORK_DIR/weights_ready.pt)

BASE_MODEL = "yolo11n.pt"
EPOCHS = 150
IMG_SIZE = 640
VAL_RATIO = 0.2
CLASS_NAMES = ["tooth"]
RANDOM_SEED = 42
EXPECTED_TOTAL_IMAGES = 179                           # 防呆用，兜不起來只印警告

random.seed(RANDOM_SEED)


def restore_image_id(filename_str):
    """去掉Roboflow加的雜湊後綴，還原成乾淨的圖片ID(不含副檔名)。"""
    f_lower = filename_str.lower().strip()
    if "_jpg" in f_lower:
        return f_lower.split("_jpg")[0]
    elif "_png" in f_lower:
        return f_lower.split("_png")[0]
    return Path(f_lower).stem


def collect_all_teeth_images():
    """把Roboflow匯出的train/valid/test(或扁平images/+labels/)全部pool起來，由本腳本自己重切。"""
    pool = []
    seen_ids = {}
    for split_name in ("train", "valid", "test", "images"):
        if split_name == "images":
            img_dir = Path(ALL_TEETH_EXPORT_DIR) / "images"
            lbl_dir = Path(ALL_TEETH_EXPORT_DIR) / "labels"
        else:
            img_dir = Path(ALL_TEETH_EXPORT_DIR) / split_name / "images"
            lbl_dir = Path(ALL_TEETH_EXPORT_DIR) / split_name / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"⚠️ {img_path.name} 沒有對應的標註檔，跳過")
                continue
            img_id = restore_image_id(img_path.name)
            if img_id in seen_ids:
                print(f"⚠️ 圖片ID重複：'{img_id}'，已出現在 {seen_ids[img_id]}，這次({split_name})跳過")
                continue
            seen_ids[img_id] = split_name
            pool.append({"img_id": img_id, "img_path": img_path, "label_path": label_path})

    print(f"✅ 全牙齒偵測資料pool完成，總共 {len(pool)} 張X光圖")
    if len(pool) != EXPECTED_TOTAL_IMAGES:
        print(f"⚠️ 預期是 {EXPECTED_TOTAL_IMAGES} 張，實際pool出 {len(pool)} 張，"
              f"請確認 ALL_TEETH_EXPORT_DIR 有沒有指錯資料夾。")
    return pool


def split_train_val(pool, val_ratio):
    shuffled = pool[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    print(f"📊 train/val切分：train {len(train_items)} 張 / val {len(val_items)} 張(val_ratio={val_ratio})")
    return train_items, val_items


def copy_items(items, dst_img_dir, dst_lbl_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        shutil.copy2(item["img_path"], dst_img_dir / item["img_path"].name)
        shutil.copy2(item["label_path"], dst_lbl_dir / item["label_path"].name)


def write_data_yaml(data_root):
    data_yaml = {
        "path": str(data_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    out_path = data_root / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True)
    return out_path


def train_tooth_detector():
    """已有 weights_ready.pt 就直接沿用；要重訓請刪除 WORK_DIR。"""
    work_dir = Path(WORK_DIR)
    ready_path = work_dir / "weights_ready.pt"
    if ready_path.exists():
        print(f"⚠️ 偵測到已訓練模型 {ready_path}，直接使用；若要重新訓練請先刪除 {work_dir}")
        return ready_path

    pool = collect_all_teeth_images()
    if not pool:
        raise FileNotFoundError(f"在 {ALL_TEETH_EXPORT_DIR} 裡找不到任何可用的訓練圖片")

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
        name="tooth_detector",
        val=True,
        patience=0,
        lr0=0.001,
        amp=False,
        batch=8,
        degrees=10,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,
    )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"找不到 {best_weights}，訓練可能中途失敗，請檢查log")

    shutil.copy2(best_weights, ready_path)
    print(f"✅ 訓練完成，模型存成: {ready_path}")
    print(f"👉 接下來執行 A2b_crop_test_teeth_yolov11.py")
    return ready_path


if __name__ == "__main__":
    train_tooth_detector()
