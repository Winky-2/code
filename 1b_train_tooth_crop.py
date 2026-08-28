"""
01c_train_tooth_crop.py
============================
對應新流程圖階段：（新增前置階段）YOLOv11 全牙齒偵測 -> 裁切正式test集
*** 單一模型、單一檔案版本（不做5-fold）***

*** 為什麼可以這樣做 ***
這是「一般牙齒長什麼樣子」的偵測器，不是要拿去算根管長度的模型本身，
資料量也有165張，不像A/B關鍵點pipeline只有20~30顆牙那麼稀少、
需要斤斤計較每一分資料。用一般train/val切分、抓效果夠不夠好即可，
不需要像ANN迴歸那樣在意"每一張都要被當過test"。

*** 這支腳本做的事 ***
1. 讀取132張「全牙齒偵測」訓練資料，切一次train/val，訓練一個YOLOv11模型。
2. 對33張「加星號」的正式test圖跑推論，抓出每張圖裡「全部」偵測到的牙齒框。
3. 每張test圖，我們已經知道「目標牙(加星號)」大概在哪裡——因為這33張
   本來就是舊pipeline(01b_split_5folds.py之後)用來標A/B關鍵點的那批圖，
   牙醫已經在Roboflow上標過目標牙的bbox了(只是那是「單顆目標牙」的框，
   不是這次「全牙齒偵測」意義下的框)。
   拿「舊的目標牙bbox」當基準，跟這次偵測器抓到的所有框做IoU比對，
   IoU最高的那一個，就是我們要裁切的目標牙。
   *** 這是目前判斷「哪一顆是加星號目標牙」的建議做法 ***：
   不需要人工標記、不需要額外規則，直接重複利用你們已經有的舊標註即可。
   將來如果要接完全沒有舊標註的全新病例，這一步就沒有ground truth可比對，
   屆時需要另外的方法(例如另外訓練一個「目標牙分類器」，或請醫師在畫面上
   點一下星號位置)，但目前這33張用IoU比對是最省事、最不容易出錯的做法。
4. 選到目標框後加一點padding再裁切(避免A/B關鍵點剛好貼在框邊被切掉)，
   輸出裁切後的圖片，並輸出一份Excel紀錄裁切結果與品質檢查
   (IoU多高、有沒有偵測失敗、舊的A/B關鍵點是不是真的落在裁切範圍內)。
5. 這份裁切結果、Excel，就是之後(尚未實作)第二階段YOLO-pose要吃的輸入。

*** 重要假設（請依實際資料夾調整）***
- ALL_TEETH_EXPORT_DIR：Roboflow匯出的「全牙齒偵測」資料集
  (底下至少要有images/、labels/，labels是標準YOLO detection格式：
  一行一顆牙 "class cx cy w h"，同一張圖可以有很多行)。
  如果Roboflow匯出時已經自己切了train/valid/test，這裡會全部pool起來，
  由這支腳本自己重新切一次train/val，不沿用Roboflow原本切法
  (避免val那份混進了離線擴增過的圖，或跟舊pipeline的test重疊)。
- TARGET_TOOTH_TEST_DIR：33張加星號test圖所在的資料夾，格式比照舊有的
  01b_split_5folds.py讀的Roboflow匯出(images/ + labels/，YOLO pose格式：
  一行 "class cx cy w h kx1 ky1 v1 kx2 ky2 v2")。
- VAL_RATIO：訓練時切多少比例當val，預設0.15，純粹給Ultralytics挑
  best.pt用，跟最終要不要相信這個模型無關。
- CROP_PADDING_RATIO / IOU_MATCH_THRESHOLD：說明同上方案4。

安裝需求：
    pip install ultralytics pandas opencv-python pyyaml openpyxl
"""

import random
import shutil
from pathlib import Path

import cv2
import pandas as pd
import yaml
from ultralytics import YOLO

# ---------------- 設定區 ----------------
ALL_TEETH_EXPORT_DIR = "rename_train_yolov11"                # 👈 Roboflow「全牙齒偵測」匯出資料夾
TARGET_TOOTH_TEST_DIR = "33-all-test.yolov8/test"            # 👈 33張加星號test圖所在資料夾(images/+labels/)

WORK_DIR = "train-valid_teeth_yolov11"                        # 訓練用資料/權重輸出的工作資料夾
CROPPED_OUTPUT_DIR = "cropped_test_teeth"          # 裁切後圖片輸出處，之後餵給pose模型用
CROP_MANIFEST_XLSX = "33張test裁切結果.xlsx"

BASE_MODEL = "yolo11n.pt"                                 # YOLOv11 detection base model
EPOCHS = 150
IMG_SIZE = 640
VAL_RATIO = 0.2                                          # 一般train/val切分比例(不是5-fold)
CLASS_NAMES = ["tooth"]                                   # 單一類別，只框「這是不是牙齒」
RANDOM_SEED = 42
EXPECTED_TOTAL_IMAGES = 132                               # 只是防呆用的期望值，兜不起來只印警告

CONF_THRESHOLD = 0.15
CROP_PADDING_RATIO = 0.01
IOU_MATCH_THRESHOLD = 0.3

random.seed(RANDOM_SEED)


# ============================================================
# 第1部分：讀取165張訓練資料、切train/val、訓練YOLOv11偵測模型
# ============================================================

def restore_image_id(filename_str):
    """去掉Roboflow加的雜湊後綴，還原成乾淨的圖片ID(不含副檔名)。"""
    f_lower = filename_str.lower().strip()
    if "_jpg" in f_lower:
        return f_lower.split("_jpg")[0]
    elif "_png" in f_lower:
        return f_lower.split("_png")[0]
    return Path(f_lower).stem


def collect_all_teeth_images():
    """把Roboflow匯出裡可能存在的train/valid/test全部pool起來，
    回傳list of dict，之後由這支腳本自己重新切train/val，
    不沿用Roboflow原本的切法。"""
    pool = []
    seen_ids = {}
    for split_name in ("train", "valid", "test", "images"):
        # 相容兩種常見結構：Roboflow標準的 train/valid/test，
        # 或使用者可能只給一個扁平的 images/+labels/ 資料夾(此時split_name='images'
        # 只是嘗試直接讀 ALL_TEETH_EXPORT_DIR/images)
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
    """一般的隨機train/val切分(不是5-fold，只切一次)。"""
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
    }
    out_path = work_dir / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True)
    return out_path


def train_tooth_detector():
    """準備train/val資料夾 + 訓練一個YOLOv11偵測模型。
    如果之前已經訓練過(work_dir底下有weights_ready.pt)，直接沿用，
    不重新訓練——要重跑請自行刪除 WORK_DIR。"""
    work_dir = Path(WORK_DIR)
    ready_path = work_dir / "weights_ready.pt"
    if ready_path.exists():
        print(f"⚠️ 偵測到已經訓練過的模型 {ready_path}，直接使用，若要重新訓練請先刪除 {work_dir}")
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
        # 這裡用一般train/val訓練，跟A/B關鍵點pipeline的val=False不同：
        # 這個偵測器不是正式要報告的模型，用Ultralytics標準流程挑best.pt即可，不需要額外顧慮leakage。
        val=True,
        patience=0,         # 關閉early stopping，讓模型訓練到指定的epoch數
        lr0=0.001,          # 把初始學習率調低(預設AdamW通常是0.001~0.01，先降一個量級試試)
        amp=False,          # 關閉混合精度，先排除AMP造成NaN的可能性
        batch=8,            # 資料量小，batch別開太大(預設常是16)
        degrees=10,
        translate=0.1,
        scale=0.2,
        fliplr=0.5,
        mosaic=0.5,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"找不到 {best_weights}，訓練可能中途失敗，請檢查log")

    shutil.copy2(best_weights, ready_path)
    print(f"✅ 訓練完成，模型存成: {ready_path}")
    return ready_path


# ============================================================
# 第2部分：用訓練好的模型偵測 + IoU配對 + 裁切33張正式test圖
# ============================================================

def yolo_to_corners(cx, cy, w, h):
    """把YOLO正規化的(cx,cy,w,h)轉成(x1,y1,x2,y2)，仍在0~1正規化空間。"""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def compute_iou(boxA, boxB):
    """boxA, boxB皆為(x1,y1,x2,y2)正規化座標。"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def load_target_tooth_gt(label_path):
    """讀取舊pipeline(01b格式)的label：class cx cy w h kx1 ky1 v1 kx2 ky2 v2。
    只需要bbox跟兩個關鍵點座標(用來QC裁切範圍有沒有把A/B關鍵點包住)。"""
    with open(label_path, "r") as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = list(map(float, line.split()))
    bbox = parts[1:5]  # cx,cy,w,h
    kpts = []
    for i in range(2):  # A, B兩個關鍵點
        start = 5 + i * 3
        kpts.append((parts[start], parts[start + 1]))
    return bbox, kpts


def crop_and_save(img_path, chosen_box_corners, out_path, padding_ratio):
    """chosen_box_corners為正規化(x1,y1,x2,y2)，加padding後裁切成像素圖並存檔。
    回傳實際裁切的像素座標(x1,y1,x2,y2)，方便寫進Excel跟QC使用。"""
    image = cv2.imread(str(img_path))
    if image is None:
        return None
    h, w = image.shape[:2]

    x1, y1, x2, y2 = chosen_box_corners
    box_w = x2 - x1
    box_h = y2 - y1
    x1 -= box_w * padding_ratio
    x2 += box_w * padding_ratio
    y1 -= box_h * padding_ratio
    y2 += box_h * padding_ratio

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    if px2 <= px1 or py2 <= py1:
        return None

    cropped = image[py1:py2, px1:px2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cropped)
    return px1, py1, px2, py2


def process_test_images(model):
    img_dir = Path(TARGET_TOOTH_TEST_DIR) / "images"
    lbl_dir = Path(TARGET_TOOTH_TEST_DIR) / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {img_dir}，請確認 TARGET_TOOTH_TEST_DIR 設定正確")

    records = []
    out_root = Path(CROPPED_OUTPUT_DIR)

    img_paths = sorted(img_dir.glob("*.*"))
    print(f"\n=== 對 {len(img_paths)} 張正式test圖跑全牙齒偵測 + 目標牙配對裁切 ===")

    for img_path in img_paths:
        label_path = lbl_dir / (img_path.stem + ".txt")
        record = {"圖片檔名": img_path.name}

        if not label_path.exists():
            record.update({"狀態": "❌ 找不到舊的目標牙標註，無法比對，跳過"})
            records.append(record)
            continue

        gt = load_target_tooth_gt(label_path)
        if gt is None:
            record.update({"狀態": "❌ 舊標註是空的，跳過"})
            records.append(record)
            continue
        gt_bbox, gt_kpts = gt
        gt_corners = yolo_to_corners(*gt_bbox)

        preds = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                               save=False, verbose=False)
        pred = preds[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            record.update({
                "狀態": "⚠️ 偵測器完全沒抓到任何牙齒，退回用舊標註bbox直接裁切",
                "偵測到牙齒數": 0,
                "最佳IoU": None,
            })
            chosen_corners = gt_corners
        else:
            # pred.boxes.xyxyn：正規化座標(x1,y1,x2,y2)，直接跟gt_corners比對IoU
            candidate_corners = pred.boxes.xyxyn.cpu().numpy().tolist()
            ious = [compute_iou(gt_corners, tuple(c)) for c in candidate_corners]
            best_idx = max(range(len(ious)), key=lambda i: ious[i])
            iou_val = ious[best_idx]

            record["偵測到牙齒數"] = len(candidate_corners)
            record["最佳IoU"] = round(iou_val, 3)

            if iou_val < IOU_MATCH_THRESHOLD:
                record["狀態"] = (f"⚠️ 最佳IoU只有{iou_val:.3f}(<{IOU_MATCH_THRESHOLD})，"
                                  f"配對不可靠，退回用舊標註bbox裁切")
                chosen_corners = gt_corners
            else:
                record["狀態"] = "✅ 配對成功，使用偵測器框裁切"
                chosen_corners = tuple(candidate_corners[best_idx])

        out_path = out_root / img_path.name
        crop_px = crop_and_save(img_path, chosen_corners, out_path, CROP_PADDING_RATIO)

        if crop_px is None:
            record["狀態"] = record.get("狀態", "") + " | ❌ 裁切失敗(座標異常)"
            records.append(record)
            continue

        px1, py1, px2, py2 = crop_px
        record["裁切檔案路徑"] = str(out_path)
        record["裁切像素座標_x1y1x2y2"] = f"{px1},{py1},{px2},{py2}"

        # QC：舊的A/B關鍵點(正規化座標)是不是真的落在裁切範圍內
        kpts_ok = True
        for kx, ky in gt_kpts:
            if not (chosen_corners[0] - CROP_PADDING_RATIO <= kx <= chosen_corners[2] + CROP_PADDING_RATIO and
                    chosen_corners[1] - CROP_PADDING_RATIO <= ky <= chosen_corners[3] + CROP_PADDING_RATIO):
                kpts_ok = False
        record["AB關鍵點是否在裁切範圍內"] = "是" if kpts_ok else "❌否，請人工複查"

        records.append(record)

    df = pd.DataFrame(records)
    df.to_excel(CROP_MANIFEST_XLSX, index=False)

    status_col = df["狀態"] if "狀態" in df.columns else pd.Series(dtype=str)
    n_ok = status_col.str.startswith("✅").sum()
    n_fallback = status_col.str.contains("退回", na=False).sum()
    kpt_col = df["AB關鍵點是否在裁切範圍內"] if "AB關鍵點是否在裁切範圍內" in df.columns else pd.Series(dtype=str)
    n_bad_kpts = (kpt_col == "❌否，請人工複查").sum()

    print(f"\n=======================================================")
    print(f"✨ 裁切完成！共處理 {len(records)} 張test圖")
    print(f"   ✅ 配對成功並用偵測框裁切: {n_ok} 張")
    print(f"   ⚠️ 退回用舊標註bbox裁切(偵測失敗或IoU太低): {n_fallback} 張 👈 建議人工複查")
    print(f"   ❌ A/B關鍵點可能被裁到框外: {n_bad_kpts} 張 👈 務必人工複查，否則pose模型會看不到完整關鍵點")
    print(f"📍 裁切後圖片: {CROPPED_OUTPUT_DIR}/")
    print(f"📍 明細與品質檢查: {CROP_MANIFEST_XLSX}")
    print(f"=======================================================")


def main():
    weights_path = train_tooth_detector()
    model = YOLO(str(weights_path))
    process_test_images(model)

    print(f"\n👉 接下來：把 {CROPPED_OUTPUT_DIR}/ 裡的裁切圖片，")
    print(f"   拿去餵給第二階段的YOLO-pose模型(A/B關鍵點)，這部分之後再實作。")
    print(f"   建議先打開 {CROP_MANIFEST_XLSX}，人工複查標「請人工複查」的那幾列，")
    print(f"   確認裁切範圍真的有把整顆目標牙(含A/B點)包進去。")


if __name__ == "__main__":
    main()
