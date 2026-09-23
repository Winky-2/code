"""
A4_crop_teeth_yolov11.py
============================
前置：先跑 A2a，產生 WEIGHTS_PATH
流程：對每張X光跑全牙齒偵測 -> label裡「每一顆」牙的GT框各自和偵測框做IoU配對
      -> 每顆牙裁成一張單獨圖片 -> 輸出Excel(QC)

label格式：YOLO detection，一行一顆牙 "class cx cy w h"（一張圖可多行）
輸出檔名：<原檔名>_t<第幾行>.<副檔名>，例如 IMG001_t0.jpg、IMG001_t1.jpg

安裝需求：
    pip install ultralytics pandas opencv-python openpyxl
"""

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ---------------- 設定區 ----------------
WEIGHTS_PATH = "train_crop_yolov11_run/weights_ready.pt"   # 👈 A2a 的輸出
INPUT_DIR = "rename_train"                         # 👈 要裁切的資料夾(images/ + labels/，detection格式)

CROPPED_OUTPUT_DIR = "cropped_teeth_yolov11"                  # 👈 裁切後單顆牙圖片
CROP_MANIFEST_XLSX = "裁切結果_yolov11.xlsx"                  # 👈 明細與QC

IMG_SIZE = 640
CONF_THRESHOLD = 0.15
CROP_PADDING_RATIO = 0.01
IOU_MATCH_THRESHOLD = 0.3


def yolo_to_corners(cx, cy, w, h):
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter_area
    return 0.0 if union <= 0 else inter_area / union


def load_det_labels(label_path):
    """讀取YOLO detection label，回傳 list of (x1,y1,x2,y2) 正規化座標，每行一顆牙。
    格式不是5個數字的行直接報錯，避免把pose/seg格式靜默讀錯。"""
    boxes = []
    with open(label_path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) != 5:
                raise ValueError(f"❌ {label_path.name} 第{line_no}行有{len(parts)}個數字，"
                                 f"不是detection格式(class cx cy w h)")
            boxes.append(yolo_to_corners(*parts[1:5]))
    return boxes


def crop_and_save(image, chosen_box_corners, out_path, padding_ratio):
    """回傳實際裁切像素座標(x1,y1,x2,y2)。"""
    h, w = image.shape[:2]

    x1, y1, x2, y2 = chosen_box_corners
    box_w, box_h = x2 - x1, y2 - y1
    x1 -= box_w * padding_ratio
    x2 += box_w * padding_ratio
    y1 -= box_h * padding_ratio
    y2 += box_h * padding_ratio
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))

    px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    if px2 <= px1 or py2 <= py1:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image[py1:py2, px1:px2])
    return px1, py1, px2, py2


def process_images(model):
    img_dir = Path(INPUT_DIR) / "images"
    lbl_dir = Path(INPUT_DIR) / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {img_dir}，請確認 INPUT_DIR")

    records = []
    out_root = Path(CROPPED_OUTPUT_DIR)
    img_paths = sorted(img_dir.glob("*.*"))
    print(f"\n=== 對 {len(img_paths)} 張圖跑全牙齒偵測 + 逐顆配對裁切 ===")

    for img_path in img_paths:
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            records.append({"圖片檔名": img_path.name, "狀態": "❌ 找不到標註檔，跳過"})
            continue

        gt_boxes = load_det_labels(label_path)
        if not gt_boxes:
            records.append({"圖片檔名": img_path.name, "狀態": "❌ 標註檔是空的，跳過"})
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            records.append({"圖片檔名": img_path.name, "狀態": "❌ 圖片讀取失敗，跳過"})
            continue

        pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                             save=False, verbose=False)[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            candidate_corners = []
        else:
            candidate_corners = [tuple(c) for c in pred.boxes.xyxyn.cpu().numpy().tolist()]

        for t_idx, gt_corners in enumerate(gt_boxes):
            record = {
                "圖片檔名": img_path.name,
                "牙齒編號(label第幾行,從0起)": t_idx,
                "該圖GT牙齒數": len(gt_boxes),
                "偵測到牙齒數": len(candidate_corners),
            }

            if not candidate_corners:
                record["最佳IoU"] = None
                record["狀態"] = "⚠️ 偵測器沒抓到任何牙齒，退回用GT框裁切"
                chosen_corners = gt_corners
            else:
                ious = [compute_iou(gt_corners, c) for c in candidate_corners]
                best_idx = max(range(len(ious)), key=lambda i: ious[i])
                iou_val = ious[best_idx]
                record["最佳IoU"] = round(iou_val, 3)
                if iou_val < IOU_MATCH_THRESHOLD:
                    record["狀態"] = (f"⚠️ 最佳IoU只有{iou_val:.3f}(<{IOU_MATCH_THRESHOLD})，"
                                      f"退回用GT框裁切")
                    chosen_corners = gt_corners
                else:
                    record["狀態"] = "✅ 配對成功，使用偵測器框裁切"
                    chosen_corners = candidate_corners[best_idx]

            out_path = out_root / f"{img_path.stem}_t{t_idx}{img_path.suffix}"
            crop_px = crop_and_save(image, chosen_corners, out_path, CROP_PADDING_RATIO)
            if crop_px is None:
                record["狀態"] += " | ❌ 裁切失敗(座標異常)"
                records.append(record)
                continue

            px1, py1, px2, py2 = crop_px
            record["裁切檔名"] = out_path.name
            record["裁切檔案路徑"] = str(out_path)
            # 裁切偏移，之後把裁切圖座標換回整張X光座標用：x_orig = x_crop + x1
            record["裁切像素座標_x1"] = px1
            record["裁切像素座標_y1"] = py1
            record["裁切像素座標_x2"] = px2
            record["裁切像素座標_y2"] = py2
            records.append(record)

    df = pd.DataFrame(records)
    df.to_excel(CROP_MANIFEST_XLSX, index=False)

    status_col = df["狀態"] if "狀態" in df.columns else pd.Series(dtype=str)
    n_crops = df["裁切檔名"].notna().sum() if "裁切檔名" in df.columns else 0
    print(f"\n=======================================================")
    print(f"✨ 裁切完成！{len(img_paths)} 張圖 → 共輸出 {n_crops} 張單顆牙圖")
    print(f"   ✅ 配對成功並用偵測框裁切: {status_col.str.startswith('✅').sum()} 顆")
    print(f"   ⚠️ 退回用GT框裁切: {status_col.str.contains('退回', na=False).sum()} 顆 👈 建議人工複查")
    print(f"   ❌ 跳過/失敗: {status_col.str.contains('❌', na=False).sum()} 列")
    print(f"📍 裁切後圖片: {CROPPED_OUTPUT_DIR}/")
    print(f"📍 明細與品質檢查: {CROP_MANIFEST_XLSX}")
    print(f"=======================================================")


def main():
    if not Path(WEIGHTS_PATH).exists():
        raise FileNotFoundError(f"❌ 找不到 {WEIGHTS_PATH}，請先跑 A2a_train_tooth_detector_yolov11.py")
    process_images(YOLO(WEIGHTS_PATH))


if __name__ == "__main__":
    main()
