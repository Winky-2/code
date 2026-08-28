"""
2a_crop_for_pose_labeling.py
============================
對應新流程圖階段：（stage 2 資料準備）用stage 1的最終模型 -> 裁切「新資料集」的所有牙齒
用途：把這批裁切出來的單顆牙齒圖，拿去讓學長姐/醫師標A、B點，
標註完成後，才是 03_train_yolo_pose.py 真正要吃的train資料。

*** 跟00_train_tooth_detector_and_crop_test.py的差異 ***
00_那支是拿「已經有舊標註」的33張加星號test圖，用IoU比對找出「唯一那顆
目標牙」來裁切——因為那33張本來就有ground truth可以比對。
這支腳本處理的是**全新、完全沒有標註過**的資料集，沒有ground truth
可以比對，所以邏輯不一樣：**一張圖裡偵測到幾顆牙，就裁切幾顆**，
全部輸出成獨立的單顆牙齒圖，不挑選、不篩選「哪一顆是目標牙」，
因為這批資料的目的本來就是要拿去大量標註A/B點、擴充pose模型的
訓練資料，不是要算根管長度，不需要限定只留目標牙。

*** 這支腳本做的事 ***
1. 載入 00_train_tooth_detector_and_crop_test.py 訓練好的最終權重
   (預設路徑 all_teeth_yolov11_run/weights_ready.pt)。
2. 對 NEW_DATASET_DIR 底下的每一張圖跑推論，抓出這張圖裡「全部」
   偵測到的牙齒框(信心度 >= CONF_THRESHOLD 才算)。
3. 每一個偵測到的框都加一點padding後裁切成獨立圖片，輸出到
   OUTPUT_CROPPED_DIR，檔名格式：{原圖檔名}_tooth{編號}.jpg，
   方便之後回頭追溯這顆牙是從哪張原圖、哪個位置裁出來的。
4. 輸出一份Excel紀錄每一顆裁切牙齒的細節(信心度、bbox座標、裁切檔名)，
   並且列出「完全沒偵測到任何牙齒」的原圖清單，方便人工複查
   (可能是圖片本身有問題，或偵測器對這張圖的信心度不夠)。

*** 重要假設（請依實際資料夾調整）***
- FINAL_MODEL_WEIGHTS：指向00_那支腳本訓練完成的最終權重。
  如果你後來有重新命名WORK_DIR或搬動過weights_ready.pt，請對應調整。
- NEW_DATASET_DIR：新資料集所在的資料夾，預期是**扁平**結構，
  裡面直接放圖片(不需要labels/，因為這批資料還沒有任何標註)。
  如果你的新資料集是images/+labels/的結構(即使labels是空的)，
  這支腳本只會讀images/，labels/會被忽略。
- CONF_THRESHOLD / CROP_PADDING_RATIO：跟00_那支用同樣的邏輯與預設值，
  維持兩邊裁切風格一致，如果你發現這個信心度篩太嚴或太鬆(例如
  漏掉真的有的牙齒，或框到雜訊)，可以調整後再重跑。

安裝需求：
    pip install ultralytics pandas opencv-python openpyxl
"""

from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# ---------------- 設定區 ----------------
FINAL_MODEL_WEIGHTS = "train-valid_teeth_yolov11/weights_ready.pt"   # 👈 stage 1最終模型權重路徑

# 👇 新資料集圖片所在資料夾，可以放好幾個(扁平結構，每個資料夾底下直接是圖片，只需要圖片)
NEW_DATASET_DIRS = [
    "Data_first",
    "Data_second",
]

OUTPUT_CROPPED_DIR = "new_data_for_label"          # 裁切後輸出，之後拿去標註A/B點
MANIFEST_XLSX = "二階訓練集裁切結果.xlsx"
RENUMBER_MAPPING_CSV = "原圖重新編號對照表.csv"    # 原圖(來源資料夾+原始檔名) -> 新編號 的對照表
NUM_DIGITS = 4                                     # 新編號補零位數，0001、0002...

IMG_SIZE = 640
CONF_THRESHOLD = 0.15
CROP_PADDING_RATIO = 0.01

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def crop_one_box(image, box_corners_norm, padding_ratio):
    """box_corners_norm為正規化(x1,y1,x2,y2)，加padding後裁切成像素圖。
    回傳 (裁切後的圖, 實際裁切像素座標(x1,y1,x2,y2))，座標異常時回傳 (None, None)。"""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box_corners_norm

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
        return None, None

    return image[py1:py2, px1:px2], (px1, py1, px2, py2)


def collect_new_dataset_images():
    """把NEW_DATASET_DIRS裡每個資料夾的圖片都收集起來，回傳
    list of (folder_tag, img_path)。folder_tag只用來記錄「這張圖原本是
    哪個資料夾來的」，方便寫進Excel/對照表追溯，不會出現在輸出檔名裡
    (檔名衝突改用下面 assign_global_ids() 的全域流水號解決)。"""
    all_items = []
    seen_stems_global = {}   # stem -> 第一次出現的folder_tag，只用來印警告，不影響後續編號

    for i, folder in enumerate(NEW_DATASET_DIRS, start=1):
        folder_tag = f"src{i}"
        img_dir = Path(folder)
        candidate_dirs = [img_dir, img_dir / "images"]  # 相容有些人會多包一層images/子資料夾
        img_paths = []
        for d in candidate_dirs:
            if d.exists():
                img_paths = sorted([p for p in d.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS])
                if img_paths:
                    break

        if not img_paths:
            print(f"⚠️ 在 {folder}(含 images/ 子資料夾)裡找不到任何圖片，略過這個來源資料夾")
            continue

        print(f"✅ [{folder_tag}] {folder}：找到 {len(img_paths)} 張圖片")

        for p in img_paths:
            if p.stem in seen_stems_global:
                print(f"   ⚠️ 檔名 '{p.name}' 跟 {seen_stems_global[p.stem]} 裡的圖片撞名，"
                      f"稍後會用全域流水號重新編號，不會互相覆蓋")
            else:
                seen_stems_global[p.stem] = folder_tag
            all_items.append((folder_tag, p))

    return all_items


def assign_global_ids(dataset_items):
    """把彙整起來的所有圖片，依照(folder順序、資料夾內排序)給一個全域流水號，
    例如 0001、0002...，不管原始檔名是什麼、也不管來自哪個資料夾，
    新編號保證全域唯一，輸出檔名不會再需要靠來源前綴來避免撞名。
    回傳 dict：img_path -> 新編號字串(例如 "0001")，並把對照表存成CSV。"""
    mapping_records = []
    img_path_to_id = {}

    for idx, (folder_tag, img_path) in enumerate(dataset_items, start=1):
        new_id = f"{idx:0{NUM_DIGITS}d}"
        img_path_to_id[img_path] = new_id
        mapping_records.append({
            "新編號": new_id,
            "來源資料夾": folder_tag,
            "原始檔名": img_path.name,
            "原始完整路徑": str(img_path),
        })

    df_map = pd.DataFrame(mapping_records)
    df_map.to_csv(RENUMBER_MAPPING_CSV, index=False, encoding="utf-8-sig")
    print(f"📄 已輸出重新編號對照表：{RENUMBER_MAPPING_CSV}"
          f"（{len(mapping_records)} 張原圖，新編號範圍 0001~{len(mapping_records):0{NUM_DIGITS}d}）")

    return img_path_to_id


def process_new_dataset(model):
    dataset_items = collect_new_dataset_images()
    if not dataset_items:
        raise FileNotFoundError(f"❌ NEW_DATASET_DIRS 裡的所有資料夾都找不到圖片，請確認路徑設定正確")

    img_id_map = assign_global_ids(dataset_items)   # img_path -> "0001"這種全域流水號
    img_paths = [p for _, p in dataset_items]
    folder_tags = {p: tag for tag, p in dataset_items}

    print(f"\n✅ 總共 {len(img_paths)} 張新資料集圖片(來自 {len(NEW_DATASET_DIRS)} 個資料夾)，"
          f"開始跑全牙齒偵測+裁切")

    out_root = Path(OUTPUT_CROPPED_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    n_no_detection = 0
    n_total_teeth = 0

    for img_path in img_paths:
        folder_tag = folder_tags[img_path]
        new_id = img_id_map[img_path]
        preds = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                               save=False, verbose=False)
        pred = preds[0]

        if pred.boxes is None or len(pred.boxes) == 0:
            n_no_detection += 1
            records.append({
                "新編號": new_id,
                "來源資料夾": folder_tag,
                "原圖檔名": img_path.name,
                "牙齒編號": None,
                "狀態": "⚠️ 這張圖完全沒偵測到任何牙齒，請人工複查",
                "信心度": None,
                "裁切檔名": None,
            })
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ 讀不到圖片，跳過：[{folder_tag}] {img_path.name}")
            continue

        candidate_corners = pred.boxes.xyxyn.cpu().numpy().tolist()
        confidences = pred.boxes.conf.cpu().numpy().tolist()

        for tooth_i, (corners, conf) in enumerate(zip(candidate_corners, confidences), start=1):
            cropped, crop_px = crop_one_box(image, tuple(corners), CROP_PADDING_RATIO)
            if cropped is None:
                records.append({
                    "新編號": new_id,
                    "來源資料夾": folder_tag,
                    "原圖檔名": img_path.name,
                    "牙齒編號": tooth_i,
                    "狀態": "❌ 裁切失敗(座標異常)",
                    "信心度": round(conf, 3),
                    "裁切檔名": None,
                })
                continue

            # 檔名改用全域流水號(不再用來源前綴)，同一張原圖裡有多顆牙才用tooth編號區分
            crop_name = f"{new_id}_tooth{tooth_i:02d}{img_path.suffix.lower()}"
            out_path = out_root / crop_name
            cv2.imwrite(str(out_path), cropped)

            n_total_teeth += 1
            px1, py1, px2, py2 = crop_px
            records.append({
                "新編號": new_id,
                "來源資料夾": folder_tag,
                "原圖檔名": img_path.name,
                "牙齒編號": tooth_i,
                "狀態": "✅ 裁切成功",
                "信心度": round(conf, 3),
                "裁切檔名": crop_name,
                "裁切像素座標_x1y1x2y2": f"{px1},{py1},{px2},{py2}",
            })

    df = pd.DataFrame(records)
    df.to_excel(MANIFEST_XLSX, index=False)

    print(f"\n=======================================================")
    print(f"✨ 新資料集裁切完成！")
    print(f"📍 原圖張數：{len(img_paths)} 張")
    print(f"📍 成功裁切牙齒數：{n_total_teeth} 顆（這些會被拿去標A/B點，就是之後pose模型的候選train資料）")
    print(f"📍 完全沒偵測到牙齒的原圖：{n_no_detection} 張 👈 建議人工複查")
    print(f"📍 裁切後圖片：{OUTPUT_CROPPED_DIR}/")
    print(f"📍 明細：{MANIFEST_XLSX}")
    print(f"=======================================================")
    print(f"\n👉 接下來：把 {OUTPUT_CROPPED_DIR}/ 裡的圖片拿去標A、B兩個關鍵點")
    print(f"   (建議先過濾掉信心度太低、或明顯不是完整牙齒的裁切圖再送去標註，")
    print(f"    可以打開 {MANIFEST_XLSX} 依「信心度」欄位排序快速篩選)")
    print(f"   標註完成、匯出成YOLO-pose格式後，就可以接 03_train_yolo_pose.py 訓練了。")


def main():
    weights_path = Path(FINAL_MODEL_WEIGHTS)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到 {weights_path}，請確認00_train_tooth_detector_and_crop_test.py"
            f"已經訓練完成，且WORK_DIR/CROPPED相關路徑沒有被搬動過。"
        )
    model = YOLO(str(weights_path))
    process_new_dataset(model)


if __name__ == "__main__":
    main()
