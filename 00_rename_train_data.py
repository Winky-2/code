"""
00_prep_merge_renumber_teeth_data.py
============================
前置整理腳本：合併多個來源資料夾 + 重新編號，解決檔名衝突問題

*** 為什麼需要這支 ***
「全牙齒偵測」的165張train資料是從好幾個不同資料夾/專案收集來的，
各自的檔名編號系統不互通(例如兩個資料夾都各自有一張"023.jpg")。
如果直接把好幾個資料夾的images/labels複製到同一個資料夾，
檔名相同的檔案會互相覆蓋掉，或是被00_train_tooth_detector_and_crop_test.py
裡的重複檢查邏輯跳過，導致實際訓練用的圖片數量比預期少。

這支腳本做的事很單純：
1. 依序讀取每一個來源資料夾(SOURCE_DIRS)裡的 images/ + labels/。
2. 每張圖配對到它自己的標註檔(同檔名、副檔名.txt)。
3. 用一個「全域流水號」重新命名(例如 0001.jpg / 0001.txt、
   0002.jpg / 0002.txt ...)，不管原本的檔名是什麼，
   全部複製到同一個輸出資料夾，保證不會撞名。
4. 額外輸出一份對照表(CSV)，記錄「新檔名 對應到 哪個來源資料夾的哪個原始檔名」，
   之後如果要追溯某張圖原本是哪個資料夾來的，不會查無對證。

*** 這支腳本不會動到你的原始資料 ***
全部都是「複製」(shutil.copy2)，不是搬移或改名原始檔案，
原本的資料夾內容不會被更動，跑壞了大不了刪掉輸出資料夾重跑。

*** 使用方式 ***
1. 把下面 SOURCE_DIRS 改成你實際的來源資料夾清單，每個資料夾底下
   要有 images/ 和 labels/ 兩個子資料夾(標準YOLO detection格式：
   一行一顆牙 "class cx cy w h"，副檔名 .txt)。
2. OUTPUT_DIR 是合併後、重新編號好的輸出位置，也就是之後
   00_train_tooth_detector_and_crop_test.py 裡 ALL_TEETH_EXPORT_DIR
   要指的那個資料夾。
3. 執行完看終端機印出的統計數字，跟每個來源資料夾比對一下有沒有少算，
   也可以打開 MAPPING_CSV 抽查幾筆。

安裝需求：
    pip install pandas
"""

import shutil
from pathlib import Path

import pandas as pd

# ---------------- 設定區 ----------------
# 👇 改成你實際的來源資料夾清單，每個資料夾底下要有 images/ 和 labels/
SOURCE_DIRS = ["132-all-trian.yolov11/train",]

OUTPUT_DIR = "rename_train_yolov11"      # 合併＋重新編號後的輸出資料夾(之後接ALL_TEETH_EXPORT_DIR)
FILENAME_PREFIX = "train"              # 新檔名前綴，例如 tooth_0001.jpg
NUM_DIGITS = 4                         # 流水號補零位數，0001、0002...
START_INDEX = 1                        # 流水號起始值

MAPPING_CSV = "merge_rename_對照表.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_pairs_from_source(source_dir):
    """讀取單一來源資料夾的 images/ + labels/，回傳配對好的(img_path, label_path)清單。
    找不到對應標註檔的圖片會被跳過並印警告，不會讓整支腳本中斷。"""
    img_dir = Path(source_dir) / "images"
    lbl_dir = Path(source_dir) / "labels"

    if not img_dir.exists():
        print(f"⚠️ 找不到 {img_dir}，略過整個來源資料夾 '{source_dir}'")
        return []
    if not lbl_dir.exists():
        print(f"⚠️ 找不到 {lbl_dir}，略過整個來源資料夾 '{source_dir}'")
        return []

    pairs = []
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"   ⚠️ [{source_dir}] {img_path.name} 沒有對應的標註檔，跳過")
            continue
        pairs.append((img_path, label_path))

    print(f"✅ [{source_dir}] 找到 {len(pairs)} 組圖片+標註")
    return pairs


def main():
    output_root = Path(OUTPUT_DIR)
    out_img_dir = output_root / "images"
    out_lbl_dir = output_root / "labels"

    if output_root.exists():
        print(f"⚠️ {output_root} 已存在，為避免新舊檔案混在一起造成編號錯亂，"
              f"請先手動刪除這個資料夾再重跑本腳本。")
        return

    out_img_dir.mkdir(parents=True)
    out_lbl_dir.mkdir(parents=True)

    all_pairs = []  # list of (來源資料夾, img_path, label_path)
    for source_dir in SOURCE_DIRS:
        pairs = collect_pairs_from_source(source_dir)
        for img_path, label_path in pairs:
            all_pairs.append((source_dir, img_path, label_path))

    if not all_pairs:
        print("❌ 所有來源資料夾都沒有找到可用的圖片+標註組合，請檢查 SOURCE_DIRS 設定")
        return

    print(f"\n=== 共 {len(all_pairs)} 組圖片+標註，開始重新編號並複製到 {output_root} ===")

    mapping_records = []
    idx = START_INDEX
    for source_dir, img_path, label_path in all_pairs:
        new_stem = f"{FILENAME_PREFIX}_{idx:0{NUM_DIGITS}d}"
        new_img_name = f"{new_stem}{img_path.suffix.lower()}"
        new_lbl_name = f"{new_stem}.txt"

        shutil.copy2(img_path, out_img_dir / new_img_name)
        shutil.copy2(label_path, out_lbl_dir / new_lbl_name)

        mapping_records.append({
            "新檔名": new_stem,
            "來源資料夾": source_dir,
            "原始圖片檔名": img_path.name,
            "原始標註檔名": label_path.name,
        })
        idx += 1

    df = pd.DataFrame(mapping_records)
    df.to_csv(MAPPING_CSV, index=False, encoding="utf-8-sig")

    print(f"\n=======================================================")
    print(f"✨ 合併＋重新編號完成！")
    print(f"📍 輸出資料夾: {output_root}/images, {output_root}/labels")
    print(f"📍 對照表: {MAPPING_CSV}（可用來追溯每張圖原本來自哪個資料夾）")
    print(f"\n各來源資料夾實際貢獻的張數：")
    print(df["來源資料夾"].value_counts().to_string())
    print(f"\n總計 {len(df)} 張圖片，新檔名範圍："
          f"{FILENAME_PREFIX}_{START_INDEX:0{NUM_DIGITS}d} ~ "
          f"{FILENAME_PREFIX}_{idx - 1:0{NUM_DIGITS}d}")
    print(f"=======================================================")
    print(f"\n👉 接下來把 00_train_tooth_detector_and_crop_test.py 的")
    print(f"   ALL_TEETH_EXPORT_DIR 改成 \"{OUTPUT_DIR}\" 即可，")
    print(f"   這個資料夾已經是扁平的 images/ + labels/ 結構、檔名保證不衝突。")


if __name__ == "__main__":
    main()
