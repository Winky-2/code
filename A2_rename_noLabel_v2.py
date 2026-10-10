"""
A2_rename_train_data_v2.py
============================
合併多個來源資料夾 + 全域流水號重新命名(只複製，不動原始檔)。

v2 變更(相對 v1)：
- 沒有對應 label 的圖片「也會 rename」，不再跳過。
- 來源資料夾沒有 labels/ 子資料夾也不再整個略過(視為全部無 label)。
- 無 label 圖片的處理由 UNLABELED_MODE 控制：
    "image_only" : 只複製圖片，不產生 .txt (預設；推論/待標註用)
    "empty_txt"  : 產生空白 .txt (YOLO 會當成「純背景、沒有任何物件」的負樣本)
                   ⚠️ 只有在你確定那張圖真的沒有目標物時才用，否則會教壞模型。
- 對照表多一欄「有標註」，原始標註檔名在無 label 時為空。
"""

import shutil
from pathlib import Path

import pandas as pd

# ---------------- 設定區 ----------------
SOURCE_DIRS = ["data set/Data_train"]

OUTPUT_DIR = "rename_train"
FILENAME_PREFIX = "train"
NUM_DIGITS = 4
START_INDEX = 1

MAPPING_CSV = "merge_rename_對照表.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

UNLABELED_MODE = "image_only"   # "image_only" 或 "empty_txt"


def collect_items_from_source(source_dir):
    """回傳 [(img_path, label_path 或 None), ...]；無 label 的圖也收。"""
    img_dir = Path(source_dir) / "images"
    lbl_dir = Path(source_dir) / "labels"

    if not img_dir.exists():
        print(f"⚠️ 找不到 {img_dir}，略過整個來源資料夾 '{source_dir}'")
        return []
    if not lbl_dir.exists():
        print(f"⚠️ 找不到 {lbl_dir}，'{source_dir}' 全部視為無標註")

    items, n_lbl = [], 0
    for img_path in sorted(img_dir.glob("*.*")):
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            items.append((img_path, label_path))
            n_lbl += 1
        else:
            items.append((img_path, None))

    print(f"✅ [{source_dir}] 圖片 {len(items)} 張（有標註 {n_lbl}、無標註 {len(items) - n_lbl}）")
    return items


def main():
    assert UNLABELED_MODE in ("image_only", "empty_txt"), "UNLABELED_MODE 設定錯誤"

    output_root = Path(OUTPUT_DIR)
    out_img_dir = output_root / "images"
    out_lbl_dir = output_root / "labels"

    if output_root.exists():
        print(f"⚠️ {output_root} 已存在，請先手動刪除再重跑（避免新舊編號混雜）。")
        return
    out_img_dir.mkdir(parents=True)
    out_lbl_dir.mkdir(parents=True)

    all_items = []
    for source_dir in SOURCE_DIRS:
        for img_path, label_path in collect_items_from_source(source_dir):
            all_items.append((source_dir, img_path, label_path))

    if not all_items:
        print("❌ 所有來源資料夾都沒有找到圖片，請檢查 SOURCE_DIRS")
        return

    print(f"\n=== 共 {len(all_items)} 張圖片，開始重新編號並複製到 {output_root} ===")

    records = []
    idx = START_INDEX
    for source_dir, img_path, label_path in all_items:
        new_stem = f"{FILENAME_PREFIX}_{idx:0{NUM_DIGITS}d}"
        shutil.copy2(img_path, out_img_dir / f"{new_stem}{img_path.suffix.lower()}")

        if label_path is not None:
            shutil.copy2(label_path, out_lbl_dir / f"{new_stem}.txt")
        elif UNLABELED_MODE == "empty_txt":
            (out_lbl_dir / f"{new_stem}.txt").touch()

        records.append({
            "新檔名": new_stem,
            "來源資料夾": source_dir,
            "原始圖片檔名": img_path.name,
            "原始標註檔名": label_path.name if label_path else "",
            "有標註": label_path is not None,
        })
        idx += 1

    df = pd.DataFrame(records)
    df.to_csv(MAPPING_CSV, index=False, encoding="utf-8-sig")

    n_lbl = int(df["有標註"].sum())
    print("\n=======================================================")
    print("✨ 合併＋重新編號完成！")
    print(f"📍 輸出: {output_root}/images, {output_root}/labels")
    print(f"📍 對照表: {MAPPING_CSV}")
    print("\n各來源資料夾張數：")
    print(df["來源資料夾"].value_counts().to_string())
    print(f"\n總計 {len(df)} 張（有標註 {n_lbl}、無標註 {len(df) - n_lbl}，"
          f"無標註處理方式 = {UNLABELED_MODE}）")
    print(f"新檔名範圍：{FILENAME_PREFIX}_{START_INDEX:0{NUM_DIGITS}d} ~ "
          f"{FILENAME_PREFIX}_{idx - 1:0{NUM_DIGITS}d}")
    print("=======================================================")


if __name__ == "__main__":
    main()
