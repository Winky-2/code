"""
01c_augment_to_new_folder.py
============================
接在 01_check_dataset.py 之後（可跳過）、02_train_yolo_pose.py 之前執行（可選）。

跟 01b_augment_train_split.py 做的事一樣（擴增 train，valid/test 不動），
差別在於**輸出到全新獨立資料夾，完全不修改原始 Roboflow 下載資料**：

  roboflow_export/              <- 原始資料，完全不會被這支腳本碰到
    ├── train/  valid/  test/
    └── data.yaml

  roboflow_export_augmented/    <- 這支腳本產生的新資料夾
    ├── train/   <- 原始train的內容 + 擴增出來的圖片+標註
    ├── valid/   <- 從原始資料「複製」過來，內容不變，只是換位置
    ├── test/    <- 從原始資料「複製」過來，內容不變，只是換位置
    └── data.yaml

適合用在：想同時保留「有擴增」跟「沒擴增」兩個版本做對照實驗時。
兩邊 valid/test 完全一樣（只是複製，沒被擴增過），才能公平比較
「同一組評估資料下，train 有沒有擴增對結果的影響」。

要做對照實驗時：
  - 沒擴增那組：02_train_yolo_pose.py 的 ROBOFLOW_EXPORT_DIR 設成 "8-2.yolov8"
    （不用跑這支腳本，也不用跑 01b）
  - 有擴增那組：02_train_yolo_pose.py 的 ROBOFLOW_EXPORT_DIR 設成 "8-2.yolov8_augmented"
    （跑這支腳本產生這個資料夾）

*** 重要提醒 ***
02_train_yolo_pose.py 訓練時，YOLO 本身也會做即時 (on-the-fly) 增強
(degrees=15, translate=0.1, scale=0.2, mosaic=0.5...)。如果這裡先離線
擴增了，訓練時又疊加一層線上增強，等於雙重增強。建議二選一：
  (a) 用這支腳本離線擴增到位，然後把 02 的線上增強參數調小
      （例如 degrees=5~8, scale=0.1, mosaic=0）；或
  (b) 不用這支腳本，只依賴 02 的線上增強。

安裝需求：
    pip install albumentations opencv-python
"""

import random
import shutil
from pathlib import Path

import albumentations as A
import cv2

# ---------------- 設定區 ----------------
ROBOFLOW_EXPORT_DIR = "64-16-20_yolov8"             # 原始資料，不會被修改
OUTPUT_DIR = "64-16-20_yolov8_augmented"            # 擴增版輸出到這裡（全新資料夾）
TARGET_TRAIN_SIZE = 100                              # 只算 train/ 資料夾（對應流程圖：train擴增到100張）
KEYPOINT_NAMES = ["A", "B"]
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

TRANSFORM = A.Compose(
    [
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.9),
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)


def read_yolo_pose_label(label_path: Path, n_keypoints: int):
    with open(label_path, "r") as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = list(map(float, line.split()))
    class_id = int(parts[0])
    bbox = parts[1:5]
    kpts = []
    for i in range(n_keypoints):
        start = 5 + i * 3
        kpts.append(parts[start : start + 3])
    return class_id, bbox, kpts


def write_yolo_pose_label(label_path: Path, class_id, bbox, kpts):
    parts = [str(class_id)] + [f"{v:.6f}" for v in bbox]
    for kp in kpts:
        parts += [f"{kp[0]:.6f}", f"{kp[1]:.6f}", str(int(kp[2]))]
    with open(label_path, "w") as f:
        f.write(" ".join(parts) + "\n")


def augment_one_image(img_path: Path, label_path: Path, n_keypoints: int):
    parsed = read_yolo_pose_label(label_path, n_keypoints)
    if parsed is None:
        print(f"⚠️ 標註是空的，跳過：{label_path.name}")
        return None
    class_id, bbox, kpts = parsed

    image = cv2.imread(str(img_path))
    if image is None:
        print(f"⚠️ 讀不到圖片，跳過：{img_path.name}")
        return None
    h, w = image.shape[:2]

    kpts_px = [(kp[0] * w, kp[1] * h) for kp in kpts]
    visibilities = [kp[2] for kp in kpts]

    try:
        transformed = TRANSFORM(image=image, bboxes=[bbox], class_labels=[class_id], keypoints=kpts_px)
    except Exception as e:
        print(f"⚠️ 擴增失敗，跳過：{img_path.name}（{e}）")
        return None

    if len(transformed["bboxes"]) == 0 or len(transformed["keypoints"]) != n_keypoints:
        print(f"⚠️ 擴增後 bbox/關鍵點超出畫面，捨棄這次擴增：{img_path.name}")
        return None

    new_h, new_w = transformed["image"].shape[:2]
    new_kpts = [
        [transformed["keypoints"][i][0] / new_w, transformed["keypoints"][i][1] / new_h, visibilities[i]]
        for i in range(n_keypoints)
    ]
    new_bbox = list(transformed["bboxes"][0])

    return transformed["image"], class_id, new_bbox, new_kpts


def copy_split_as_is(src_dir: Path, dst_dir: Path, split_name: str):
    """valid/test 原封不動複製過去，不做任何擴增"""
    src_split = src_dir / split_name
    dst_split = dst_dir / split_name
    if not src_split.exists():
        print(f"⚠️ 找不到 {src_split}，略過")
        return
    if dst_split.exists():
        print(f"{split_name}/ 已存在於輸出資料夾，略過複製")
        return
    shutil.copytree(src_split, dst_split)
    n_img = len(list((dst_split / "images").glob("*.*")))
    print(f"{split_name}: 原封不動複製 {n_img} 張影像（不擴增）")


def prepare_augmented_train(src_dir: Path, dst_dir: Path, n_keypoints: int, target_count: int):
    """把 train 複製過去，再對複製後的版本做擴增（原始資料夾完全不動）"""
    src_train = src_dir / "train"
    dst_train = dst_dir / "train"

    if dst_train.exists():
        print("train/ 已存在於輸出資料夾，略過複製+擴增（如需重跑請先手動刪除該資料夾）")
        return

    shutil.copytree(src_train, dst_train)
    original_images = sorted((dst_train / "images").glob("*.*"))
    n_original = len(original_images)
    print(f"train: 複製 {n_original} 張原始影像到 {dst_train}")

    n_to_generate = max(0, target_count - n_original)
    print(f"目標 = {target_count}, 需額外生成 = {n_to_generate}")

    if n_to_generate == 0:
        print("已達目標張數，不需要擴增")
        return

    per_image = n_to_generate // n_original
    remainder = n_to_generate % n_original
    extra_targets = set(random.sample(range(n_original), remainder)) if remainder > 0 else set()

    lbl_dir = dst_train / "labels"
    img_dir = dst_train / "images"
    generated = 0
    for idx, img_path in enumerate(original_images):
        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        n_copies = per_image + (1 if idx in extra_targets else 0)
        for copy_i in range(n_copies):
            result = augment_one_image(img_path, label_path, n_keypoints)
            if result is None:
                continue
            aug_img, class_id, bbox, kpts = result

            aug_name = f"{img_path.stem}_aug{copy_i + 1}"
            out_img_path = img_dir / f"{aug_name}{img_path.suffix}"
            out_lbl_path = lbl_dir / f"{aug_name}.txt"

            cv2.imwrite(str(out_img_path), aug_img)
            write_yolo_pose_label(out_lbl_path, class_id, bbox, kpts)
            generated += 1

    print(f"✅ 實際生成 {generated} 張擴增影像，train/ 現在共 {n_original + generated} 張")


def copy_data_yaml(src_dir: Path, dst_dir: Path):
    for name in ["data.yaml", "data.yml"]:
        src_yaml = src_dir / name
        if src_yaml.exists():
            shutil.copy2(src_yaml, dst_dir / name)
            print(f"✅ 已複製 {name} 到擴增版資料夾（路徑結構跟原始資料一致，可直接使用）")
            return
    print("⚠️ 原始資料夾找不到 data.yaml，請自行確認擴增版資料夾裡的 data.yaml 是否需要手動建立")


def main():
    n_keypoints = len(KEYPOINT_NAMES)
    src_dir = Path(ROBOFLOW_EXPORT_DIR)
    dst_dir = Path(OUTPUT_DIR)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(f"目標：train 擴增到約 {TARGET_TRAIN_SIZE} 張")
    print(f"輸出到獨立資料夾：{dst_dir}（原始 {src_dir} 不會被修改）\n")

    prepare_augmented_train(src_dir, dst_dir, n_keypoints, TARGET_TRAIN_SIZE)
    copy_split_as_is(src_dir, dst_dir, "valid")
    copy_split_as_is(src_dir, dst_dir, "test")
    copy_data_yaml(src_dir, dst_dir)

    print(f"\n✅ 完成。要跑「有擴增」的實驗時，02_train_yolo_pose.py 的 "
          f"ROBOFLOW_EXPORT_DIR 請設成 \"{OUTPUT_DIR}\"")
    print(f"   要跑「沒擴增」的實驗時，維持設成 \"{ROBOFLOW_EXPORT_DIR}\" 即可，"
          f"完全不用跑這支腳本")


if __name__ == "__main__":
    main()
