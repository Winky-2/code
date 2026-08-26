"""
01d_split_5folds.py
============================
接在 Roboflow 匯出之後、02訓練之前執行。

*** 為什麼需要這支新腳本 ***
老師希望「每一張圖都當過test」，這代表YOLO那一關也要做5-fold分組
交叉驗證，而不是像之前那樣只切一次固定的train/valid/test。

但Roboflow匯出時，train/valid/test比例(64/16/20)是寫死的，沒辦法
直接拿來做5-fold。而且如果直接對「已經擴增過」的01c輸出資料夾
(xxx_yolov8_augmented)做分組，同一顆牙的好幾個_aug版本會被誤判成
好幾筆「獨立資料」，分組時又會重演你們之前修過的資料洩漏問題。

所以這支腳本改成：
1. 讀取「原始、尚未離線擴增過」的Roboflow匯出資料夾
   (ROBOFLOW_EXPORT_DIR，不是xxx_augmented那份！)，把train/valid/test
   三個資料夾的圖片全部合併回一個池子。
2. 用還原後的原始檔名(去掉Roboflow加的雜湊後綴)當作「牙齒ID」，
   確認每顆牙只出現一次。
3. 用固定亂數種子把這些牙齒ID隨機分成 N_FOLDS 組(預設5組)。
4. 針對每一fold i：
     - 這一組 -> 當這一輪的test，原圖直接複製過去，不擴增
       (它是這一輪「held-out」的資料，YOLO完全沒看過)
     - 其他4組 -> 當這一輪的train，複製原圖後再做離線擴增，
       擴增邏輯、參數跟01_augment_to_new_folder.py完全一樣
     - 不切valid：呼應你們的決定，YOLO訓練本來就val=False，
       不靠valid指標選best.pt，5-fold下也不切，避免train portion
       又被瓜分掉一塊
   輸出到 OUTPUT_DIR/fold1 ~ fold5，每個fold底下都是一份完整、
   可以直接餵給 Ultralytics 的 train/test + data.yaml 資料夾結構

5. 額外輸出 fold_assignment.csv，記錄每顆牙屬於哪個fold。
   *** 這份對照表很關鍵 ***：02輸出的Excel會用同一套fold編號，
   之後MATLAB那邊做ANN的5-fold時也要沿用同一套分組，這樣才能保證
   「YOLO沒看過的牙齒」跟「ANN沒看過的牙齒」是同一批，整條pipeline
   才沒有任何一關偷看到不該看的資料。

安裝需求：
    pip install albumentations opencv-python pyyaml

執行前請確認：
- ROBOFLOW_EXPORT_DIR 指向的是「原始」匯出資料夾（底下train/labels
  裡沒有 _aug1, _aug2... 這種檔名），不是01c產生的augmented版本
"""

import csv
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import yaml

# ---------------- 設定區 ----------------
ROBOFLOW_EXPORT_DIR = "64-16-20_yolov8"        # 👈 原始、未離線擴增過的Roboflow匯出資料夾
OUTPUT_DIR = "64-16-20_yolov8_5fold"           # 產生 fold1~fold5 的地方
N_FOLDS = 5
TARGET_TRAIN_SIZE_PER_FOLD = 100               # 每個fold的train擴增到大約這個張數，跟01c原本的設定一致，可自行調整
KEYPOINT_NAMES = ["A", "B"]
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# 跟 01_augment_to_new_folder.py 完全相同的擴增設定，
# 確保5-fold版本跟原本單次切分版本的擴增強度一致，
# 之後比較「有無擴增」「單次切分 vs 5-fold」才公平。
TRANSFORM = A.Compose(
    [
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, fill=0, p=0.9),
        A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ],
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
)

def restore_tooth_id(filename_str):
    """把Roboflow加的雜湊後綴去掉，還原成牙齒的原始編號(不含副檔名)。
    跟 03_merge_data.py 的 restore_roboflow_name() 用同一套規則，
    這裡額外去掉副檔名，方便直接拿來當分組用的key。"""
    f_lower = filename_str.lower().strip()
    if "_jpg" in f_lower:
        return f_lower.split("_jpg")[0]
    elif "_png" in f_lower:
        return f_lower.split("_png")[0]
    return Path(f_lower).stem


def collect_all_images():
    """讀取原始(未擴增)Roboflow匯出資料夾的train/valid/test，合併成一個池子。
    回傳 list of dict: {"tooth_id":..., "img_path":..., "label_path":...}"""
    pool = []
    seen_tooth_ids = {}
    for split_name in ("train", "valid", "test"):
        img_dir = Path(ROBOFLOW_EXPORT_DIR) / split_name / "images"
        lbl_dir = Path(ROBOFLOW_EXPORT_DIR) / split_name / "labels"
        if not img_dir.exists():
            print(f"⚠️ 找不到 {img_dir}，略過")
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"⚠️ {img_path.name} 沒有對應的標註檔，跳過")
                continue
            tooth_id = restore_tooth_id(img_path.name)
            if tooth_id in seen_tooth_ids:
                # 理論上原始(未擴增)匯出資料夾裡，train/valid/test互斥，
                # 同一顆牙不應該出現兩次。如果出現，很可能是誤把
                # xxx_augmented資料夾當成來源，一定要先查清楚再繼續。
                raise ValueError(
                    f"❌ 牙齒ID重複：'{tooth_id}' 同時出現在 {seen_tooth_ids[tooth_id]} 跟 {split_name}！"
                    f"請確認 ROBOFLOW_EXPORT_DIR 是不是指到了已經離線擴增過的資料夾"
                    f"(應該指到沒有_aug版本的原始匯出資料夾)。"
                )
            seen_tooth_ids[tooth_id] = split_name
            pool.append({"tooth_id": tooth_id, "img_path": img_path, "label_path": label_path})

    print(f"✅ 合併完成，總共 {len(pool)} 顆牙齒(來自原始train/valid/test，未擴增)")
    return pool


def assign_folds(pool, n_folds):
    """把牙齒ID隨機分成n_folds組，回傳 {tooth_id: fold_index(0-based)}。
    先洗牌再用 i % n_folds 分配，確保每組數量最多只差1顆牙。"""
    tooth_ids = [item["tooth_id"] for item in pool]
    shuffled = tooth_ids[:]
    random.shuffle(shuffled)

    assignment = {}
    for i, tooth_id in enumerate(shuffled):
        assignment[tooth_id] = i % n_folds
    return assignment


def save_fold_assignment_csv(pool, assignment, output_dir):
    csv_path = Path(output_dir) / "fold_assignment.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["牙齒ID", "Fold"])
        for item in pool:
            writer.writerow([item["tooth_id"], assignment[item["tooth_id"]] + 1])
    print(f"📄 已輸出分組對照表：{csv_path}")
    print(f"   👉 之後MATLAB做ANN的5-fold時，請沿用這份對照表的分組，不要重新隨機切，")
    print(f"      否則YOLO沒看過的牙齒跟ANN沒看過的牙齒會對不起來。")


def copy_raw(items, dst_img_dir, dst_lbl_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        shutil.copy2(item["img_path"], dst_img_dir / item["img_path"].name)
        shutil.copy2(item["label_path"], dst_lbl_dir / item["label_path"].name)


def read_yolo_pose_label(label_path, n_keypoints):
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


def write_yolo_pose_label(label_path, class_id, bbox, kpts):
    parts = [str(class_id)] + [f"{v:.6f}" for v in bbox]
    for kp in kpts:
        parts += [f"{kp[0]:.6f}", f"{kp[1]:.6f}", str(int(kp[2]))]
    with open(label_path, "w") as f:
        f.write(" ".join(parts) + "\n")


def augment_one_image(img_path, label_path, n_keypoints):
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


def augment_train_folder(dst_img_dir, dst_lbl_dir, n_keypoints, target_count):
    """對已經複製好原圖的train資料夾做離線擴增到target_count張，
    邏輯跟01_augment_to_new_folder.py的prepare_augmented_train完全一樣，
    只是來源/目的地換成這個fold專屬的資料夾。"""
    original_images = sorted(dst_img_dir.glob("*.*"))
    n_original = len(original_images)

    n_to_generate = max(0, target_count - n_original)
    if n_to_generate == 0:
        print(f"   train已有{n_original}張，達到目標{target_count}張，不需要擴增")
        return

    per_image = n_to_generate // n_original
    remainder = n_to_generate % n_original
    extra_targets = set(random.sample(range(n_original), remainder)) if remainder > 0 else set()

    generated = 0
    for idx, img_path in enumerate(original_images):
        label_path = dst_lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        n_copies = per_image + (1 if idx in extra_targets else 0)
        for copy_i in range(n_copies):
            result = augment_one_image(img_path, label_path, n_keypoints)
            if result is None:
                continue
            aug_img, class_id, bbox, kpts = result
            aug_name = f"{img_path.stem}_aug{copy_i + 1}"
            cv2.imwrite(str(dst_img_dir / f"{aug_name}{img_path.suffix}"), aug_img)
            write_yolo_pose_label(dst_lbl_dir / f"{aug_name}.txt", class_id, bbox, kpts)
            generated += 1

    print(f"   ✅ 額外生成 {generated} 張擴增影像，train現在共 {n_original + generated} 張")


def write_fold_data_yaml(fold_dir, base_yaml_path):
    """每個fold都需要自己的data.yaml給Ultralytics讀。
    names/kpt_shape/nc等結構性設定直接沿用原始Roboflow匯出的data.yaml，
    只把train/val路徑換成這個fold自己的資料夾。
    註：val指到跟test一樣的路徑純粹是滿足Ultralytics格式需求，
    02訓練時是val=False，實際不會拿val集合做驗證或挑best.pt。"""
    with open(base_yaml_path, "r", encoding="utf-8") as f:
        base_yaml = yaml.safe_load(f)

    fold_yaml = dict(base_yaml)
    fold_yaml["path"] = str(fold_dir.resolve())
    fold_yaml["train"] = "train/images"
    fold_yaml["val"] = "test/images"   # 只是滿足格式需求，val=False時不會真的拿來驗證
    fold_yaml.pop("test", None)

    out_path = fold_dir / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fold_yaml, f, allow_unicode=True)


def main():
    n_keypoints = len(KEYPOINT_NAMES)
    base_yaml_path = Path(ROBOFLOW_EXPORT_DIR) / "data.yaml"
    if not base_yaml_path.exists():
        raise FileNotFoundError(f"找不到 {base_yaml_path}，請確認 ROBOFLOW_EXPORT_DIR 設定正確")

    output_root = Path(OUTPUT_DIR)
    if output_root.exists():
        print(f"⚠️ {output_root} 已存在，略過整支腳本(如需重跑請先手動刪除該資料夾)")
        return
    output_root.mkdir(parents=True)

    pool = collect_all_images()
    assignment = assign_folds(pool, N_FOLDS)
    save_fold_assignment_csv(pool, assignment, output_root)

    for fold_i in range(N_FOLDS):
        fold_dir = output_root / f"fold{fold_i + 1}"
        print(f"\n=== 產生 fold{fold_i + 1} ===")

        test_items = [item for item in pool if assignment[item["tooth_id"]] == fold_i]
        train_items = [item for item in pool if assignment[item["tooth_id"]] != fold_i]

        print(f"   train顆數(擴增前): {len(train_items)}　test顆數(這一fold的held-out): {len(test_items)}")

        copy_raw(train_items, fold_dir / "train" / "images", fold_dir / "train" / "labels")
        copy_raw(test_items, fold_dir / "test" / "images", fold_dir / "test" / "labels")

        augment_train_folder(fold_dir / "train" / "images", fold_dir / "train" / "labels",
                              n_keypoints, TARGET_TRAIN_SIZE_PER_FOLD)

        write_fold_data_yaml(fold_dir, base_yaml_path)

    print(f"\n✅ 5-fold資料夾已全部產生在：{output_root}")
    print(f"👉 接下來請把 02_train_yolo_pose_v4.py 改成迴圈跑 fold1~fold5(參考02_train_yolo_pose_v5.py)")


if __name__ == "__main__":
    main()
