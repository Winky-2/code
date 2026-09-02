"""
2b_train_yolo_pose.py
============================
對應新流程圖階段：（stage 2）訓練YOLO11-pose模型，定位A(切端)/B(根尖)兩個關鍵點
*** 單一模型、一般train/val切分版本(不做5-fold，比照stage 1的做法) ***

*** 這支的train資料從哪裡來 ***
不是舊pipeline的33張加星號test圖(那批要完整保留當最終held-out test，
完全不參與訓練)，而是 02_crop_new_dataset_for_pose_labeling.py 用stage 1
最終模型裁切出來、之後請學長姐/醫師另外標過A、B點的「新資料集」。
這樣33張test才能真正做到「模型從頭到尾沒看過」，最後拿來報告的
根管長度預測結果才有意義。

*** 為什麼用YOLO11-pose，不是純偵測(detection) ***
A/B點是「關鍵點定位」任務，不只要框出牙齒的bbox，還要標出牙齒內部
兩個特定位置(切端、根尖)，這正是pose任務的設計目的：
同時輸出bbox + N個關鍵點座標，跟純detection(只有bbox)不一樣。

*** 為什麼不做5-fold(比照stage 1的決定) ***
使用者已經確認這批新資料集的角色比較像「持續累積、擴充中」的訓練資料，
不是那種每一分都要斤斤計較、都要當過test的稀少樣本(那是舊的33張的
角色，這批新資料只需要一般train/val切分即可，val純粹拿來給
Ultralytics挑best.pt、看有沒有過擬合，不是正式報告用的指標。
之後真正的成績，是拿訓練好的pose模型去對「裁切+letterbox後的33張
test圖」跑推論，比對舊標註的A/B點，才是要報告的數字。

*** 標註格式需求 ***
POSE_DATA_DIR 底下需要是Roboflow匯出的YOLO-pose格式：
  - images/、labels/(或train/valid/test三個子資料夾各自有images/、labels/)
  - 每個labels/*.txt一行代表一顆牙："class cx cy w h kx1 ky1 v1 kx2 ky2 v2"
    (bbox 4個值 + 2個關鍵keypoint各3個值：x, y, 可見度v)
  - Roboflow匯出時記得選「YOLOv8 Pose」或相容的pose格式，
    並確認A、B兩個關鍵點的標註順序全資料集要一致
    (例如永遠A(切端)在前、B(根尖)在後)，順序不一致會讓模型學到錯誤的
    「第一個點/第二個點」語意。

*** 跟舊版02b_train_yolo_pose_v2.py的差異 ***
1. 基底模型從 yolov8n-pose.pt 換成 yolo11n-pose.pt。
2. 不做5-fold，改成跟stage 1一樣的一般train/val切分(只切一次)。
3. 加入之前在stage 1診斷出來、對小資料集訓練穩定性有幫助的參數
   (patience、lr0、amp、batch)，避免重蹈fitness collapse的問題。

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
POSE_DATA_DIR = "132-train-second-model.yolov8"          # Roboflow匯出的A/B關鍵點標註資料集
WORK_DIR = "yolo11_pose_run"                  # 訓練用資料/權重輸出的工作資料夾

BASE_MODEL = "yolo11n-pose.pt"                # 注意跟stage 1的yolo11n.pt不同，這是pose版本
EPOCHS = 150
IMG_SIZE = 640                                # 如果之後決定用非正方形letterbox，這裡建議跟那個尺寸對齊
VAL_RATIO = 0.2                              # 一般train/val切分比例(不是5-fold)
CLASS_NAMES = ["tooth"]                       # 單一類別，關鍵點語意由KPT_NAMES另外標示
KPT_SHAPE = [2, 3]                            # 2個關鍵點(A, B)，每個關鍵點3個值(x, y, 可見度v)
KPT_NAMES = ["A_incisal", "B_apex"]           # 只是方便自己看的註記，實際不影響訓練
RANDOM_SEED = 42

CONF_THRESHOLD = 0.15                         # 之後拿這個模型做推論時預設用的信心度門檻

# 小資料集訓練穩定性設定(跟stage 1診斷fitness collapse時學到的教訓一致)
PATIENCE = 0         # early stopping耐心值，0代表關閉；先給30，不要跟一開始一樣完全不管
LR0 = 0.001            # 初始學習率調低，避免小資料集訓練發散
AMP = False            # 關閉混合精度，排除AMP造成NaN導致fitness collapse的可能性
BATCH = 8              # 資料量小，batch別開太大

random.seed(RANDOM_SEED)


def restore_image_id(filename_str):
    """去掉Roboflow加的雜湊後綴，還原成乾淨的圖片ID(不含副檔名)。"""
    f_lower = filename_str.lower().strip()
    if "_jpg" in f_lower:
        return f_lower.split("_jpg")[0]
    elif "_png" in f_lower:
        return f_lower.split("_png")[0]
    return Path(f_lower).stem


def collect_pose_dataset():
    """相容Roboflow標準train/valid/test匯出，或扁平images/+labels/結構，
    全部pool起來，由這支腳本自己重新切一次train/val。"""
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
            img_id = restore_image_id(img_path.name)
            if img_id in seen_ids:
                print(f"⚠️ 圖片ID重複：'{img_id}'，已出現在 {seen_ids[img_id]}，這次({split_name})跳過")
                continue
            seen_ids[img_id] = split_name
            pool.append({"img_id": img_id, "img_path": img_path, "label_path": label_path})

    print(f"✅ 已標註A/B關鍵點的資料pool完成，總共 {len(pool)} 顆牙齒")
    if len(pool) < 20:
        print(f"⚠️ 目前只有 {len(pool)} 顆牙有標註，資料量偏少，訓練出來的pose模型")
        print(f"   可能不夠穩定，建議持續請學長姐/醫師標註更多裁切圖(來源見02_腳本輸出)。")
    return pool


def split_train_val(pool, val_ratio):
    shuffled = pool[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_items = shuffled[:n_val]
    train_items = shuffled[n_val:]
    print(f"📊 train/val切分：train {len(train_items)} 顆 / val {len(val_items)} 顆"
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
    """撈Box跟Pose兩種mAP，跟舊版02b_train_yolo_pose_v2.py的邏輯一樣。"""
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
        print(f"⚠️ 偵測到已經訓練過的pose模型 {ready_path}，直接使用，若要重新訓練請先刪除 {work_dir}")
        return ready_path

    pool = collect_pose_dataset()
    if not pool:
        raise FileNotFoundError(f"在 {POSE_DATA_DIR} 裡找不到任何已標註A/B點的資料")

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
        fliplr=0.0,     # A/B點是切端/根尖，跟牙齒方向有生理意義，預設不做左右翻轉
                        # 除非確認翻轉後A/B的語意仍然正確，否則不要打開
        mosaic=0.3,     # pose任務對mosaic比較敏感(關鍵點座標容易被裁切邊界影響)，強度調低一些
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"找不到 {best_weights}，訓練可能中途失敗，請檢查log")

    shutil.copy2(best_weights, ready_path)
    print(f"✅ pose模型訓練完成，模型存成: {ready_path}")

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
        print(f"   ⚠️ 這是val集(約{len(val_items)}顆牙)上的表現，只能看訓練有沒有基本學起來，")
        print(f"      不是正式要報告的指標——正式指標要看這個模型對33張test圖的推論結果。")

    return ready_path


def main():
    weights_path = train_pose_model()

    print(f"\n=======================================================")
    print(f"✨ YOLO11-pose (A/B關鍵點)模型訓練完成！")
    print(f"📍 權重路徑: {weights_path}")
    print(f"=======================================================")
    print(f"\n👉 接下來：拿這個模型去對「裁切+letterbox後的33張test圖」")
    print(f"   (01_resize_cropped_teeth_letterbox.py 的輸出)跑推論，")
    print(f"   取得每顆test牙齒的A/B預測座標，換算回像素距離，")
    print(f"   再接03b_merge_data.py + MATLAB ANN(像素轉毫米)那條路徑。")
    print(f"   這部分需要一支新的推論腳本，之後可以再幫你寫。")


if __name__ == "__main__":
    main()
