"""
02_train_yolo_pose.py
============================
對應流程圖階段：YOLO-Pose訓練 -> 驗證並選取最佳權重 -> 預測關鍵點(計算切端-根尖像素距離)

*** 拆分說明：原本 02_train_yolo_pose_v3.py 訓練+推論+跟醫師資料合併全部
    混在一支裡，現在拆成兩支 ***
這支只負責「訓練 + 推論算像素距離」，輸出中繼檔案；
跟醫師 Excel 合併、抓漏的部分移到 03_merge_doctor_data.py。

拆開的好處：合併邏輯（例如檔名還原規則）之後要調整時，不用連訓練都重跑，
直接讀這支輸出的中繼檔案去跑 03 就好。

用途：
用 Roboflow 匯出資料夾（裡面已經有 train/valid/test + 現成 data.yaml）
訓練一個 YOLO-Pose 模型（YOLO 訓練時會依 valid 指標自動存最佳權重 best.pt）。
訓練完成後，對 train+test 資料夾做一次推論，計算 A/B 兩點的像素距離，
輸出成中繼檔案給下一支腳本使用。

*** test 只能在這裡被「用一次」***
如果之後又拿 test 的結果回頭調整模型架構/超參數/增強策略，再重跑一次
test 評估，test 就變相被拿來調參了，最終指標會失去「held-out」的意義。
如果要調參，請看訓練 log 裡 valid 的指標去調，test 留到真的要報告最終
結果時才跑。

安裝需求：
    pip install ultralytics pandas pyyaml

執行前請確認：
- ROBOFLOW_EXPORT_DIR 底下有 data.yaml（Roboflow 匯出 "YOLOv8 Pose" 格式
  時會自動附上），以及 train/valid/test 三個資料夾
- KEYPOINT_NAMES 順序跟 Roboflow 標註時 A、B 的順序一致
- 如果有做過 01b/01c 離線擴增，這裡的 AUGMENT_PARAMS 建議調小，
  避免雙重增強
"""

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ---------------- 設定區 ----------------
ROBOFLOW_EXPORT_DIR = "64-16-20_yolov8_augmented"   # 底下應有 data.yaml + train/valid/test
EPOCHS = 150
IMG_SIZE = 640

# 對應流程圖「選用 YOLO-Pose 模型 (v8 / v5)」：
# 小資料集下兩個版本都值得跑一次比較 valid 指標再決定，權重檔名只差在
# 版本代號，其餘訓練流程完全共用，切換時只需改這一行。
# YOLOv8-Pose: "yolov8n-pose.pt" / "yolov8s-pose.pt"
# YOLOv5-Pose (需另外用 ultralytics 支援的 v5 pose 權重): "yolov5s6-pose.pt" / "yolov5n6-pose.pt"
BASE_MODEL = "yolov8n-pose.pt"

OUTPUT_DIR = "yolo_runs"
# 建議 RUN_NAME 帶上 BASE_MODEL 資訊，方便 v8 vs v5 兩次訓練結果不互相覆蓋
RUN_NAME = f"single_run_{BASE_MODEL.replace('.pt', '')}"
KEYPOINT_NAMES = ["A", "B"]
CONF_THRESHOLD = 0.15

# 這支腳本輸出的中繼檔案，03_merge_doctor_data.py 會讀取這個檔案的兩個工作表
YOLO_OUTPUT_XLSX = "yolo像素預測.xlsx"   # 工作表：像素結果 / 偵測失敗清單

# 若已用 01b/01c 離線擴增過，建議把這幾個值調小，
# 避免離線擴增 + 線上即時增強疊加成雙重增強。
AUGMENT_PARAMS = dict(
    degrees=15,
    translate=0.1,
    scale=0.2,
    shear=0.0,
    fliplr=0.0,   # 若確認 A/B 點跟左右方向無關，可調成 0.5
    flipud=0.0,
    mosaic=0.5,
    mixup=0.0,
)


def train_model():
    data_yaml = str(Path(ROBOFLOW_EXPORT_DIR) / "data.yaml")
    model = YOLO(BASE_MODEL)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        optimizer="AdamW",
        project=OUTPUT_DIR,
        name=RUN_NAME,
        val=False,
        **AUGMENT_PARAMS,
    )

    # 不要自己組路徑字串去猜 best.pt 存在哪裡——不同版本的 Ultralytics
    # 對 project/name 的實際存檔路徑處理不一致（例如會自動包一層
    # runs/<task>/ 進去，或是資料夾已存在時自動加 -2、-3 後綴）。
    # 直接用訓練器回報的 save_dir 才是可靠的做法。
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"找不到 {best_weights}，訓練可能中途失敗，請檢查 log")

    print(f"✅ 訓練完成，最佳權重: {best_weights}")
    return str(best_weights)


def extract_pixels_from_folder(model, folder_name):
    """輔助函式：用來預測指定資料夾內的所有圖片並計算像素長度

    回傳 (rows, skipped)：
      rows    -> 成功算出像素長度的清單
      skipped -> 沒能算出像素長度的清單，每筆是 {"圖片檔名":..., "資料夾":..., "原因":...}，
                 方便追查到底是「YOLO完全沒偵測到」還是「只偵測到1個點」等等
    """
    img_dir = Path(ROBOFLOW_EXPORT_DIR) / folder_name / "images"
    if not img_dir.exists():
        return [], []

    print(f"🔄 正在從 {folder_name} 資料夾提取 YOLO 關鍵點像素...")
    preds = model.predict(source=str(img_dir), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, save=False, verbose=False)
    rows = []
    skipped = []
    for p in preds:
        img_name = Path(p.path).name
        if p.keypoints is None or p.keypoints.xy is None or len(p.keypoints.xy) == 0:
            skipped.append({"圖片檔名": img_name, "資料夾": folder_name, "原因": "完全沒偵測到任何關鍵點(信心度太低/沒偵測到牙齒)"})
            continue
        kpts = p.keypoints.xy[0].cpu().numpy()
        if len(kpts) < 2:
            skipped.append({"圖片檔名": img_name, "資料夾": folder_name, "原因": f"只偵測到{len(kpts)}個關鍵點(需要2個A/B點)"})
            continue

        # 計算 A, B 兩點的歐幾里得像素距離
        pixel_length = ((kpts[0][0] - kpts[1][0]) ** 2 + (kpts[0][1] - kpts[1][1]) ** 2) ** 0.5
        rows.append({
            "圖片檔名": img_name,
            "資料夾": folder_name,
            "像素長度": round(pixel_length, 2),  # 純數字，不帶 px，方便後續計算
        })
    return rows, skipped


def main():
    weights_path = train_model()
    model = YOLO(weights_path)

    all_yolo_rows = []
    all_yolo_skipped = []
    for split_name in ("train", "valid", "test"):
        rows, skipped = extract_pixels_from_folder(model, split_name)
        all_yolo_rows.extend(rows)
        all_yolo_skipped.extend(skipped)

    df_yolo = pd.DataFrame(all_yolo_rows)
    if df_yolo.empty:
        print("❌ YOLO 沒有預測出任何像素，請檢查模型")
        return

    if all_yolo_skipped:
        print(f"\n⚠️ 【YOLO關鍵點偵測階段】共有 {len(all_yolo_skipped)} 張圖片沒能算出像素長度：")
        for item in all_yolo_skipped:
            print(f"   - [{item['資料夾']}] {item['圖片檔名']} -> {item['原因']}")
    else:
        print(f"\n✅ 【YOLO關鍵點偵測階段】train+test 全部圖片都成功算出像素長度，共 {len(df_yolo)} 張")

    df_yolo.to_excel(YOLO_OUTPUT_XLSX, index=False, sheet_name="像素結果")
    df_skipped = pd.DataFrame(all_yolo_skipped)
    with pd.ExcelWriter(YOLO_OUTPUT_XLSX, engine="openpyxl", mode="a") as writer:
        df_skipped.to_excel(writer, index=False, sheet_name="偵測失敗清單")

    print(f"\n=======================================================")
    print(f"✨ YOLO 訓練+推論階段完成！")
    print(f"📍 結果檔案: {YOLO_OUTPUT_XLSX}")
    print(f"   - 工作表「像素結果」: {len(df_yolo)} 筆")
    print(f"   - 工作表「偵測失敗清單」: {len(df_skipped)} 筆")
    print(f"👉 接下來請執行 03_merge_doctor_data.py 跟醫師資料合併")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
