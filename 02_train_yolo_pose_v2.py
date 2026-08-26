"""
02_train_yolo_pose.py
============================
對應流程圖階段：YOLO-Pose訓練 -> 驗證並選取最佳權重 -> 預測關鍵點(計算切端-根尖像素距離)

*** 改版說明：直接用 Roboflow 匯出資料夾裡現成的 data.yaml，不用另外產生 ***

用途：
用 Roboflow 匯出資料夾（裡面已經有 train/valid/test + 現成 data.yaml）
訓練一個 YOLO-Pose 模型（YOLO 訓練時會依 valid 指標自動存最佳權重 best.pt）。
訓練完成後，只用最終、完全沒看過的 test 資料夾做一次性預測，計算 A/B
兩點的像素距離。

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
- 如果有做過 01b_augment_train_split.py 離線擴增，這裡的 AUGMENT_PARAMS
  建議調小，避免雙重增強
"""

from pathlib import Path

import pandas as pd
from ultralytics import YOLO
import numpy as np
import os


# ---------------- 設定區 ----------------
ROBOFLOW_EXPORT_DIR = "8-2_yolov8_augmented"   # 底下應有 data.yaml + train/valid/test（Roboflow匯出時已依8:2切好train/valid，test另外held-out）
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
CONF_THRESHOLD = 0.25
TEST_OUTPUT_CSV = "test_keypoint_predictions.csv"

# 若已用 01b_augment_train_split.py 離線擴增過，建議把這幾個值調小，
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
        val=False,  # 👈 補上這行，強制關閉 YOLO 訓練時的驗證檢查
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
    """輔助函式：用來預測指定資料夾內的所有圖片並計算像素長度"""
    img_dir = Path(ROBOFLOW_EXPORT_DIR) / folder_name / "images"
    if not img_dir.exists():
        return []
    
    print(f"🔄 正在從 {folder_name} 資料夾提取 YOLO 關鍵點像素...")
    preds = model.predict(source=str(img_dir), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, save=False, verbose=False)
    rows = []
    for p in preds:
        img_name = Path(p.path).name
        if p.keypoints is None or len(p.keypoints.xy) == 0 or p.keypoints.xy is None:
            continue
        kpts = p.keypoints.xy[0].cpu().numpy()
        if len(kpts) < 2:
            continue
        
        # 計算 A, B 兩點的歐幾里得像素距離
        pixel_length = ((kpts[0][0] - kpts[1][0]) ** 2 + (kpts[0][1] - kpts[1][1]) ** 2) ** 0.5
        rows.append({
            "圖片檔名": img_name,
            "像素長度": round(pixel_length, 2) # 純數字，不帶 px，方便後續計算
        })
    return rows

def main():
    weights_path = train_model()
    model = YOLO(weights_path)
    
    # 1. 叫 YOLO 榨出所有圖片的像素距離
    all_yolo_rows = []
    all_yolo_rows.extend(extract_pixels_from_folder(model, "train"))
    all_yolo_rows.extend(extract_pixels_from_folder(model, "test"))
    df_yolo = pd.DataFrame(all_yolo_rows)
    
    if df_yolo.empty:
        print("❌ YOLO 沒有預測出任何像素，請檢查模型")
        return

    # 2. 讀取醫師的原始檔案
    doctor_excel_name = "根管充填長度_20260803.xlsx"
    if not os.path.exists(doctor_excel_name):
        raise FileNotFoundError(f"❌ 找不到醫師的原始檔案 '{doctor_excel_name}'！")
    
        # 1. 讀取完整的 Excel（先不跳行，讓 Python 自己找）
    df_raw = pd.read_excel(doctor_excel_name, header=None)
    
    # 2. 🔍 自動尋找哪一行包含「圖片檔名」
    header_row_idx = None
    for idx, row in df_raw.iterrows():
        # 用 fillna("") 把空值補成空字串，再強制轉成字串列表，確保不會有 float 亂入
        row_str = row.fillna("").astype(str).tolist()
        # 只要那一行同時出現這兩個關鍵字，它就是真正的表頭
        if any("圖片檔名" in item for item in row_str) and any("填充物長度" in item for item in row_str):
            header_row_idx = idx
            break

            
    if header_row_idx is None:
        raise ValueError("❌ 在 Excel 中找不到包含『圖片檔名』與『填充物長度』的欄位行，請檢查 Excel 表頭文字！")
        
    print(f"🔍 成功定位表頭！真正的欄位在第 {header_row_idx + 1} 行。")
    
    # 3. 重新以正確的表頭重新包裝資料
    df_doctor = pd.read_excel(doctor_excel_name, skiprows=header_row_idx)
    df_doctor.columns = df_doctor.columns.str.strip() # 清理空白
    
    # 4. 解決大小寫或括號可能的落差：只要欄位名稱包含關鍵字就強制更名
    rename_dict = {}
    for col in df_doctor.columns:
        if "圖片檔名" in col:
            rename_dict[col] = "圖片檔名"
        elif "填充物長度" in col:
            rename_dict[col] = "填充物長度(mm)"
            
    df_doctor = df_doctor.rename(columns=rename_dict)
    
    # 5. 安全過濾空白列
    df_doctor = df_doctor.dropna(subset=["圖片檔名", "填充物長度(mm)"])

    # 3. 🧠 自動與醫師資料合併，直接保留「圖片檔名」與「填充物長度(mm)」
    # merged_df = pd.merge(df_doctor, df_yolo, on="圖片檔名", how="inner")
    def restore_roboflow_name(filename_str):
        f_lower = filename_str.lower().strip()
        # 以 _jpg 或 _png 切開，並精準抓取前面最原始的編號部分 (例如 "001")
        if "_jpg" in f_lower:
            base_number = f_lower.split("_jpg")[0]
            return f"{base_number}.jpg"
        elif "_png" in f_lower:
            base_number = f_lower.split("_png")[0]
            return f"{base_number}.png"
        return f_lower

    # 在 YOLO 的名單中建立一個「還原後的檔名」欄位，用來跟醫生表格對接
    df_yolo["原始圖片檔名"] = df_yolo["圖片檔名"].apply(restore_roboflow_name)
    df_doctor["原始圖片檔名"] = df_doctor["圖片檔名"].astype(str).str.strip().str.lower()

    # 進行合併（用還原後的原始檔名來連連看！）
    merged_df = pd.merge(df_doctor, df_yolo, on="原始圖片檔名", how="inner")
    
    # 清理與還原欄位，維持畫面整潔並符合 MATLAB 期待
    if not merged_df.empty:
        merged_df = merged_df.drop(columns=["原始圖片檔名"])
        merged_df = merged_df.rename(columns={"圖片檔名_y": "YOLO增強檔名"})
        merged_df["圖片檔名"] = merged_df["圖片檔名_x"]
        merged_df = merged_df.drop(columns=["圖片檔名_x"])
    else:
        print("\n❌ 經過 Roboflow 檔名還原後依然配對失敗！請檢查下方範例：")
        print(f"YOLO 抓到的前 3 筆實體檔名: {df_yolo['圖片檔名'].head(3).tolist()}")
        print(f"還原後的檔名比對範例: {df_yolo['原始圖片檔名'].head(3).tolist()}")
        print(f"醫師表格內的前 3 筆檔名: {df_doctor['圖片檔名'].head(3).tolist()}")
        return

    # 4. 🚀 輸出成新的 Excel，這就是 MATLAB 接下來要讀取的輸入
    output_excel_path = "根管填充物像素長度_已配對.xlsx"
    merged_df.to_excel(output_excel_path, index=False)
    
    print(f"\n=======================================================")
    print(f"✨ Python 階段完成！已輸出帶有像素的新 Excel 檔：")
    print(f"📍 檔案名稱: {output_excel_path}")
    print(f"📊 總共成功對接了 {len(merged_df)} 筆完整資料")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
