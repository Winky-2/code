"""
02_train_yolo_pose_v5.py
============================
對應流程圖階段：YOLO-Pose訓練 -> 預測關鍵點(計算切端-根尖像素距離)
*** 5-fold交叉驗證版本 ***

*** 跟v4的差異 ***
v4是「訓練一次YOLO、用固定的train/valid/test資料夾」。
這一版改成：讀取 01d_split_5folds.py 產生的 fold1~fold5 資料夾，
每個fold各自訓練一個YOLO模型，只預測「該fold自己的test資料夾」
(也就是這個模型完全沒看過的牙齒)，5輪跑完後把5份test預測結果
合併成一份Excel。

*** 為什麼只預測test，不像v4一樣也預測train ***
5-fold的精神是：每一顆牙都恰好被某一個fold的模型當成「沒看過的
held-out資料」預測過一次。5輪合起來，每顆牙都有剛好一筆「out-of-fold
(沒被該模型訓練用過)」的像素長度，資料涵蓋率是100%，不需要
另外再跑train推論。下一支 03_merge_doctor_data.py 收到的就是
這份「每顆牙都只有一筆、且都是模型沒看過它時預測出來」的表。

*** Fold欄位很重要 ***
輸出Excel裡的「Fold」欄位，會沿用01d產生的fold_assignment.csv那套
分組。之後MATLAB做ANN的5-fold時，請直接讀這個「Fold」欄位來分組，
不要重新隨機切一次——這樣才能保證YOLO沒看過的牙齒，跟ANN沒看過的
牙齒，是同一批，整條pipeline才沒有任何一關偷看到不該看的資料。

*** test 只能被自己所屬的那個fold用一次 ***
如果之後又拿5-fold的整體結果回頭調整模型架構/超參數，再重跑一次
5-fold評估，就等於把這5折都拿來調參了，最終指標會失去「held-out」
的意義。如果要調參，請只看訓練log裡的loss曲線去調，5-fold的完整
評估留到真的要報告最終結果時才跑。

安裝需求：
    pip install ultralytics pandas pyyaml openpyxl

執行前請確認：
- 已經先執行過 01d_split_5folds.py，FOLD_ROOT_DIR 底下要有
  fold1 ~ fold5，每個裡面都有 data.yaml + train/test 資料夾
- KEYPOINT_NAMES 順序跟 Roboflow 標註時 A、B 的順序一致
"""

from pathlib import Path

import pandas as pd
from ultralytics import YOLO

# ---------------- 設定區 ----------------
FOLD_ROOT_DIR = "64-16-20_yolov8_5fold"   # 01d_split_5folds.py 的輸出資料夾，底下有fold1~fold5
N_FOLDS = 5
EPOCHS = 150
IMG_SIZE = 640

# 對應流程圖「選用 YOLO-Pose 模型 (v8 / v5)」
BASE_MODEL = "yolov8n-pose.pt"

OUTPUT_DIR = "yolo_runs"
KEYPOINT_NAMES = ["A", "B"]
CONF_THRESHOLD = 0.15

# 這支腳本輸出的中繼檔案，03_merge_doctor_data.py 會讀取這個檔案
YOLO_OUTPUT_XLSX = "yolo像素預測_5fold.xlsx"   # 工作表：像素結果 / 偵測失敗清單

# 5-fold下每個fold的train已經是大部分資料了，不再額外切valid，
# 呼應v4本來就val=False、不靠valid指標選best.pt的做法。
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


def train_one_fold(fold_dir: Path, fold_i: int):
    data_yaml = str(fold_dir / "data.yaml")
    run_name = f"fold{fold_i + 1}_{BASE_MODEL.replace('.pt', '')}"

    model = YOLO(BASE_MODEL)
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        optimizer="AdamW",
        project=OUTPUT_DIR,
        name=run_name,
        val=False,
        **AUGMENT_PARAMS,
    )

    # 跟v4一樣，不自己組路徑字串猜best.pt存哪，直接用訓練器回報的save_dir。
    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"找不到 {best_weights}，fold{fold_i + 1} 訓練可能中途失敗，請檢查 log")

    print(f"✅ fold{fold_i + 1} 訓練完成，最佳權重: {best_weights}")
    return str(best_weights)


def extract_pixels_from_fold_test(model, fold_dir: Path, fold_i: int):
    """只預測這個fold自己的test資料夾(該模型完全沒看過的牙齒)。
    回傳 (rows, skipped)，邏輯跟v4的extract_pixels_from_folder一樣，
    差別是多標註一個Fold欄位。"""
    img_dir = fold_dir / "test" / "images"
    if not img_dir.exists():
        return [], []

    print(f"🔄 fold{fold_i + 1}：正在預測held-out test資料夾的關鍵點...")
    preds = model.predict(source=str(img_dir), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, save=False, verbose=False)
    rows = []
    skipped = []
    for p in preds:
        img_name = Path(p.path).name
        if p.keypoints is None or p.keypoints.xy is None or len(p.keypoints.xy) == 0:
            skipped.append({
                "圖片檔名": img_name, "Fold": fold_i + 1,
                "原因": "完全沒偵測到任何關鍵點(信心度太低/沒偵測到牙齒)",
            })
            continue
        kpts = p.keypoints.xy[0].cpu().numpy()
        if len(kpts) < 2:
            skipped.append({
                "圖片檔名": img_name, "Fold": fold_i + 1,
                "原因": f"只偵測到{len(kpts)}個關鍵點(需要2個A/B點)",
            })
            continue

        pixel_length = ((kpts[0][0] - kpts[1][0]) ** 2 + (kpts[0][1] - kpts[1][1]) ** 2) ** 0.5
        rows.append({
            "圖片檔名": img_name,
            "Fold": fold_i + 1,
            "像素長度": round(pixel_length, 2),
        })
    return rows, skipped


def main():
    fold_root = Path(FOLD_ROOT_DIR)
    if not fold_root.exists():
        raise FileNotFoundError(f"找不到 {fold_root}，請先執行 01d_split_5folds.py")

    all_yolo_rows = []
    all_yolo_skipped = []

    for fold_i in range(N_FOLDS):
        fold_dir = fold_root / f"fold{fold_i + 1}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"找不到 {fold_dir}，請確認 01d_split_5folds.py 有成功產生全部5個fold")

        print(f"\n========== Fold {fold_i + 1}/{N_FOLDS} ==========")
        weights_path = train_one_fold(fold_dir, fold_i)
        model = YOLO(weights_path)

        rows, skipped = extract_pixels_from_fold_test(model, fold_dir, fold_i)
        all_yolo_rows.extend(rows)
        all_yolo_skipped.extend(skipped)

    df_yolo = pd.DataFrame(all_yolo_rows)
    if df_yolo.empty:
        print("❌ 5個fold加起來沒有預測出任何像素，請檢查模型")
        return

    if all_yolo_skipped:
        print(f"\n⚠️ 【YOLO關鍵點偵測階段】共有 {len(all_yolo_skipped)} 張圖片沒能算出像素長度：")
        for item in all_yolo_skipped:
            print(f"   - [fold{item['Fold']}] {item['圖片檔名']} -> {item['原因']}")
    else:
        print(f"\n✅ 【YOLO關鍵點偵測階段】5個fold的held-out test圖片全部成功算出像素長度，共 {len(df_yolo)} 張")

    # 保險檢查：5-fold合起來應該恰好覆蓋全部牙齒各一次，不該有重複檔名
    n_total = len(df_yolo)
    n_unique = df_yolo["圖片檔名"].nunique()
    if n_unique != n_total:
        print(f"⚠️ 警告：合併後有 {n_total} 列，但只有 {n_unique} 個唯一檔名，"
              f"代表同一張圖出現在不只一個fold的test裡，請檢查 01d_split_5folds.py 的分組是否互斥。")

    df_yolo.to_excel(YOLO_OUTPUT_XLSX, index=False, sheet_name="像素結果")
    df_skipped = pd.DataFrame(all_yolo_skipped)
    with pd.ExcelWriter(YOLO_OUTPUT_XLSX, engine="openpyxl", mode="a") as writer:
        df_skipped.to_excel(writer, index=False, sheet_name="偵測失敗清單")

    print(f"\n=======================================================")
    print(f"✨ YOLO 5-fold訓練+推論階段完成！")
    print(f"📍 結果檔案: {YOLO_OUTPUT_XLSX}")
    print(f"   - 工作表「像素結果」: {len(df_yolo)} 筆(每顆牙恰好一筆，且都是該fold模型沒看過的資料)")
    print(f"   - 工作表「偵測失敗清單」: {len(df_skipped)} 筆")
    print(f"👉 接下來請執行 03_merge_doctor_data.py 跟醫師資料合併(記得把YOLO_INPUT_XLSX")
    print(f"   改成讀這支輸出的「{YOLO_OUTPUT_XLSX}」，並保留「Fold」欄位帶到合併後的表)")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
