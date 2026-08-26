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

    # 順便把 Ultralytics 在訓練結尾那次驗證的指標撈出來。
    # 注意：這裡的 val 路徑就是這個 fold 的 test 資料夾(01b 寫 data.yaml 時指過去的)，
    # 而 val=False 代表訓練過程不靠它挑 best.pt，所以這組 mAP 是「模型沒看過的資料」
    # 上的成績，可以放心當成該 fold 的關鍵點偵測品質來看。
    val_metrics = extract_val_metrics(model.trainer)

    return str(best_weights.resolve()), val_metrics


def extract_val_metrics(trainer):
    """從 trainer.metrics 撈 Box / Pose 的 mAP。
    不同 Ultralytics 版本的 key 名稱會變，所以整段包在 try 裡，
    撈不到就回空 dict——這只是附帶資訊，不該讓整條 pipeline 掛掉。"""
    out = {}
    try:
        raw = getattr(trainer, "metrics", None) or {}
        for key, value in raw.items():
            if not key.startswith("metrics/"):
                continue
            name = key.replace("metrics/", "")
            # key 長得像 mAP50-95(B) / mAP50(P)，(B)=bounding box，(P)=pose 關鍵點
            if name.endswith("(B)"):
                out[f"Box_{name[:-3]}"] = round(float(value), 4)
            elif name.endswith("(P)"):
                out[f"Pose_{name[:-3]}"] = round(float(value), 4)
    except Exception as e:
        print(f"   ⚠️ 讀不到 fold 的驗證指標({e})，不影響像素預測，摘要表那幾欄會留空")
    return out


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


def count_images(img_dir: Path):
    """數資料夾裡有幾張圖，資料夾不存在就回 0（不讓摘要表因此中斷）。"""
    if not img_dir.exists():
        return 0
    return sum(1 for f in img_dir.glob("*.*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"})


def print_fold_summary(fold_records):
    """5折全部跑完後，把每個fold的最佳權重路徑集中列出來。
    訓練過程中每折雖然都會印一次「最佳權重: ...」，但那幾行會被
    Ultralytics 的訓練log淹掉，回頭要找某一折的權重很痛苦。
    這張摘要表在最後一次列齊，方便你複製路徑去做單張推論或重現結果。"""
    if not fold_records:
        return

    print(f"\n=======================================================")
    print(f"📋 各 fold 的最佳權重 (best.pt)")
    print(f"=======================================================")
    for rec in fold_records:
        print(f"\n  ▸ fold{rec['Fold']}")
        print(f"      權重      : {rec['最佳權重路徑']}")
        print(f"      訓練/驗證 : train {rec['訓練張數_含擴增']} 張(含離線擴增) / "
              f"held-out {rec['held_out張數']} 張")
        print(f"      像素預測  : 成功 {rec['成功算出像素']} 張，失敗 {rec['偵測失敗']} 張")

        # mAP 是附帶資訊，撈得到才印
        map_items = [(k, v) for k, v in rec.items() if k.startswith(("Box_", "Pose_"))]
        if map_items:
            metrics_str = "  ".join(f"{k}={v}" for k, v in map_items)
            print(f"      held-out  : {metrics_str}")

    missing = [rec["Fold"] for rec in fold_records if not Path(rec["最佳權重路徑"]).exists()]
    if missing:
        print(f"\n  ⚠️ fold{missing} 的權重檔現在找不到了，可能訓練後被移動或刪除。")

    print(f"\n  💡 這些 best.pt 是各折獨立訓練的成果，彼此不可混用："
          f"\n     每個模型只有在自己那一折的 test 上才算「沒看過的資料」，"
          f"\n     拿 fold1 的權重去預測 fold2 的 test 就等於偷看了訓練資料。")
    print(f"  💡 上面的 mAP 是各折在自己 held-out test 上的成績"
          f"(val=False，best.pt 不是靠它挑的，所以這組數字沒有偷看)。")


def main():
    fold_root = Path(FOLD_ROOT_DIR)
    if not fold_root.exists():
        raise FileNotFoundError(f"找不到 {fold_root}，請先執行 01d_split_5folds.py")

    all_yolo_rows = []
    all_yolo_skipped = []
    fold_records = []   # 每個fold的權重路徑與摘要，跑完後一次列出來

    for fold_i in range(N_FOLDS):
        fold_dir = fold_root / f"fold{fold_i + 1}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"找不到 {fold_dir}，請確認 01d_split_5folds.py 有成功產生全部5個fold")

        print(f"\n========== Fold {fold_i + 1}/{N_FOLDS} ==========")
        weights_path, val_metrics = train_one_fold(fold_dir, fold_i)
        model = YOLO(weights_path)

        rows, skipped = extract_pixels_from_fold_test(model, fold_dir, fold_i)
        all_yolo_rows.extend(rows)
        all_yolo_skipped.extend(skipped)

        record = {
            "Fold": fold_i + 1,
            "最佳權重路徑": weights_path,
            "訓練張數_含擴增": count_images(fold_dir / "train" / "images"),
            "held_out張數": count_images(fold_dir / "test" / "images"),
            "成功算出像素": len(rows),
            "偵測失敗": len(skipped),
        }
        record.update(val_metrics)
        fold_records.append(record)

    # 摘要要在「像素有沒有算出來」之前印：就算 5 折全部偵測失敗，
    # 權重路徑照樣要看得到，不然出問題時連要載哪個模型去 debug 都不知道。
    print_fold_summary(fold_records)

    df_yolo = pd.DataFrame(all_yolo_rows)
    if df_yolo.empty:
        print("\n❌ 5個fold加起來沒有預測出任何像素，請檢查模型"
              "(可以用上面列出的權重路徑手動載入來看看模型到底抓到什麼)")
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

    # 沒有失敗紀錄時也要給定欄位，否則寫出來是一張完全空白的工作表，
    # 03 讀「偵測失敗清單」時會拿到沒有「圖片檔名」欄的 DataFrame 而報錯。
    df_skipped = pd.DataFrame(all_yolo_skipped, columns=["圖片檔名", "Fold", "原因"])
    df_folds = pd.DataFrame(fold_records)

    df_yolo.to_excel(YOLO_OUTPUT_XLSX, index=False, sheet_name="像素結果")
    with pd.ExcelWriter(YOLO_OUTPUT_XLSX, engine="openpyxl", mode="a") as writer:
        df_skipped.to_excel(writer, index=False, sheet_name="偵測失敗清單")
        df_folds.to_excel(writer, index=False, sheet_name="各fold權重")

    print(f"\n=======================================================")
    print(f"✨ YOLO 5-fold訓練+推論階段完成！")
    print(f"📍 結果檔案: {YOLO_OUTPUT_XLSX}")
    print(f"   - 工作表「像素結果」: {len(df_yolo)} 筆(每顆牙恰好一筆，且都是該fold模型沒看過的資料)")
    print(f"   - 工作表「偵測失敗清單」: {len(df_skipped)} 筆")
    print(f"   - 工作表「各fold權重」: {len(df_folds)} 筆(每折的 best.pt 路徑與 held-out 表現)")
    print(f"👉 接下來請執行 03_merge_doctor_data.py 跟醫師資料合併(記得把YOLO_INPUT_XLSX")
    print(f"   改成讀這支輸出的「{YOLO_OUTPUT_XLSX}」，並保留「Fold」欄位帶到合併後的表)")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
