"""
03_merge_doctor_data.py
============================
對應流程圖階段：預測關鍵點(像素距離) -> 像素轉毫米 前置作業

*** 拆分說明：這支是從 02_train_yolo_pose_v3.py 拆出來的後半段 ***
負責讀取 02_train_yolo_pose.py 輸出的中繼檔案（YOLO 算出的像素長度），
跟醫師的原始 Excel（填充物長度）做檔名還原+合併，執行合併診斷、智慧抓漏，
輸出最終要餵給 MATLAB 的配對結果 Excel。

拆開的好處：這支幾乎是秒級完成的資料清理/合併工作，跟前面訓練+推論
（動輒跑幾十分鐘）完全脫鉤。之後醫師 Excel 更新、或檔名還原規則
(restore_roboflow_name) 要調整，只要重跑這支即可，不用重新訓練模型。

安裝需求：
    pip install pandas openpyxl

執行前請確認：
- 02_train_yolo_pose.py 已先跑過，PIXEL_INPUT_XLSX / SKIPPED_INPUT_XLSX
  兩個中繼檔案都存在
- DOCTOR_EXCEL_NAME 路徑正確
"""

import os

import pandas as pd

# ---------------- 設定區 ----------------
YOLO_INPUT_XLSX = "yolo像素預測.xlsx"   # 02 輸出的結果（含「像素結果」「偵測失敗清單」兩個工作表）
DOCTOR_EXCEL_NAME = "根管充填長度_20260803.xlsx"
OUTPUT_EXCEL_PATH = "根管填充物像素長度_已配對.xlsx"


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


def load_yolo_results():
    if not os.path.exists(YOLO_INPUT_XLSX):
        raise FileNotFoundError(f"❌ 找不到 {YOLO_INPUT_XLSX}，請先執行 02_train_yolo_pose.py")

    df_yolo = pd.read_excel(YOLO_INPUT_XLSX, sheet_name="像素結果")

    try:
        df_skipped = pd.read_excel(YOLO_INPUT_XLSX, sheet_name="偵測失敗清單")
        all_yolo_skipped = df_skipped.to_dict("records")
    except ValueError:
        print(f"⚠️ {YOLO_INPUT_XLSX} 裡沒有「偵測失敗清單」工作表，智慧抓漏會少一部分診斷資訊，但不影響合併本身")
        all_yolo_skipped = []

    return df_yolo, all_yolo_skipped


def load_doctor_excel():
    if not os.path.exists(DOCTOR_EXCEL_NAME):
        raise FileNotFoundError(f"❌ 找不到醫師的原始檔案 '{DOCTOR_EXCEL_NAME}'！")

    # 1. 讀取完整的 Excel（先不跳行，讓 Python 自己找）
    df_raw = pd.read_excel(DOCTOR_EXCEL_NAME, header=None)

    # 2. 🔍 自動尋找哪一行包含「圖片檔名」
    header_row_idx = None
    for idx, row in df_raw.iterrows():
        row_str = row.fillna("").astype(str).tolist()
        if any("圖片檔名" in item for item in row_str) and any("填充物長度" in item for item in row_str):
            header_row_idx = idx
            break

    if header_row_idx is None:
        raise ValueError("❌ 在 Excel 中找不到包含『圖片檔名』與『填充物長度』的欄位行，請檢查 Excel 表頭文字！")

    print(f"🔍 成功定位表頭！真正的欄位在第 {header_row_idx + 1} 行。")

    # 3. 重新以正確的表頭重新包裝資料
    df_doctor = pd.read_excel(DOCTOR_EXCEL_NAME, skiprows=header_row_idx)
    df_doctor.columns = df_doctor.columns.str.strip()

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

    return df_doctor, header_row_idx


def merge_yolo_and_doctor(df_doctor, df_yolo):
    df_yolo = df_yolo.copy()
    df_yolo["原始圖片檔名"] = df_yolo["圖片檔名"].apply(restore_roboflow_name)
    df_doctor = df_doctor.copy()
    df_doctor["原始圖片檔名"] = df_doctor["圖片檔名"].astype(str).str.strip().str.lower()

    merged_df = pd.merge(df_doctor, df_yolo, on="原始圖片檔名", how="inner")

    # --- 診斷用：檢查合併鍵是否有「一對多」的情況 ---
    n_unique_doctor_keys = df_doctor["原始圖片檔名"].nunique()
    n_unique_matched_keys = merged_df["原始圖片檔名"].nunique() if not merged_df.empty else 0
    key_counts = merged_df["原始圖片檔名"].value_counts() if not merged_df.empty else pd.Series(dtype=int)
    duplicated_keys = key_counts[key_counts > 1]

    print(f"\n🧪 【合併診斷】醫師表格唯一圖片數: {n_unique_doctor_keys}　"
          f"實際被配對到的唯一圖片數: {n_unique_matched_keys}　"
          f"合併後總列數(可能因重複key而膨脹): {len(merged_df)}")
    if len(duplicated_keys) > 0:
        print(f"⚠️ 有 {len(duplicated_keys)} 個原始檔名對應到不只一列(通常是同一張原圖的多個_aug版本都成功偵測到)：")
        for key, cnt in duplicated_keys.items():
            variants = merged_df.loc[merged_df["原始圖片檔名"] == key, "圖片檔名_y"].tolist()
            print(f"   - {key} 出現 {cnt} 次，對應的YOLO實體檔名: {variants}")
        print("   👉 這代表 merged_df 裡有重複列，建議之後只保留每個原始圖片一列(例如取第一筆或平均像素長度)，"
              "否則同一顆牙齒會在Excel裡被重複計算好幾次，影響MATLAB那邊的統計。")
    if n_unique_doctor_keys != n_unique_matched_keys:
        print(f"⚠️ 醫師表格有 {n_unique_doctor_keys} 張唯一圖片，但只有 {n_unique_matched_keys} 張真正配對成功，"
              f"代表有 {n_unique_doctor_keys - n_unique_matched_keys} 張圖(不論其aug版本)全部偵測失敗或檔名對不上，"
              f"下面的智慧抓漏會列出詳細名單。")

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

    return merged_df


def smart_leak_detection(merged_df, all_yolo_skipped, header_row_idx):
    df_orig_doctor = pd.read_excel(DOCTOR_EXCEL_NAME, skiprows=header_row_idx)
    orig_cols = [c for c in df_orig_doctor.columns if "圖片檔名" in c]

    if not orig_cols:
        return

    all_raw_doctor_pics = set(df_orig_doctor[orig_cols[0]].dropna().astype(str).str.strip().str.lower().tolist())
    all_raw_doctor_pics = {p for p in all_raw_doctor_pics if p.endswith((".jpg", ".png", ".jpeg"))}

    all_merged_cleaned = set(merged_df["圖片檔名"].astype(str).str.strip().str.lower().tolist())
    missing_pics = all_raw_doctor_pics - all_merged_cleaned

    if missing_pics:
        print(f"\n📢 【智慧抓漏成功！】一開始的 Excel 共有 {len(all_raw_doctor_pics)} 筆有效圖片紀錄")
        print(f"⚠️ 提示：其中有 {len(missing_pics)} 筆圖片未出現在最終的 {len(merged_df)} 筆配對中。")
        print(f"📍 沒能成功對接的原始圖片檔名為：{sorted(list(missing_pics))}")

        skipped_names_lower = {item["圖片檔名"].lower() for item in all_yolo_skipped}
        skipped_restored_lower = {restore_roboflow_name(item["圖片檔名"]) for item in all_yolo_skipped}

        print("\n🔎 逐筆診斷原因：")
        for pic in sorted(missing_pics):
            if pic in skipped_restored_lower or pic in skipped_names_lower:
                matched_reason = next(
                    (item["原因"] for item in all_yolo_skipped
                     if item["圖片檔名"].lower() == pic or restore_roboflow_name(item["圖片檔名"]) == pic),
                    "YOLO關鍵點偵測失敗"
                )
                print(f"   - {pic}：YOLO階段就抓不到 -> {matched_reason}")
            else:
                print(f"   - {pic}：YOLO有算出像素，但跟醫師檔名對不上（懷疑是 restore_roboflow_name() 的檔名還原規則沒對到，"
                      f"請檢查 Roboflow 匯出檔名格式是否跟預期的 '<編號>_jpg...' / '<編號>_png...' 不一致）")
        print("\n💡 備註：上面「YOLO階段就抓不到」的圖片代表模型信心度太低或沒偵測到牙齒；"
              "「檔名對不上」的圖片代表 YOLO 其實有結果，只是這支腳本的檔名還原邏輯沒能跟醫師表格的檔名配對成功，"
              "通常改一下 restore_roboflow_name() 就能救回來。")
    else:
        print(f"\n✅ 【智慧抓漏】一開始的 Excel 共有 {len(all_raw_doctor_pics)} 筆唯一有效圖片紀錄，"
              f"全部都有出現在最終配對結果裡，沒有「完全消失」的圖片。")
        print(f"👉 如果你看到最終列數({len(merged_df)})比預期的圖片數({len(all_raw_doctor_pics)})少，"
              f"請回頭看上面【合併診斷】那段：很可能不是「漏抓」，而是「重複的_aug版本互相抵銷/膨脹」"
              f"或是「醫師Excel本身的唯一圖片數就不是你預期的那個數字」，可以對照上面印出的唯一圖片數確認。")


def main():
    df_yolo, all_yolo_skipped = load_yolo_results()
    df_doctor, header_row_idx = load_doctor_excel()

    merged_df = merge_yolo_and_doctor(df_doctor, df_yolo)
    if merged_df.empty:
        return

    merged_df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    print(f"\n=======================================================")
    print(f"✨ 資料合併階段完成！已輸出帶有像素的新 Excel 檔：")
    print(f"📍 檔案名稱: {OUTPUT_EXCEL_PATH}")
    print(f"📊 總共成功對接了 {len(merged_df)} 筆完整資料")
    print(f"=======================================================")

    smart_leak_detection(merged_df, all_yolo_skipped, header_row_idx)


if __name__ == "__main__":
    main()
