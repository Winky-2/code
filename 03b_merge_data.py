"""
03_merge_data_v2_5fold.py
============================
對應流程圖階段：預測關鍵點(像素距離) -> 像素轉毫米 前置作業
*** 5-fold 交叉驗證版本（接 02_train_yolo_pose_v5.py）***

*** 跟舊版 03_merge_data.py 的差異 ***
舊版是接 v4（單次固定 train/valid/test 切分）的產物，YOLO 那份 Excel 裡
有「資料夾」欄位（train/valid/test），而且因為 train 資料夾裡含有離線擴增
的 _aug 版本，同一顆牙常常對應到好幾列，需要在合併階段處理膨脹問題。

這一版接 02_train_yolo_pose_v5.py 的輸出（yolo像素預測_5fold.xlsx）：
- 欄位變成「圖片檔名 / Fold / 像素長度」，沒有「資料夾」欄位
- 每顆牙恰好一列，而且那一列都是「該 fold 的模型沒看過它時」預測出來的
  （01b 產生 fold 時，test 資料夾是直接複製原圖、不擴增，所以不會有 _aug 版本）
- 「Fold」欄位一路帶到最終 Excel，MATLAB 端的 ANN 5-fold 要直接讀這欄分組，
  不可以重新隨機切，否則 YOLO 沒看過的牙齒跟 ANN 沒看過的牙齒會對不起來

*** 額外新增的檢查 ***
1. 唯一性檢查：合併後每顆牙應該只有一列。若有重複，代表 01b 的分組不互斥
   或來源資料夾拿錯（例如誤用已擴增的資料夾），會明確示警。
2. 涵蓋率檢查：5 個 fold 加起來應該覆蓋醫師 Excel 裡全部的牙齒，一顆不漏。
3. Fold 對照檢查（選用）：如果找得到 01b 輸出的 fold_assignment.csv，
   會比對「Excel 裡的 Fold」跟「當初分組表的 Fold」是否一致，
   避免中途重跑 01b、換了亂數種子而導致兩邊對不起來。

安裝需求：
    pip install pandas openpyxl

執行前請確認：
- 02_train_yolo_pose_v5.py 已先跑過，YOLO_INPUT_XLSX 存在
- DOCTOR_EXCEL_NAME 路徑正確
- FOLD_ASSIGNMENT_CSV 指向 01b_split_5folds.py 產生的那份對照表（找不到會自動略過檢查）
"""

import os

import pandas as pd

# ---------------- 設定區 ----------------
YOLO_INPUT_XLSX = "yolo像素預測_5fold.xlsx"        # 02_train_yolo_pose_v5.py 的輸出
DOCTOR_EXCEL_NAME = "根管充填長度_20260803.xlsx"
FOLD_ASSIGNMENT_CSV = "64-16-20_yolov8_5fold/fold_assignment.csv"   # 01b 的輸出，找不到就略過檢查
OUTPUT_EXCEL_PATH = "根管填充物像素長度_已配對_5fold.xlsx"

N_FOLDS = 5

# 理論上 5-fold 下每顆牙只會有一列。萬一真的出現重複（分組沒互斥），
# True = 保留第一筆並示警後繼續；False = 直接中止，逼你回頭修 01b。
KEEP_FIRST_ON_DUPLICATE = True


def restore_roboflow_name(filename_str):
    """把 Roboflow 加的雜湊後綴去掉，還原成醫師表格裡的原始檔名。
    規則跟 01b_split_5folds.py 的 restore_tooth_id() 一致（那邊多去掉副檔名）。"""
    f_lower = str(filename_str).lower().strip()
    if "_jpg" in f_lower:
        return f"{f_lower.split('_jpg')[0]}.jpg"
    elif "_png" in f_lower:
        return f"{f_lower.split('_png')[0]}.png"
    return f_lower


def load_yolo_results():
    if not os.path.exists(YOLO_INPUT_XLSX):
        raise FileNotFoundError(
            f"❌ 找不到 {YOLO_INPUT_XLSX}，請先執行 02_train_yolo_pose_v5.py。"
            f"（如果你跑的是舊的 v4 單次切分版本，請改用舊版 03_merge_data.py）"
        )

    df_yolo = pd.read_excel(YOLO_INPUT_XLSX, sheet_name="像素結果")

    required_cols = {"圖片檔名", "Fold", "像素長度"}
    missing = required_cols - set(df_yolo.columns)
    if missing:
        raise ValueError(
            f"❌ {YOLO_INPUT_XLSX} 的「像素結果」工作表缺少欄位 {sorted(missing)}。"
            f"目前欄位是 {list(df_yolo.columns)}。"
            f"這支腳本是給 5-fold 版本（v5）用的，若你讀到的是 v4 的輸出"
            f"（欄位為「資料夾」），請改用舊版 03_merge_data.py。"
        )

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

    # 1. 先整張讀進來（不指定表頭），讓程式自己找表頭在第幾行
    df_raw = pd.read_excel(DOCTOR_EXCEL_NAME, header=None)

    # 2. 🔍 自動尋找同時包含「圖片檔名」與「填充物長度」的那一行
    header_row_idx = None
    for idx, row in df_raw.iterrows():
        row_str = row.fillna("").astype(str).tolist()
        if any("圖片檔名" in item for item in row_str) and any("填充物長度" in item for item in row_str):
            header_row_idx = idx
            break

    if header_row_idx is None:
        raise ValueError("❌ 在 Excel 中找不到包含『圖片檔名』與『填充物長度』的欄位行，請檢查 Excel 表頭文字！")

    print(f"🔍 成功定位表頭！真正的欄位在第 {header_row_idx + 1} 行。")

    # 3. 用正確的表頭重新讀一次
    df_doctor = pd.read_excel(DOCTOR_EXCEL_NAME, skiprows=header_row_idx)
    df_doctor.columns = df_doctor.columns.astype(str).str.strip()

    # 4. 欄位名稱只要含關鍵字就統一改名，避免大小寫/括號差異造成抓不到
    rename_dict = {}
    for col in df_doctor.columns:
        if "圖片檔名" in col:
            rename_dict[col] = "圖片檔名"
        elif "填充物長度" in col:
            rename_dict[col] = "填充物長度(mm)"
    df_doctor = df_doctor.rename(columns=rename_dict)

    # 5. 濾掉空白列
    df_doctor = df_doctor.dropna(subset=["圖片檔名", "填充物長度(mm)"])

    return df_doctor, header_row_idx


def merge_yolo_and_doctor(df_doctor, df_yolo):
    df_yolo = df_yolo.copy()
    df_yolo["原始圖片檔名"] = df_yolo["圖片檔名"].apply(restore_roboflow_name)
    df_doctor = df_doctor.copy()
    df_doctor["原始圖片檔名"] = df_doctor["圖片檔名"].astype(str).str.strip().str.lower()

    merged_df = pd.merge(df_doctor, df_yolo, on="原始圖片檔名", how="inner", suffixes=("_x", "_y"))

    # --- 診斷 1：合併鍵有沒有一對多 ---
    n_unique_doctor_keys = df_doctor["原始圖片檔名"].nunique()
    n_unique_matched_keys = merged_df["原始圖片檔名"].nunique() if not merged_df.empty else 0
    key_counts = merged_df["原始圖片檔名"].value_counts() if not merged_df.empty else pd.Series(dtype=int)
    duplicated_keys = key_counts[key_counts > 1]

    print(f"\n🧪 【合併診斷】醫師表格唯一圖片數: {n_unique_doctor_keys}　"
          f"實際被配對到的唯一圖片數: {n_unique_matched_keys}　"
          f"合併後總列數: {len(merged_df)}")

    if len(duplicated_keys) > 0:
        print(f"\n❗ 有 {len(duplicated_keys)} 顆牙齒對應到不只一列，這在 5-fold 下不應該發生：")
        for key, cnt in duplicated_keys.items():
            variants = merged_df.loc[merged_df["原始圖片檔名"] == key, "圖片檔名_y"].tolist()
            folds = merged_df.loc[merged_df["原始圖片檔名"] == key, "Fold"].tolist()
            print(f"   - {key} 出現 {cnt} 次，YOLO 實體檔名: {variants}，Fold: {folds}")
        print("   👉 可能原因：(a) 01b_split_5folds.py 的分組沒有互斥，同一顆牙被分到兩個 fold 的 test；")
        print("      (b) 01b 的 ROBOFLOW_EXPORT_DIR 誤指到已離線擴增的資料夾，導致 test 裡混進 _aug 版本。")
        print("      建議刪掉 5fold 資料夾重跑 01b，而不是在這裡硬把重複列砍掉。")
        if not KEEP_FIRST_ON_DUPLICATE:
            raise ValueError("❌ 偵測到重複牙齒，已依 KEEP_FIRST_ON_DUPLICATE=False 的設定中止。")
        merged_df = merged_df.drop_duplicates(subset=["原始圖片檔名"], keep="first")
        print(f"   ⚠️ 已暫時保留每顆牙的第一筆，剩下 {len(merged_df)} 列繼續往下跑（但請務必回頭查清楚原因）。")

    if n_unique_doctor_keys != n_unique_matched_keys:
        print(f"\n⚠️ 醫師表格有 {n_unique_doctor_keys} 顆牙，但只有 {n_unique_matched_keys} 顆配對成功，"
              f"代表有 {n_unique_doctor_keys - n_unique_matched_keys} 顆偵測失敗或檔名對不上，"
              f"下面的智慧抓漏會列出詳細名單。")

    if merged_df.empty:
        print("\n❌ 經過 Roboflow 檔名還原後依然配對失敗！請檢查下方範例：")
        print(f"YOLO 抓到的前 3 筆實體檔名: {df_yolo['圖片檔名'].head(3).tolist()}")
        print(f"還原後的檔名比對範例: {df_yolo['原始圖片檔名'].head(3).tolist()}")
        print(f"醫師表格內的前 3 筆檔名: {df_doctor['圖片檔名'].head(3).tolist()}")
        return merged_df

    # --- 欄位整理：醫師的檔名留下來當主鍵，YOLO 的實體檔名另存一欄備查 ---
    merged_df = merged_df.drop(columns=["原始圖片檔名"])
    merged_df = merged_df.rename(columns={"圖片檔名_y": "YOLO實體檔名"})
    merged_df["圖片檔名"] = merged_df["圖片檔名_x"]
    merged_df = merged_df.drop(columns=["圖片檔名_x"])

    # 把 MATLAB 端最常用的欄位排到前面：檔名 / Fold / 像素長度 / 實際長度
    front_cols = ["圖片檔名", "Fold", "像素長度", "填充物長度(mm)"]
    other_cols = [c for c in merged_df.columns if c not in front_cols]
    merged_df = merged_df[front_cols + other_cols]
    merged_df = merged_df.sort_values(["Fold", "圖片檔名"]).reset_index(drop=True)

    return merged_df


def check_fold_distribution(merged_df):
    """檢查 Fold 欄位本身合不合理：值域是不是 1~N_FOLDS、每組有沒有牙齒。"""
    print(f"\n🧪 【Fold 分佈檢查】")
    counts = merged_df["Fold"].value_counts().sort_index()
    for fold_i in range(1, N_FOLDS + 1):
        n = int(counts.get(fold_i, 0))
        flag = "" if n > 0 else "   ❗ 這個 fold 一顆牙都沒有，MATLAB 那邊會少一輪"
        print(f"   fold{fold_i}: {n} 顆{flag}")

    bad_folds = sorted(set(merged_df["Fold"].unique()) - set(range(1, N_FOLDS + 1)))
    if bad_folds:
        print(f"   ❗ 出現預期外的 Fold 值：{bad_folds}，請確認 N_FOLDS 設定跟 01b/02 是否一致")


def check_against_fold_assignment(merged_df):
    """跟 01b 產生的 fold_assignment.csv 對照，確認分組沒有在中途被換掉。
    這關很重要：MATLAB 的 ANN 5-fold 會沿用同一套 Fold，
    如果 01b 重跑過、種子換過，這裡就會抓到不一致。"""
    if not os.path.exists(FOLD_ASSIGNMENT_CSV):
        print(f"\nℹ️ 找不到 {FOLD_ASSIGNMENT_CSV}，略過 Fold 對照檢查"
              f"（不影響輸出，但建議把 01b 的對照表放到這個路徑，多一層保險）。")
        return

    # dtype=str 很重要：牙齒ID長得像 "001"，用預設設定會被 pandas 讀成整數 1，
    # 之後跟 "001.jpg" 去掉副檔名的 "001" 永遠對不起來，會誤報成「不在分組表裡」。
    df_assign = pd.read_csv(FOLD_ASSIGNMENT_CSV, encoding="utf-8-sig", dtype=str)
    if not {"牙齒ID", "Fold"}.issubset(df_assign.columns):
        print(f"\n⚠️ {FOLD_ASSIGNMENT_CSV} 的欄位不是預期的「牙齒ID / Fold」，略過對照檢查。")
        return

    assign_map = {
        str(row["牙齒ID"]).strip().lower(): int(row["Fold"])
        for _, row in df_assign.iterrows()
    }

    mismatches = []
    not_in_csv = []
    for _, row in merged_df.iterrows():
        tooth_id = os.path.splitext(str(row["圖片檔名"]).strip().lower())[0]
        if tooth_id not in assign_map:
            not_in_csv.append(row["圖片檔名"])
        elif assign_map[tooth_id] != int(row["Fold"]):
            mismatches.append((row["圖片檔名"], assign_map[tooth_id], int(row["Fold"])))

    print(f"\n🧪 【Fold 對照檢查】比對 {FOLD_ASSIGNMENT_CSV}")
    if not mismatches and not not_in_csv:
        print(f"   ✅ {len(merged_df)} 顆牙的 Fold 都跟 01b 的分組表一致，MATLAB 可以放心沿用這欄。")
        return

    if mismatches:
        print(f"   ❗ 有 {len(mismatches)} 顆牙的 Fold 跟分組表對不起來（檔名, 分組表, Excel）：")
        for name, expected, actual in mismatches:
            print(f"      - {name}: 分組表 fold{expected} vs Excel fold{actual}")
        print("      👉 通常代表 01b 重跑過（換了亂數種子或資料量變了），但 02 的結果還是舊的。"
              "請把 5fold 資料夾刪掉，01b → 02 → 03 整條重跑一次。")
    if not_in_csv:
        print(f"   ⚠️ 有 {len(not_in_csv)} 顆牙不在分組表裡：{not_in_csv}"
              f"（可能是後來才加進醫師 Excel、但 01b 沒重跑）")


def smart_leak_detection(merged_df, all_yolo_skipped, header_row_idx):
    """列出醫師 Excel 裡有、但最終配對結果裡沒有的圖片，並逐筆判斷是哪一關掉的。"""
    df_orig_doctor = pd.read_excel(DOCTOR_EXCEL_NAME, skiprows=header_row_idx)
    df_orig_doctor.columns = df_orig_doctor.columns.astype(str).str.strip()
    orig_cols = [c for c in df_orig_doctor.columns if "圖片檔名" in c]
    if not orig_cols:
        return

    all_raw_doctor_pics = set(
        df_orig_doctor[orig_cols[0]].dropna().astype(str).str.strip().str.lower().tolist()
    )
    all_raw_doctor_pics = {p for p in all_raw_doctor_pics if p.endswith((".jpg", ".png", ".jpeg"))}

    all_merged_cleaned = set(merged_df["圖片檔名"].astype(str).str.strip().str.lower().tolist())
    missing_pics = all_raw_doctor_pics - all_merged_cleaned

    if not missing_pics:
        print(f"\n✅ 【涵蓋率檢查】醫師 Excel 共 {len(all_raw_doctor_pics)} 顆牙，"
              f"全部都在最終配對結果裡（{len(merged_df)} 列），5-fold 涵蓋率 100%。")
        print(f"   每顆牙的像素長度，都是由「訓練時沒看過它」的那個 fold 模型預測出來的。")
        return

    print(f"\n📢 【涵蓋率檢查】醫師 Excel 共 {len(all_raw_doctor_pics)} 顆牙，"
          f"其中 {len(missing_pics)} 顆沒有出現在最終的 {len(merged_df)} 列配對結果中。")
    print(f"📍 沒能成功對接的原始圖片檔名：{sorted(missing_pics)}")
    print(f"⚠️ 注意：5-fold 的前提是「每顆牙都恰好被 held-out 預測過一次」，"
          f"漏掉的牙齒等於在最終評估裡完全缺席，會讓 MATLAB 端的樣本數變少。")

    skipped_names_lower = {str(item["圖片檔名"]).lower() for item in all_yolo_skipped}
    skipped_restored_lower = {restore_roboflow_name(item["圖片檔名"]) for item in all_yolo_skipped}

    print("\n🔎 逐筆診斷原因：")
    for pic in sorted(missing_pics):
        if pic in skipped_restored_lower or pic in skipped_names_lower:
            matched = next(
                (item for item in all_yolo_skipped
                 if str(item["圖片檔名"]).lower() == pic or restore_roboflow_name(item["圖片檔名"]) == pic),
                None,
            )
            reason = matched.get("原因", "YOLO關鍵點偵測失敗") if matched else "YOLO關鍵點偵測失敗"
            fold = matched.get("Fold", "?") if matched else "?"
            print(f"   - {pic}：fold{fold} 的模型抓不到 -> {reason}")
        else:
            print(f"   - {pic}：YOLO 有算出像素，但跟醫師檔名對不上"
                  f"（懷疑 restore_roboflow_name() 的還原規則沒對到，"
                  f"請確認 Roboflow 匯出檔名是不是 '<編號>_jpg...' / '<編號>_png...' 的格式）")

    print("\n💡 備註：「模型抓不到」代表該 fold 的模型對這顆牙信心度太低或沒偵測到牙齒，"
          "屬於模型能力問題（可考慮調低 CONF_THRESHOLD 或增加訓練資料）；"
          "「檔名對不上」則是這支腳本的還原邏輯問題，改 restore_roboflow_name() 就能救回來。")


def main():
    df_yolo, all_yolo_skipped = load_yolo_results()
    df_doctor, header_row_idx = load_doctor_excel()

    merged_df = merge_yolo_and_doctor(df_doctor, df_yolo)
    if merged_df.empty:
        return

    check_fold_distribution(merged_df)
    check_against_fold_assignment(merged_df)

    merged_df.to_excel(OUTPUT_EXCEL_PATH, index=False)

    print(f"\n=======================================================")
    print(f"✨ 5-fold 資料合併階段完成！")
    print(f"📍 檔案名稱: {OUTPUT_EXCEL_PATH}")
    print(f"📊 共 {len(merged_df)} 顆牙，每顆一列，像素長度皆為 out-of-fold 預測")
    print(f"=======================================================")

    smart_leak_detection(merged_df, all_yolo_skipped, header_row_idx)

    print(f"\n👉 接下來 MATLAB 端（pixelToMmPredictor）請注意：")
    print(f"   1. 讀檔名改成 '{OUTPUT_EXCEL_PATH}'")
    print(f"   2. 分組欄位從 '資料夾來源' 改成 'Fold'，並改成 5-fold 迴圈"
          f"（第 k 輪：Fold==k 當 test，其餘當 train）")
    print(f"   3. 不要在 MATLAB 裡重新隨機切分，否則 YOLO 沒看過的牙齒"
          f"跟 ANN 沒看過的牙齒就對不起來了")


if __name__ == "__main__":
    main()
