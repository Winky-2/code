#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B4_build_C1_input.py
===============================================================
取代原本手動維護的「根管填充物像素長度_已配對_全片版.xlsx」。
(就是 C1 檔頭寫「由 03b_merge_data.py 產生，尚待撰寫」的那一支)

把三個來源併成 C1 吃得下的格式：

    B3 的逐顆牙對照   → 三種方法的像素長度(原圖px)
    B2 的像素預測     → scale、原圖尺寸等追溯欄位
    醫師的 mm 記錄    → 目標值 Y

*** 為什麼一次產出三份 ***
C1 只有一個輸入槽，三種方法必須跑三次。如果三份 Excel 是分開手動
維護的，很容易出現「這份少了兩顆、那份切分不一樣」的情況，跑出來
的三組數字就不可比，而三法比較正是這個專案要回答的問題。

所以本腳本：
  - 只取三種方法「同時都有值」的牙(任一方法失敗就三份一起排除)
  - 用同一組 SPLIT_SEED 產生切分，三份的 資料夾來源 欄完全相同
  - 三份的列數、列順序、檔名都一致

這樣三次 C1 的差異就只剩下「像素長度是怎麼量的」這一個變因。

*** 關於 train/test 切分 ***
醫師的 mm 記錄涵蓋的牙，全部都是 pose/seg 的 test 影像(對兩個模型
都是 held-out)，所以不存在「哪些被訓練過」的區分，train/test 只能
從這批裡面自己切。

也因為 train 列不再有「未被模型訓練過的 GT 座標」可用，train 和
test 的像素長度都來自模型推論——這反而比舊設計乾淨：舊設計 train
用 GT、test 用預測，兩邊分布不同，會系統性偏袒 bias 小的方法。現在
三個方法站在同一條起跑線，只比誤差的分散程度。

另外輸出 折數fold 欄(預設 5 摺)，之後 C1 要改成交叉驗證時直接可用。
樣本數這個量級，單次切分的理想比率/過長率一筆對錯就跳好幾個百分點，
k-fold 會穩定得多。

*** 檔名對應 ***
B3/B2 的檔名帶 Roboflow 後綴(xxx_jpg.rf.<hash>.jpg)，醫師的記錄通常
是原始檔名。本腳本會自動剝掉後綴再比對，對不上的不會靜默丟掉，一律
列進「未匹配清單」sheet。真的對不上的用 OVERRIDE_CSV 手動指定。
===============================================================
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 第1部分：設定
# ============================================================

# --- 輸入檔 ---
B3_XLSX = "B3_mask輔助對照_11.xlsx"
B3_SHEET = "逐顆牙對照"

B2_XLSX = "yolo像素預測_11.xlsx"          # 只拿追溯欄位，留空可略過
B2_SHEET = 0

MM_XLSX = "根管充填長度_20260904.xlsx"     # 醫師記錄的實際 mm
MM_SHEET = "資料填寫"                      # 工作表名稱或索引
MM_HEADER_ROW = None                      # None = 自動找標頭列(這份檔前面有說明列)
MM_NAME_COL = None                        # None = 自動偵測檔名欄
MM_VALUE_COL = None                       # None = 自動偵測 mm 欄

# 醫師表裡的選填欄位，帶進輸出供後續錯誤分析用(不進模型)
MM_EXTRA_COLS = ["牙位", "醫師備註（選填）"]

# 對不上的檔名手動指定：CSV 兩欄，標頭為 影像檔名,醫師檔名
OVERRIDE_CSV = ""

# --- 輸出 ---
OUTPUT_PREFIX = "根管填充物像素長度_已配對"
DIAGNOSTIC_XLSX = "B4_配對診斷.xlsx"

# --- 三種方法：輸出檔名後綴 → B3 的欄位名 ---
METHODS = {
    "原始預測": "AB長度_原始預測_原圖px",
    "長軸投影": "AB長度_長軸投影_原圖px",
    "mask幾何": "AB長度_mask幾何_原圖px",
}

# --- 切分 ---
TEST_RATIO = 0.25
SPLIT_SEED = 42        # 固定種子：重跑結果一致，論文才可重現
N_FOLDS = 5

# --- 合理範圍檢查(超出只警告，不自動剔除) ---
MM_RANGE = (5.0, 35.0)          # 根管充填長度的合理範圍
PX_PER_MM_CV_WARN = 0.15        # px/mm 變異係數超過此值 → 刻度可能仍不一致

# 是否把 mask狀態 非 ✅ 的牙排除。強烈建議 True：
# 那些列的 mask幾何/長軸投影 是空的，留著會讓三份檔案的列數對不齊。
DROP_MASK_FAILED = True


# ============================================================
# 第2部分：檔名正規化
# ============================================================

_RF_SUFFIX = re.compile(r"_(?:jpg|jpeg|png)\.rf\.[0-9a-z]+$", re.I)
_EXT = re.compile(r"\.(?:jpg|jpeg|png|tif|tiff|dcm)$", re.I)
_AUG = re.compile(r"_aug\d+$", re.I)


def normalize_name(name) -> str:
    """把各種來源的檔名壓成同一把鑰匙。

    xxx_jpg.rf.a1b2c3.jpg → xxx
    XXX .PNG              → xxx
    大小寫、空白、全形括號差異一併吸收。
    """
    s = str(name).strip()
    s = _EXT.sub("", s)          # 先去副檔名
    s = _RF_SUFFIX.sub("", s)    # 再去 Roboflow 後綴
    s = _EXT.sub("", s)          # 後綴剝完可能又露出一層
    s = _AUG.sub("", s)
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"[\s_\-]+", "", s)
    return s.lower()


def guess_column(df, keywords, exclude=()):
    """從欄名裡猜出想要的那一欄，猜不到回傳 None。"""
    for col in df.columns:
        c = str(col)
        if any(x in c for x in exclude):
            continue
        if any(k in c for k in keywords):
            return col
    return None


# ============================================================
# 第3部分：讀取來源
# ============================================================

def load_b3():
    p = Path(B3_XLSX)
    if not p.exists():
        raise FileNotFoundError(f"找不到 B3 輸出：{p}(請先跑 B3)")
    df = pd.read_excel(p, sheet_name=B3_SHEET)

    missing = [c for c in METHODS.values() if c not in df.columns]
    if missing:
        raise KeyError(
            f"B3 的「{B3_SHEET}」缺少欄位：{missing}\n"
            f"   現有欄位：{list(df.columns)}\n"
            f"   如果只看到 letterbox px 版本，代表 B3 跑的時候 scale 沒算出來，"
            f"請先確認 B2 的 縮放比scale 欄有值。"
        )
    return df


def find_header_row(path, sheet, max_scan=15):
    """找出真正的標頭列。

    醫師填的表格前面通常有標題、填寫說明、注意事項等等，pandas 預設拿
    第一列當欄名就會整張讀歪(欄名變成一長串說明文字、其餘變 Unnamed)。
    這裡掃前幾列，找第一個「同時出現檔名類與長度類關鍵字」的列。
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=max_scan)
    name_kw = ("檔名", "影像", "圖片", "編號")
    val_kw = ("mm", "毫米", "長度")
    MAX_HEADER_LEN = 15   # 欄名都很短；說明文字動輒數十字，用長度濾掉

    for i in range(len(raw)):
        cells = [str(c).strip() for c in raw.iloc[i].tolist() if pd.notna(c)]
        short = [c for c in cells if len(c) <= MAX_HEADER_LEN]
        # 關鍵字必須落在「不同的短欄名」上，避免整句說明同時命中兩邊
        name_hits = {c for c in short if any(k in c for k in name_kw)}
        val_hits = {c for c in short if any(k in c.lower() for k in val_kw)}
        if name_hits and (val_hits - name_hits):
            return i
    return 0


def load_mm():
    p = Path(MM_XLSX)
    if not p.exists():
        raise FileNotFoundError(f"找不到醫師的 mm 記錄：{p}")

    header_row = MM_HEADER_ROW
    if header_row is None:
        header_row = find_header_row(p, MM_SHEET)
        if header_row > 0:
            print(f"📋 偵測到標頭在第 {header_row + 1} 列(前面是說明文字)")
    df = pd.read_excel(p, sheet_name=MM_SHEET, header=header_row)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]

    name_col = MM_NAME_COL or guess_column(
        df, ("檔名", "影像", "圖片", "編號"), exclude=("備註",))
    value_col = MM_VALUE_COL or guess_column(
        df, ("mm", "毫米", "長度"), exclude=("像素", "px", "備註"))

    if name_col is None or value_col is None:
        raise KeyError(
            f"自動偵測不到 mm 檔的欄位(檔名欄={name_col}, mm欄={value_col})\n"
            f"   讀到的欄位：{list(df.columns)}\n"
            f"   若欄位明顯不對，多半是標頭列找錯了，請手動指定 MM_HEADER_ROW"
            f"(0 起算)；欄名對但猜錯的話填 MM_NAME_COL / MM_VALUE_COL。"
        )
    print(f"📋 mm 記錄：檔名欄=「{name_col}」，數值欄=「{value_col}」")

    keep = [name_col, value_col] + [c for c in MM_EXTRA_COLS if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={name_col: "醫師檔名", value_col: "填充物長度(mm)"})

    out["填充物長度(mm)"] = pd.to_numeric(out["填充物長度(mm)"], errors="coerce")
    n_raw = len(out)
    out = out.dropna(subset=["醫師檔名", "填充物長度(mm)"])
    if len(out) < n_raw:
        print(f"   略過 {n_raw - len(out)} 列空白或無 mm 值的列")

    dup = out["醫師檔名"].duplicated().sum()
    if dup:
        print(f"⚠️ 醫師檔名有 {dup} 筆重複，配對時會產生多對一，請先確認")

    out["_key"] = out["醫師檔名"].map(normalize_name)
    return out


def load_overrides():
    """手動檔名對應表：影像檔名 → 醫師檔名。"""
    if not OVERRIDE_CSV or not Path(OVERRIDE_CSV).exists():
        return {}
    df = pd.read_csv(OVERRIDE_CSV)
    return {normalize_name(r["影像檔名"]): normalize_name(r["醫師檔名"])
            for _, r in df.iterrows()}


# ============================================================
# 第4部分：切分
# ============================================================

def assign_split(n, seed, test_ratio, n_folds):
    """產生 train/test 標籤與 fold 編號。

    先固定種子打亂再切，避免資料本身的排序(通常按檔名，可能隱含
    拍攝日期或病人順序)洩漏成系統性差異。
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    n_test = max(1, int(round(n * test_ratio)))
    split = np.array(["train"] * n, dtype=object)
    split[order[:n_test]] = "test"

    fold = np.empty(n, dtype=int)
    fold[order] = np.arange(n) % n_folds + 1
    return split, fold


# ============================================================
# 第5部分：主流程
# ============================================================

def main():
    df3 = load_b3()
    df_mm = load_mm()
    overrides = load_overrides()

    df3 = df3.copy()
    df3["_key"] = df3["圖片檔名"].map(normalize_name)
    df3["_key"] = df3["_key"].map(lambda k: overrides.get(k, k))

    total_images = len(df3)
    dropped = []

    # ---- 排除 mask 失敗的 ----
    if DROP_MASK_FAILED and "mask狀態" in df3.columns:
        bad = df3[df3["mask狀態"] != "✅"]
        for _, r in bad.iterrows():
            dropped.append({"圖片檔名": r["圖片檔名"],
                            "排除原因": f"mask狀態={r['mask狀態']}"})
        df3 = df3[df3["mask狀態"] == "✅"]

    # ---- 排除任一方法沒有值的 ----
    cols = list(METHODS.values())
    incomplete = df3[df3[cols].isna().any(axis=1)]
    for _, r in incomplete.iterrows():
        empties = [c for c in cols if pd.isna(r[c])]
        dropped.append({"圖片檔名": r["圖片檔名"],
                        "排除原因": f"缺少長度欄位：{empties}"})
    df3 = df3.dropna(subset=cols)

    # ---- 併 mm ----
    merged = df3.merge(df_mm, on="_key", how="left")

    no_mm = merged[merged["填充物長度(mm)"].isna()]
    for _, r in no_mm.iterrows():
        dropped.append({"圖片檔名": r["圖片檔名"], "排除原因": "醫師記錄找不到對應 mm"})
    merged = merged.dropna(subset=["填充物長度(mm)"])

    # 反向：醫師有記錄但影像端沒對上的
    matched_keys = set(merged["_key"])
    orphan_mm = df_mm[~df_mm["_key"].isin(matched_keys)]

    if merged.empty:
        raise RuntimeError(
            "配對後一筆都不剩。最常見原因是檔名規則對不上，"
            f"請看 {DIAGNOSTIC_XLSX} 的「未匹配清單」，"
            "必要時用 OVERRIDE_CSV 手動指定。"
        )

    # ---- 併 B2 追溯欄位 ----
    if B2_XLSX and Path(B2_XLSX).exists():
        df2 = pd.read_excel(B2_XLSX, sheet_name=B2_SHEET)
        keep = [c for c in ("圖片檔名", "縮放比scale", "原圖寬", "原圖高",
                            "原圖檔名", "scale驗證") if c in df2.columns]
        if len(keep) > 1:
            merged = merged.merge(df2[keep], on="圖片檔名", how="left",
                                  suffixes=("", "_b2"))

    merged = merged.sort_values("圖片檔名").reset_index(drop=True)

    # ---- 切分(三份共用) ----
    split, fold = assign_split(len(merged), SPLIT_SEED, TEST_RATIO, N_FOLDS)
    merged["資料夾來源"] = split
    merged["折數fold"] = fold

    # ---- 合理性檢查 ----
    warnings = []
    lo, hi = MM_RANGE
    weird = merged[(merged["填充物長度(mm)"] < lo) | (merged["填充物長度(mm)"] > hi)]
    if not weird.empty:
        warnings.append(f"有 {len(weird)} 筆 mm 值落在 {lo}–{hi} 之外，請確認是否登打錯誤")

    # 醫師備註 shortening/elongation 代表充填物沒有剛好到根尖，
    # 那幾筆的「填充物長度」與影像量到的「切端到根尖」本來就不該相等，
    # 是系統性的離群來源，不自動剔除但要標出來。
    note_col = "醫師備註（選填）"
    if note_col in merged.columns:
        flagged = merged[merged[note_col].notna()]
        if not flagged.empty:
            warnings.append(
                f"有 {len(flagged)} 筆帶醫師備註({', '.join(map(str, flagged[note_col].unique()))})，"
                f"充填物可能未達根尖，跑完 C1 後檢查這幾筆是不是誤差最大的")

    # px/mm 比值：scale 修正如果做對了，這個比值應該相當穩定。
    # 變異大代表各片的實際刻度仍不一致(例如不同機器的 pixel pitch 不同)，
    # 那是 ANN 再怎麼調都補不回來的資訊缺口。
    ratio_stats = []
    for label, col in METHODS.items():
        ratio = merged[col] / merged["填充物長度(mm)"]
        cv = float(ratio.std(ddof=1) / ratio.mean()) if len(ratio) > 1 else np.nan
        ratio_stats.append({
            "方法": label,
            "px每mm_平均": round(float(ratio.mean()), 3),
            "px每mm_標準差": round(float(ratio.std(ddof=1)), 3) if len(ratio) > 1 else None,
            "變異係數CV": round(cv, 4),
            "判讀": "⚠️ 刻度仍不一致" if cv > PX_PER_MM_CV_WARN else "✅ 尚可",
        })

    # ---- 產出三份 ----
    written = []
    for label, col in METHODS.items():
        out = pd.DataFrame({
            "圖片檔名": merged["圖片檔名"],
            "填充物長度(mm)": merged["填充物長度(mm)"].round(3),
            "像素長度": merged[col].round(4),
            "資料夾來源": merged["資料夾來源"],
            "折數fold": merged["折數fold"],
            "長度方法": label,
        })
        for extra in ("縮放比scale", "原圖寬", "原圖高", "QC備註", *MM_EXTRA_COLS):
            if extra in merged.columns:
                out[extra] = merged[extra]

        path = f"{OUTPUT_PREFIX}_{label}.xlsx"
        out.to_excel(path, index=False)
        written.append(path)

    # ---- 診斷檔 ----
    with pd.ExcelWriter(DIAGNOSTIC_XLSX, engine="openpyxl") as w:
        pd.DataFrame(ratio_stats).to_excel(w, sheet_name="刻度一致性", index=False)
        (pd.DataFrame(dropped) if dropped else
         pd.DataFrame(columns=["圖片檔名", "排除原因"])).to_excel(
            w, sheet_name="排除清單", index=False)
        if not orphan_mm.empty:
            orphan_mm[["醫師檔名", "填充物長度(mm)"]].to_excel(
                w, sheet_name="未匹配清單", index=False)
        merged.drop(columns=["_key"]).to_excel(w, sheet_name="完整合併表", index=False)

    # ---- 終端摘要 ----
    n = len(merged)
    print(f"\n✅ 完成：{total_images} 張進來，{n} 顆牙成功配對")
    if dropped:
        print(f"⚠️  排除 {len(dropped)} 筆，原因見「排除清單」")
    if not orphan_mm.empty:
        print(f"⚠️  醫師記錄有 {len(orphan_mm)} 筆對不到影像，見「未匹配清單」")
    for msg in warnings:
        print(f"⚠️  {msg}")

    print(f"\n切分(seed={SPLIT_SEED})：train {int((split=='train').sum())} 筆 / "
          f"test {int((split=='test').sum())} 筆，另附 {N_FOLDS} 摺 fold 欄")
    print("三份檔案的列數、列順序、切分完全相同，唯一差異是「像素長度」的算法：")
    for p in written:
        print(f"   📄 {p}")
    print(f"   📄 {DIAGNOSTIC_XLSX}")

    print("\n=== 刻度一致性(px 每 mm) ===")
    print(pd.DataFrame(ratio_stats).to_string(index=False))
    print("CV 偏大代表各片的實際刻度仍不一致——這是資訊缺口，不是模型調得不夠好。")
    print("真要解決得從原始 DICOM 的 PixelSpacing 標籤下手。")

    print("\n👉 接著把 C1 的 filename 依序指向上面三份，各跑一次做對照。")


if __name__ == "__main__":
    main()
