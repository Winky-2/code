"""
B3_merge_pose_to_C1.py
============================
對應流程圖：B2(pose推論量px) ──► 【B3 配對】 ──► C1(ANN px→mm)

就是 C1 檔頭寫「由 03b_merge_data.py 產生，尚待撰寫」的那一支。
把兩個來源併成 C1 指定的輸入檔：

    B2 輸出 (yolo像素預測_*.xlsx「預測結果」)
        → AB像素長度_原圖px   (模型預測)
        → 真實AB像素長度_原圖px (GT 座標算的，B2 有比對到 GT 才有)
    醫師 mm 記錄 (根管充填長度_*.xlsx「資料填寫」)
        → 填充物長度(mm)、牙位、年齡、性別、醫師備註

產出 C1 寫死讀的檔名與欄位：
    根管填充物像素長度_已配對_全片版.xlsx
    圖片檔名 / 填充物長度(mm) / 像素長度 / 資料夾來源 (+ 追溯用附加欄)

*** train/test 的像素長度從哪來(對齊 C1 檔頭的設計) ***
醫師有 mm 的牙全部是 B2 的 test 圖，對 pose 模型都是 held-out，
所以 train/test 只能在這批裡用固定種子自己切。
    TRAIN_PX_SOURCE = "GT"   : train 列用 GT 座標算的長度，test 列用模型預測
                               → C1 檔頭的原始設計：ANN 學乾淨的 px→mm，
                                 test 量的是 pose+ANN 端到端誤差
    TRAIN_PX_SOURCE = "pred": train/test 都用模型預測
                               → 兩邊分布一致；pose 若有系統性 bias，
                                 ANN 會在 train 端一併學掉
兩種各跑一次 C1 就是只差一個變因的對照，切分完全相同。

*** 不會默默丟資料 ***
對不上、B2 失敗、缺 GT、mm 異常，一律列進診斷檔的「排除清單」/「未匹配清單」。

安裝需求：
    pip install pandas numpy openpyxl
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- 設定區 ----------------
B2_XLSX = "yolo像素預測_26_batch4.xlsx"   # 👈 B2 的 OUTPUT_XLSX
B2_SHEET = "預測結果"

MM_XLSX = "根管充填長度_20260921-67.xlsx"                     # 👈 醫師 mm 記錄
MM_SHEET = "資料填寫"                                          # 檔內另有「工作表1」是草稿，別讀錯
MM_EXTRA_COLS = ["牙位", "年齡", "性別", "醫師備註（選填）"]    # 帶進輸出供錯誤分析，C1 不讀

OUTPUT_XLSX = "根管填充物像素長度_已配對_26_batch4.xlsx"           # ⚠️ C1 寫死讀這個檔名，別改
DIAGNOSTIC_XLSX = "B3_配對診斷_重新標註-2.xlsx"

TRAIN_PX_SOURCE = "pred"      # "GT" 或 "pred"，見檔頭；C1 走 k-fold 時必須是 "pred"
TEST_RATIO = 0.2              # 對齊 C1 輸出檔名的 train80_test20
SPLIT_SEED = 42               # 固定種子，重跑切分一致
N_FOLDS = 5                   # 另附「折數fold」欄給 C1_v6 做 k-fold(每顆牙都輪到當一次 test)
                              # ⚠️ k-fold 時每顆牙都會輪流當 train/test，請用 TRAIN_PX_SOURCE="pred"

DROP_LOW_IOU = False          # True：B2 狀態標「選框IoU偏低」的牙一律排除
MM_RANGE = (5.0, 35.0)        # 超出只警告，不自動剔除
PX_PER_MM_CV_WARN = 0.15      # px/mm 變異係數超過此值 → 各片刻度可能不一致

B2_PRED_COL = "AB像素長度_原圖px"
B2_GT_COL = "真實AB像素長度_原圖px"


# ============================================================
# 第0部分：找輸入檔
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
SEARCH_UP_LEVELS = 3          # 往上找幾層資料夾(例：pose的舊資料 → code → 小專題)


def find_input(name):
    """依序找：原樣路徑(絕對或相對於目前目錄) → 腳本所在資料夾 → 往上 SEARCH_UP_LEVELS 層
    → 腳本資料夾底下任一子資料夾。找到第一個就用，並印出實際路徑。"""
    p = Path(name)
    if p.is_absolute():
        return p if p.exists() else None
    tried = [Path.cwd() / p, SCRIPT_DIR / p]
    tried += [SCRIPT_DIR.parents[i] / p for i in range(min(SEARCH_UP_LEVELS, len(SCRIPT_DIR.parents)))]
    for c in tried:
        if c.exists():
            return c
    hits = sorted(SCRIPT_DIR.rglob(p.name))
    return hits[0] if hits else None


def require_input(name, label):
    p = find_input(name)
    if p is None:
        raise FileNotFoundError(
            f"❌ 找不到{label}：{name}\n"
            f"   已找過：目前目錄({Path.cwd()})、腳本資料夾({SCRIPT_DIR})、往上 {SEARCH_UP_LEVELS} 層、"
            f"腳本資料夾的所有子資料夾。\n"
            f"   請把檔案放進其中一處，或在設定區直接填絕對路徑。")
    if p.resolve() != (Path.cwd() / name).resolve():
        print(f"📂 {label}：使用 {p}")
    return p


# ============================================================
# 第1部分：檔名對應
# ============================================================

_LEADING_NUM = re.compile(r"^\s*0*(\d+)")


def name_key(name):
    """取檔名開頭編號當配對 key：
    '001_jpg.rf.abc123.jpg'(Roboflow) / '001.jpg'(醫師表) → '1'。"""
    m = _LEADING_NUM.match(str(name))
    return str(int(m.group(1))) if m else str(name).strip().lower()


# ============================================================
# 第2部分：讀取來源
# ============================================================

def find_header_row(path, sheet, max_scan=15):
    """醫師表前幾列是標題與填寫說明，找第一個同時有「檔名」與「mm」短欄名的列。"""
    raw = pd.read_excel(path, sheet_name=sheet, header=None, nrows=max_scan)
    for i in range(len(raw)):
        cells = [str(c).strip() for c in raw.iloc[i] if pd.notna(c)]
        short = [c for c in cells if len(c) <= 15]
        if any("檔名" in c for c in short) and any("mm" in c.lower() for c in short):
            return i
    raise ValueError(f"在 {path}「{sheet}」前 {max_scan} 列找不到標頭列(需同時有『檔名』與『mm』欄)")


def load_mm():
    p = require_input(MM_XLSX, "醫師 mm 記錄")
    hdr = find_header_row(p, MM_SHEET)
    df = pd.read_excel(p, sheet_name=MM_SHEET, header=hdr)
    df.columns = [str(c).strip() for c in df.columns]

    name_col = next(c for c in df.columns if "檔名" in c)
    mm_col = next(c for c in df.columns if "mm" in c.lower())
    keep = [name_col, mm_col] + [c for c in MM_EXTRA_COLS if c in df.columns]
    out = df[keep].rename(columns={name_col: "醫師檔名", mm_col: "填充物長度(mm)"})

    out["填充物長度(mm)"] = pd.to_numeric(out["填充物長度(mm)"], errors="coerce")
    out = out.dropna(subset=["醫師檔名", "填充物長度(mm)"]).copy()
    out["_key"] = out["醫師檔名"].map(name_key)

    dup = out["_key"].duplicated(keep="first")
    if dup.any():
        print(f"⚠️ 醫師表有 {int(dup.sum())} 個編號重複，只保留第一筆：{list(out.loc[dup, '醫師檔名'])}")
        out = out[~dup]
    print(f"📋 醫師表：標頭在第 {hdr + 1} 列，有效 {len(out)} 筆")
    return out


def load_b2():
    p = require_input(B2_XLSX, "B2 輸出(請先跑 B2_test_pose_and_measure.py)")
    df = pd.read_excel(p, sheet_name=B2_SHEET)
    if B2_PRED_COL not in df.columns:
        raise KeyError(f"B2「{B2_SHEET}」缺少「{B2_PRED_COL}」欄，現有欄位：{list(df.columns)}")
    if B2_GT_COL not in df.columns:
        df[B2_GT_COL] = np.nan   # 整批都沒比對到 GT 時 B2 不會產這欄
    df["_key"] = df["圖片檔名"].map(name_key)

    dup = df["_key"].duplicated(keep="first")
    if dup.any():
        print(f"⚠️ B2 有 {int(dup.sum())} 張圖編號重複(擴增圖混進 test？)，只保留第一張")
        df = df[~dup]
    print(f"📋 B2：預測成功 {len(df)} 張")
    return df


# ============================================================
# 第3部分：切分
# ============================================================

def assign_split(n, seed, test_ratio):
    """固定種子打亂再切，避免檔名排序(可能隱含拍攝順序)造成系統性差異。"""
    order = np.random.default_rng(seed).permutation(n)
    split = np.array(["train"] * n, dtype=object)
    split[order[:max(1, int(round(n * test_ratio)))]] = "test"
    return split


# ============================================================
# 主流程
# ============================================================

def main():
    if TRAIN_PX_SOURCE not in ("GT", "pred"):
        raise ValueError(f"TRAIN_PX_SOURCE 只能是 'GT' 或 'pred'，目前是 {TRAIN_PX_SOURCE!r}")

    df_mm = load_mm()
    df_b2 = load_b2()
    dropped = []

    # ---- 配對(雙向檢查，對不上的都記下來) ----
    merged = df_b2.merge(df_mm, on="_key", how="inner")
    for _, r in df_b2[~df_b2["_key"].isin(df_mm["_key"])].iterrows():
        dropped.append({"圖片檔名": r["圖片檔名"], "排除原因": "醫師表找不到這個編號"})
    orphan_mm = df_mm[~df_mm["_key"].isin(df_b2["_key"])][["醫師檔名", "填充物長度(mm)"]].copy()
    orphan_mm["可能原因"] = "B2 推論失敗/選框失敗，或不在 B2 的 IMAGE_DIR 裡(見 B2「偵測失敗清單」)"

    # ---- 選框 IoU 偏低(可選排除) ----
    if DROP_LOW_IOU and "狀態" in merged.columns:
        low = merged["狀態"].astype(str).str.contains("IoU")
        for _, r in merged[low].iterrows():
            dropped.append({"圖片檔名": r["圖片檔名"], "排除原因": "B2 選框IoU偏低"})
        merged = merged[~low]

    # ---- 缺 GT 長度的牙：兩種模式都排除 ----
    # 只在 GT 模式排除的話，GT/pred 兩模式的列數與切分會不同，就不是單一變因對照了。
    no_gt = merged[B2_GT_COL].isna()
    for _, r in merged[no_gt].iterrows():
        dropped.append({"圖片檔名": r["圖片檔名"], "排除原因": "B2 沒有 GT 長度(舊標註 A/B 可見度=0)"})
    merged = merged[~no_gt].copy()

    # 依編號排序後再切，切分只取決於「哪些牙」+ 種子，跟 B2 輸出的列順序無關
    merged["_ord"] = pd.to_numeric(merged["_key"], errors="coerce")
    merged = merged.sort_values(["_ord", "_key"]).reset_index(drop=True)

    if merged.empty:
        raise RuntimeError(f"配對後 0 筆，請看 {DIAGNOSTIC_XLSX} 的排除/未匹配清單")

    merged["資料夾來源"] = assign_split(len(merged), SPLIT_SEED, TEST_RATIO)
    fold_order = np.random.default_rng(SPLIT_SEED).permutation(len(merged))
    merged["折數fold"] = 0
    merged.loc[fold_order, "折數fold"] = np.arange(len(merged)) % N_FOLDS + 1
    is_train = merged["資料夾來源"] == "train"

    if TRAIN_PX_SOURCE == "GT":
        merged["像素長度"] = np.where(is_train, merged[B2_GT_COL], merged[B2_PRED_COL])
        merged["像素長度來源"] = np.where(is_train, "GT座標", "模型預測")
    else:
        merged["像素長度"] = merged[B2_PRED_COL]
        merged["像素長度來源"] = "模型預測"

    # ---- 輸出 C1 輸入檔(第一個 sheet 就是資料，C1 預設讀第一個) ----
    cols = ["圖片檔名", "填充物長度(mm)", "像素長度", "資料夾來源", "折數fold", "像素長度來源",
            B2_PRED_COL, B2_GT_COL]
    for c in ("IoU_vs舊GT框", "A_B順序疑似顛倒", "狀態", *MM_EXTRA_COLS):
        if c in merged.columns:
            cols.append(c)
    out = merged[cols].copy()
    out["像素長度"] = out["像素長度"].round(4)
    out.to_excel(OUTPUT_XLSX, index=False)

    # ---- 診斷 ----
    warnings = []
    lo, hi = MM_RANGE
    n_weird = int(((out["填充物長度(mm)"] < lo) | (out["填充物長度(mm)"] > hi)).sum())
    if n_weird:
        warnings.append(f"{n_weird} 筆 mm 值在 {lo}–{hi} 之外，請確認登打")

    note_col = "醫師備註（選填）"
    if note_col in out.columns and out[note_col].notna().any():
        flagged = out[out[note_col].notna()]
        warnings.append(f"{len(flagged)} 筆帶醫師備註({', '.join(map(str, flagged[note_col].unique()))})，"
                        f"充填物可能沒到根尖，C1 跑完檢查這幾筆是否誤差最大")

    if "A_B順序疑似顛倒" in out.columns:
        n_swap = int((out["A_B順序疑似顛倒"] == "⚠️是").sum())
        if n_swap:
            warnings.append(f"{n_swap} 筆 A/B 疑似顛倒(長度不受影響，但代表標註順序不一致)")

    # px/mm 一致性 + pose 對長度的影響(test 端 ANN 吃到的誤差)
    stats = []
    for label, col in (("GT座標", B2_GT_COL), ("模型預測", B2_PRED_COL)):
        r = out[col] / out["填充物長度(mm)"]
        cv = float(r.std(ddof=1) / r.mean()) if len(r) > 1 else np.nan
        stats.append({
            "像素長度來源": label,
            "px每mm_平均": round(float(r.mean()), 3),
            "變異係數CV": round(cv, 4),
            "與mm相關r": round(float(np.corrcoef(out[col], out["填充物長度(mm)"])[0, 1]), 3),
            "判讀": "⚠️ 各片刻度不一致" if cv > PX_PER_MM_CV_WARN else "✅ 尚可",
        })
    len_err = out[B2_PRED_COL] - out[B2_GT_COL]
    stats.append({"像素長度來源": "預測−GT 長度誤差(px)",
                  "px每mm_平均": round(float(len_err.mean()), 3),     # 此列借欄位放 bias
                  "變異係數CV": round(float(len_err.abs().mean()), 3),  # 此列借欄位放 MAE
                  "與mm相關r": None,
                  "判讀": "此列：px每mm欄=bias、CV欄=MAE"})

    with pd.ExcelWriter(DIAGNOSTIC_XLSX, engine="openpyxl") as w:
        pd.DataFrame(stats).to_excel(w, sheet_name="刻度一致性", index=False)
        (pd.DataFrame(dropped) if dropped else pd.DataFrame(columns=["圖片檔名", "排除原因"])
         ).to_excel(w, sheet_name="排除清單", index=False)
        orphan_mm.to_excel(w, sheet_name="未匹配清單", index=False)
        pd.DataFrame([{
            "B2輸入": B2_XLSX, "醫師表": MM_XLSX, "TRAIN_PX_SOURCE": TRAIN_PX_SOURCE,
            "TEST_RATIO": TEST_RATIO, "SPLIT_SEED": SPLIT_SEED, "DROP_LOW_IOU": DROP_LOW_IOU,
            "配對成功": len(out), "train": int(is_train.sum()), "test": int((~is_train).sum()),
            "排除": len(dropped), "醫師表未匹配": len(orphan_mm),
        }]).to_excel(w, sheet_name="執行摘要", index=False)

    # ---- 終端摘要 ----
    print(f"\n=======================================================")
    print(f"✅ 配對完成：{len(out)} 顆牙 → {OUTPUT_XLSX}")
    print(f"   切分(seed={SPLIT_SEED})：train {int(is_train.sum())} / test {int((~is_train).sum())}")
    print(f"   train 像素長度 = {'GT座標' if TRAIN_PX_SOURCE == 'GT' else '模型預測'}，test = 模型預測")
    if dropped:
        print(f"⚠️ 排除 {len(dropped)} 筆，見 {DIAGNOSTIC_XLSX}「排除清單」")
    if len(orphan_mm):
        print(f"⚠️ 醫師表有 {len(orphan_mm)} 筆對不到 B2 結果，見「未匹配清單」")
    for msg in warnings:
        print(f"⚠️ {msg}")
    print("\n=== 刻度一致性 ===")
    print(pd.DataFrame(stats).to_string(index=False))
    print("CV 偏大 = 各片 pixel pitch 不一致，是資訊缺口，ANN 調參補不回來。")
    print(f"=======================================================")
    print(f"👉 接下來執行 C1_pixelToMmPredictor.m → C2_plotMyResults.m")


if __name__ == "__main__":
    main()
