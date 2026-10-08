#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B34_run_seeds.py
===============================================================
多 seed 版的 B3 → B4。B3、B4 原檔「一行都不改」，本檔 import 它們，
每個 seed 只換路徑設定再呼叫各自的 main()，處理邏輯保證跟單跑時完全相同。

流程(每個 seed)：
  B3：SEG_WEIGHTS / B2_XLSX / 輸出檔 / 視覺化資料夾 換成該 seed 的
  B4：B3_XLSX / 診斷檔 換成該 seed 的，原始輸出先放 B4_seed_raw/seed{s}/

*** 跨 seed 對齊(單一變因的關鍵) ***
B4 會依 mask 失敗、QC、缺冠寬 排除牙齒；不同 seed 排除的牙可能不同 →
列數不同 → B4 的 fold 指派也會跟著不同，C1 的差異就混進「樣本/分摺不同」。
所以全部 seed 跑完後：
  1) 取所有 seed 都有的牙(交集)
  2) 用 B4 自己的 assign_split(同 SPLIT_SEED/TEST_RATIO/N_FOLDS)重新指派 fold
  3) 寫出最終給 C1 的檔：根管填充物像素長度_已配對_{方法}_seed{s}.xlsx
→ 各 seed 的列、順序、fold 完全相同，只剩「seg 權重不同」這一個變因。
若所有 seed 的牙都一樣，對齊不改任何東西，fold 與 B4 單跑結果相同。

C1 的 METHOD 依序設成：mask幾何_seed0 … mask幾何_seed4、冠寬比例尺_seed0 …
牙位基準 不看影像，對齊後各 seed 完全相同，C1 跑一次即可。

輸出：
  B3_mask輔助對照_11n_padded_seed{s}.xlsx、B3_mask視覺化_11n_padded_seed{s}/
  B4_配對診斷_11n_padded_seed{s}.xlsx(對齊前的診斷)
  根管填充物像素長度_已配對_{方法}_seed{s}.xlsx      ← C1 輸入
  B34_seed彙整_11n_padded.xlsx(各 seed 指標、mean±SD、對齊報告)

⚠️ 本檔要跟 B3、B4 放在同一個資料夾，並從該資料夾執行(相對路徑以工作目錄為準)。
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 可手動修改 ====================
RUN_NOW = True                     # False：只印要跑的路徑，不執行

TRAIN_SEEDS = [0, 1, 2, 3, 4]      # 👈 與 B1 v2 / B2 v4 一致
B3_MODULE = "B3_mask_assist_and_compare_v2"
B4_MODULE = "B4_build_C1_input"

# --- 各 seed 的輸入(B1/B2 的輸出) ---
SEG_WEIGHTS_TMPL = "yolo11n_seg_run_padded_seed{s}/weights_ready.pt"
B2_XLSX_TMPL = "B2_seg推論統計_11n_padded_seed{s}.xlsx"

# --- 各 seed 的輸出 ---
B3_OUT_TMPL = "B3_mask輔助對照_11n_padded_seed{s}.xlsx"
B3_VIS_TMPL = "B3_mask視覺化_11n_padded_seed{s}"
B3_SAVE_VIS = True                 # 5 個 seed 都畫圖較慢；只要數字可設 False
SKIP_B3_IF_EXISTS = False          # True：B3 輸出已存在就不重跑(只改 B4 時省時間)

B4_RAW_DIR_TMPL = "B4_seed_raw/seed{s}"            # B4 原始輸出(對齊前)
B4_DIAG_TMPL = "B4_配對診斷_11n_padded_seed{s}.xlsx"
PREFIX_NAME = "根管填充物像素長度_已配對"          # 與 B4 的 OUTPUT_PREFIX、C1 一致
FINAL_TMPL = PREFIX_NAME + "_{label}_seed{s}.xlsx"  # C1 的 METHOD = "{label}_seed{s}"

SUMMARY_XLSX = "B34_seed彙整_11n_padded.xlsx"
# ===================================================

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ============================================================
# 第1部分：逐 seed 跑 B3、B4
# ============================================================

def run_b3(b3, s):
    out = B3_OUT_TMPL.format(s=s)
    if SKIP_B3_IF_EXISTS and Path(out).exists():
        print(f"⏭️  seed {s}：{out} 已存在，跳過 B3")
        return out
    weights = SEG_WEIGHTS_TMPL.format(s=s)
    if not Path(weights).exists():
        raise FileNotFoundError(f"seed {s} 找不到權重 {weights}(先跑 B1 v2)")
    b3.SEG_WEIGHTS = weights
    b3.B2_XLSX = B2_XLSX_TMPL.format(s=s)
    b3.OUTPUT_XLSX = out
    b3.VIS_DIR = B3_VIS_TMPL.format(s=s)
    b3.SAVE_VISUALIZATION = B3_SAVE_VIS
    b3.main()
    return out


def run_b4(b4, s, b3_out):
    raw_dir = Path(B4_RAW_DIR_TMPL.format(s=s))
    raw_dir.mkdir(parents=True, exist_ok=True)
    b4.B3_XLSX = b3_out
    b4.OUTPUT_PREFIX = str(raw_dir / PREFIX_NAME)
    b4.DIAGNOSTIC_XLSX = B4_DIAG_TMPL.format(s=s)
    b4.main()
    files = sorted(raw_dir.glob(f"{PREFIX_NAME}_*.xlsx"))
    if not files:
        raise FileNotFoundError(f"seed {s}：B4 沒有在 {raw_dir} 產出任何檔案")
    return {f.stem[len(PREFIX_NAME) + 1:]: pd.read_excel(f) for f in files}


# ============================================================
# 第2部分：跨 seed 對齊
# ============================================================

def align(b4, raw):
    """raw: {seed: {label: df}} → 對齊後的同結構 dict + 對齊報告 df。"""
    seeds = list(raw)
    labels = sorted(set.intersection(*(set(v) for v in raw.values())))
    if not labels:
        raise RuntimeError("各 seed 的 B4 輸出沒有共同的方法")

    # 同一 seed 內各方法的列本來就一致(B4 保證)；這裡再確認一次
    names = {}
    for s in seeds:
        sets = {lb: set(raw[s][lb]["圖片檔名"]) for lb in labels}
        ref = sets[labels[0]]
        if any(v != ref for v in sets.values()):
            raise RuntimeError(f"seed {s} 的各方法列不一致，B4 輸出異常")
        names[s] = ref

    common = sorted(set.intersection(*names.values()))
    if not common:
        raise RuntimeError("所有 seed 沒有任何共同的牙，無法對齊")

    # 用 B4 自己的切分函式，依排序後的共同牙重新指派(與 B4 內部做法相同)
    split, fold = b4.assign_split(len(common), b4.SPLIT_SEED, b4.TEST_RATIO, b4.N_FOLDS)
    ref = pd.DataFrame({"圖片檔名": common, "資料夾來源": split, "折數fold": fold})

    report, aligned = [], {}
    for s in seeds:
        dropped = sorted(names[s] - set(common))
        report.append({"train_seed": s, "B4原始張數": len(names[s]),
                       "對齊後張數": len(common), "因對齊排除": len(dropped),
                       "排除檔名": ", ".join(dropped)})
        aligned[s] = {}
        for lb in labels:
            df = raw[s][lb]
            cols = list(df.columns)
            df = (df[df["圖片檔名"].isin(common)]
                  .drop(columns=["資料夾來源", "折數fold"])
                  .merge(ref, on="圖片檔名", how="left")
                  .sort_values("圖片檔名").reset_index(drop=True))
            aligned[s][lb] = df[cols]
    return aligned, pd.DataFrame(report), labels


# ============================================================
# 第3部分：彙整
# ============================================================

def b3_metrics(path):
    out = {}
    try:
        lv = pd.read_excel(path, sheet_name="長度vsGT")
        row = lv[lv["空間"] == "原圖"]
        if not row.empty:
            r = row.iloc[0]
            out["B3_長度MAE_原圖px"] = r["長度MAE px"]
            out["B3_bias_原圖px"] = r["長度偏差(bias) px"]
            out["B3_誤差SD_原圖px"] = r["誤差標準差 px"]
    except ValueError:
        pass                                   # 沒有 GT → 沒有這張表
    df = pd.read_excel(path, sheet_name="逐顆牙對照")
    out["B3_mask失敗張數"] = int((df["mask狀態"] != "✅").sum())
    out["B3_建議複查張數"] = int((df["建議人工複查"] == "⚠️是").sum())
    return out


def aligned_metrics(b4, frames):
    """在對齊後的資料上算(各 seed 同一批牙，可直接比)。"""
    out = {}
    for lb, df in frames.items():
        X, Y = df["像素長度"].astype(float), df["填充物長度(mm)"].astype(float)
        if X.nunique() > 1:
            out[f"{lb}_與Y相關r"] = round(float(np.corrcoef(X, Y)[0, 1]), 4)
        if lb in b4.METHODS:                   # 單位是 px
            ratio = X / Y
            out[f"{lb}_px每mm_CV"] = round(float(ratio.std(ddof=1) / ratio.mean()), 4)
        else:                                  # 單位是 mm(衍生方法)
            d = X - Y
            out[f"{lb}_去bias_MAE_mm"] = round(float((d - d.mean()).abs().mean()), 4)
    return out


# ============================================================
# 第4部分：主流程
# ============================================================

def main():
    if not TRAIN_SEEDS:
        raise ValueError("TRAIN_SEEDS 至少要有一個值")
    if not RUN_NOW:
        print("RUN_NOW=False，只列出路徑：")
        for s in TRAIN_SEEDS:
            print(f"  seed {s}: {SEG_WEIGHTS_TMPL.format(s=s)} + {B2_XLSX_TMPL.format(s=s)}"
                  f" → {B3_OUT_TMPL.format(s=s)} → {FINAL_TMPL.format(label='<方法>', s=s)}")
        return

    b3 = importlib.import_module(B3_MODULE)
    b4 = importlib.import_module(B4_MODULE)

    raw, b3_rows = {}, {}
    for s in TRAIN_SEEDS:
        print(f"\n################ seed {s}：B3 ################")
        b3_out = run_b3(b3, s)
        b3_rows[s] = b3_metrics(b3_out)
        print(f"\n################ seed {s}：B4 ################")
        raw[s] = run_b4(b4, s, b3_out)

    aligned, report, labels = align(b4, raw)

    written = []
    for s in TRAIN_SEEDS:
        for lb in labels:
            p = FINAL_TMPL.format(label=lb, s=s)
            aligned[s][lb].to_excel(p, index=False)
            written.append(p)

    # 驗證：各 seed 的 檔名 + fold 完全相同
    ref = aligned[TRAIN_SEEDS[0]][labels[0]][["圖片檔名", "折數fold"]]
    for s in TRAIN_SEEDS[1:]:
        if not aligned[s][labels[0]][["圖片檔名", "折數fold"]].equals(ref):
            raise RuntimeError(f"seed {s} 對齊後檔名/fold 與 seed {TRAIN_SEEDS[0]} 不同")

    rows = [{"train_seed": s, **b3_rows[s], **aligned_metrics(b4, aligned[s])}
            for s in TRAIN_SEEDS]
    sdf = pd.DataFrame(rows)
    num = [c for c in sdf.columns if c != "train_seed"]
    stat = pd.DataFrame({"mean": sdf[num].mean(), "SD": sdf[num].std(ddof=1),
                         "n_seed": sdf[num].count()})

    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as w:
        sdf.to_excel(w, sheet_name="各seed", index=False)
        stat.to_excel(w, sheet_name="mean±SD")
        report.to_excel(w, sheet_name="對齊報告", index=False)

    # ---- 終端摘要 ----
    n_common = int(report["對齊後張數"].iloc[0])
    print(f"\n=================== 跨 {len(TRAIN_SEEDS)} 個 seed ===================")
    print(report[["train_seed", "B4原始張數", "對齊後張數", "因對齊排除"]].to_string(index=False))
    if report["因對齊排除"].sum():
        print(f"⚠️ 有 seed 的可用牙不同，已取交集 {n_common} 張重新分摺；"
              f"n 比單跑 B4 少，舊結果需用新檔重跑才可比")
    else:
        print(f"✅ 所有 seed 可用牙相同({n_common} 張)，fold 與 B4 單跑結果一致")

    print("\n各指標 mean ± SD(跨 seed)：")
    for c in num:
        m, sd = sdf[c].mean(), sdf[c].std(ddof=1)
        print(f"   {c}: {m:.4f} ± {sd:.4f}" if len(sdf) > 1 else f"   {c}: {m:.4f}")
    print("   ⚠️ B3_長度MAE 是 mask 對 GT mask 的長度誤差(px)；最終指標是 C1 的 MAE(mm)。")

    if "牙位基準" in labels and len(TRAIN_SEEDS) > 1:
        # 只比 C1 會讀的欄；QC備註、冠寬等附帶欄本來就隨 seed 變
        core = ["圖片檔名", "填充物長度(mm)", "像素長度", "折數fold"]
        base = aligned[TRAIN_SEEDS[0]]["牙位基準"][core]
        same = all(aligned[s]["牙位基準"][core].equals(base) for s in TRAIN_SEEDS[1:])
        print(f"\n牙位基準 跨 seed {'完全相同 → C1 只需跑一次' if same else '⚠️ 不同，請檢查'}")

    print(f"\n📄 {SUMMARY_XLSX}")
    print(f"📄 C1 輸入 {len(written)} 份，C1 的 METHOD 依序設成：")
    for lb in labels:
        print(f"   {', '.join(f'{lb}_seed{s}' for s in TRAIN_SEEDS)}")


if __name__ == "__main__":
    main()
