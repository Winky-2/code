"""
B7_size_normalize_v2.py
===============================================================
問題：px/mm 在全體 CV 0.41，但同一種「整張 X 光尺寸」內只有 0.05–0.10，
      且 px/mm 約與整張長邊成正比 → 不同解析度匯出造成的刻度差。
做法：X_新 = 像素長度 × (REF ÷ 整張尺度)    (= 換算成「長邊 REF px」等效像素)
      只用影像本身的尺寸、沒用到醫師 mm → 不會資料洩漏
      複製既有 C1 輸入檔，只換「像素長度」欄 → 列、順序、fold 完全相同(單一變因)

*** v2 變更：多 seed ***
輸入改讀 B34_run_seeds.py 的對齊後輸出：根管填充物像素長度_已配對_{方法}_seed{s}.xlsx
每個 seed 各產一份：根管填充物像素長度_已配對_{方法}_{尺度}正規化_seed{s}.xlsx
  → C1 的 METHOD = '{方法}_{尺度}正規化_seed{s}'
B34 已保證各 seed 列/fold 相同，本檔再檢查一次。
TRAIN_SEEDS = None 退回舊版行為：讀不帶 seed 的單一檔。
只正規化 mask幾何：冠寬比例尺是長寬比(無刻度)，牙位基準不看影像，兩者都不該乘尺寸。

產出：直接寫在本檔所在資料夾(code\，跟 B4/B34 的輸出放一起)
  根管填充物像素長度_已配對_<方法>_<尺度>正規化[_seed{s}].xlsx
  B7_正規化摘要.xlsx(三種尺度 px/mm CV 的跨 seed 表、逐尺寸分組)、B7_config.json
  檔名都帶「正規化」/「B7_」，不會蓋到 B4/B34 或其他既有檔

操作：
1. FULL_IMAGE_DIR 填 test.py 用的那個整張 X 光資料夾
2. 先直接跑(RUN_NOW=False)：只印三種尺度的 CV，不寫檔
3. 選好 NORM 後改 RUN_NOW=True 再跑
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ==================== 可手動修改 ====================
RUN_NOW = True

FULL_IMAGE_DIR = r"data set/Data_test"      # 👈 跟 test.py 的 RAW_DIR 一樣
TRAIN_SEEDS = [0, 1, 2, 3, 4]               # 👈 與 B34 一致；None = 舊版單一檔
METHODS = ["mask幾何"]                      # 只放有 px 刻度的方法
INPUT_TMPL = "根管填充物像素長度_已配對_{label}_seed{s}.xlsx"   # B34 的輸出(只讀)
INPUT_TMPL_NOSEED = "根管填充物像素長度_已配對_{label}.xlsx"    # TRAIN_SEEDS=None 時用
NORM = "長邊"            # "長邊" | "短邊" | "對角線"
REF = 1200               # 換算成長邊 1200 px 等效，只為數字好讀；C1 斜率會吸收常數
# ===================================================

HERE = Path(__file__).resolve().parent

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_NUM = re.compile(r"^\s*0*(\d+)")
SCALES = {"長邊": lambda w, h: max(w, h),
          "短邊": lambda w, h: min(w, h),
          "對角線": lambda w, h: float(np.hypot(w, h))}


def key(name):
    m = _NUM.match(Path(str(name)).stem)
    return str(int(m.group(1))) if m else Path(str(name)).stem.lower()


def build_inputs():
    """→ {(方法, seed或None): 檔案路徑}"""
    if TRAIN_SEEDS is None:
        return {(lb, None): INPUT_TMPL_NOSEED.format(label=lb) for lb in METHODS}
    if not TRAIN_SEEDS:
        raise ValueError("TRAIN_SEEDS 至少要有一個值(或設 None 用舊版單一檔)")
    return {(lb, s): INPUT_TMPL.format(label=lb, s=s) for lb in METHODS for s in TRAIN_SEEDS}


def method_name(label, s):
    base = f"{label}_{NORM}正規化"
    return base if s is None else f"{base}_seed{s}"


def read_sizes():
    d = Path(FULL_IMAGE_DIR)
    if not d.is_dir():
        raise FileNotFoundError(f"找不到資料夾 {d.resolve()}：FULL_IMAGE_DIR 要填整張 X 光的「資料夾」")
    sizes = {}
    for p in d.iterdir():
        if p.suffix.lower() in EXTS:
            with Image.open(p) as im:
                sizes[key(p.name)] = im.size
    return sizes


def attach(df, sizes):
    df = df.copy()
    wh = df["圖片檔名"].map(lambda s: sizes.get(key(s)))
    miss = df.loc[wh.isna(), "圖片檔名"].tolist()
    if miss:
        raise KeyError(f"{len(miss)} 張對不到整張影像(列數必須不變才是單一變因)：{miss[:5]}")
    df["整張寬"] = [w for w, _ in wh]
    df["整張高"] = [h for _, h in wh]
    df["整張尺寸"] = [f"{max(w, h)}x{min(w, h)}" for w, h in wh]
    return df


def cv(s):
    return float(s.std(ddof=1) / s.mean())


def scale_table(df):
    """三種尺度正規化後的 px/mm CV(只是診斷；選哪個請事先決定，別逐一試 C1 挑最好)。"""
    y = df["填充物長度(mm)"]
    rows = [{"尺度": "不正規化", "px/mm CV": cv(df["像素長度"] / y),
             "X–Y r": df["像素長度"].corr(y)}]
    for name, f in SCALES.items():
        x = df["像素長度"] * REF / [f(w, h) for w, h in zip(df["整張寬"], df["整張高"])]
        rows.append({"尺度": name, "px/mm CV": cv(x / y), "X–Y r": x.corr(y)})
    return pd.DataFrame(rows).round(3)


def check_aligned(data):
    """同一方法下各 seed 的 檔名 + fold 必須相同(B34 已保證，這裡再驗一次)。"""
    for lb in METHODS:
        frames = [(s, df) for (l, s), df in data.items() if l == lb and s is not None]
        if len(frames) < 2:
            continue
        s0, ref = frames[0]
        cols = ["圖片檔名"] + (["折數fold"] if "折數fold" in ref.columns else [])
        for s, df in frames[1:]:
            if not df[cols].reset_index(drop=True).equals(ref[cols].reset_index(drop=True)):
                raise RuntimeError(f"{lb}：seed {s} 與 seed {s0} 的列/fold 不同，"
                                   f"請改用 B34 的對齊後輸出")
        print(f"✅ {lb}：{len(frames)} 個 seed 的列與 fold 完全相同(n={len(ref)})")


def main():
    import os
    os.chdir(HERE)                    # 相對路徑一律以 code\ 為準，從哪裡執行都一樣
    inputs = build_inputs()
    sizes = read_sizes()

    data, diag = {}, []
    for (label, s), path in inputs.items():
        if not Path(path).exists():
            print(f"⚠️ 略過 {label} seed {s}：找不到 {path}")
            continue
        data[(label, s)] = attach(pd.read_excel(path), sizes)
        diag.append(scale_table(data[(label, s)]).assign(方法=label, train_seed=s))
    if not data:
        raise FileNotFoundError("輸入檔一個都找不到(先跑 B34_run_seeds.py)")
    check_aligned(data)

    diag = pd.concat(diag, ignore_index=True)
    diag_cols = ["方法", "train_seed", "尺度", "px/mm CV", "X–Y r"]
    diag = diag[diag_cols]
    # 跨 seed：每種尺度的 CV / r 平均 ± SD(單一 seed 時 SD 為 NaN)
    diag_ms = (diag.groupby(["方法", "尺度"], sort=False)[["px/mm CV", "X–Y r"]]
                   .agg(["mean", "std"]).round(4))
    print("\n=== 三種尺度的 px/mm CV(跨 seed mean / std) ===")
    print(diag_ms.to_string())

    if not RUN_NOW:
        print(f"\nRUN_NOW=False：只檢查，未寫檔。目前 NORM = {NORM}")
        return

    f = SCALES[NORM]
    out = HERE
    protected = {Path(p).resolve() for p in inputs.values()}
    summary, after = {}, []
    for (label, s), df in data.items():
        df = df.copy()
        df["像素長度_未正規化"] = df["像素長度"]
        df["像素長度"] = (df["像素長度"] * REF /
                       [f(w, h) for w, h in zip(df["整張寬"], df["整張高"])]).round(4)
        method = method_name(label, s)
        if "長度方法" in df.columns:
            df["長度方法"] = method
        dst = out / f"根管填充物像素長度_已配對_{method}.xlsx"
        if dst.resolve() in protected:
            raise RuntimeError(f"輸出檔 {dst.name} 跟輸入檔同名，會蓋掉原檔，已停止")
        if dst.exists():
            print(f"♻️  覆寫 B7 之前的產出：{dst.name}")
        df.to_excel(dst, index=False)

        y = df["填充物長度(mm)"]
        g = (df.assign(px每mm=df["像素長度"] / y)
               .groupby("整張尺寸")["px每mm"]
               .agg(張數="size", 中位數="median", CV=cv)
               .sort_values("張數", ascending=False).round(3))
        summary[method] = g
        after.append({"方法": label, "train_seed": s,
                      "CV_未正規化": round(cv(df["像素長度_未正規化"] / y), 4),
                      "CV_正規化後": round(cv(df["像素長度"] / y), 4),
                      "r_未正規化": round(df["像素長度_未正規化"].corr(y), 4),
                      "r_正規化後": round(df["像素長度"].corr(y), 4)})
    after = pd.DataFrame(after)

    with pd.ExcelWriter(out / "B7_正規化摘要.xlsx", engine="openpyxl") as w:
        after.to_excel(w, sheet_name=f"{NORM}正規化_前後", index=False)
        diag.to_excel(w, sheet_name="三種尺度_各seed", index=False)
        diag_ms.to_excel(w, sheet_name="三種尺度_跨seed")
        for method, g in summary.items():
            g.to_excel(w, sheet_name=f"{method}_分組"[:31])
    json.dump({"FULL_IMAGE_DIR": FULL_IMAGE_DIR, "TRAIN_SEEDS": TRAIN_SEEDS,
               "METHODS": METHODS, "NORM": NORM, "REF": REF,
               "INPUTS": {f"{lb}|{s}": p for (lb, s), p in inputs.items()}},
              open(out / "B7_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n=== {NORM}正規化 前 → 後 ===")
    print(after.to_string(index=False))
    if len(after) > 1:
        for c in ("CV_正規化後", "r_正規化後"):
            print(f"   {c}: {after[c].mean():.4f} ± {after[c].std(ddof=1):.4f}")

    print(f"\n📁 輸出在 {out}(跟 B4/B34 的檔案同一層)")
    print("👉 MATLAB：")
    print(f"   cd('{out.as_posix()}');")
    for (label, s) in data:
        print(f"   C1 的 METHOD = '{method_name(label, s)}'")
    if TRAIN_SEEDS is None:
        print("   對照組：同一份資料未正規化的 C1 結果(METHOD = 'mask幾何')")
    else:
        print("   對照組：同一 seed 未正規化的 C1 結果(METHOD = 'mask幾何_seed{s}')，"
              "兩邊都報跨 seed mean ± SD")


if __name__ == "__main__":
    print(__doc__)
    main()
