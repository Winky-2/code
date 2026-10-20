"""
B7_size_normalize.py
===============================================================
問題：px/mm 在全體 CV 0.41，但同一種「整張 X 光尺寸」內只有 0.05–0.10，
      且 px/mm 約與整張長邊成正比 → 不同解析度匯出造成的刻度差。
做法：X_新 = 像素長度 × (REF ÷ 整張尺度)    (= 換算成「長邊 REF px」等效像素)
      只用影像本身的尺寸、沒用到醫師 mm → 不會資料洩漏
      複製既有 C1 輸入檔，只換「像素長度」欄 → 列、順序、fold 完全相同(單一變因)

產出：直接寫在本檔所在資料夾(code\，跟 B4 的輸出放一起)
  根管填充物像素長度_已配對_<原方法>_<尺度>正規化.xlsx   ← C1 的 METHOD = 檔名後綴
  B7_正規化摘要.xlsx(三種尺度的 px/mm CV、逐尺寸分組)、B7_config.json
  檔名都帶「正規化」/「B7_」，不會蓋到 B4 或其他既有檔；重跑 B7 只會覆寫 B7 自己的產出

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
INPUTS = {                                          # 原方法名 : 既有 C1 輸入檔(只讀)
    "mask幾何": "根管填充物像素長度_已配對_mask幾何.xlsx",
}
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


def main():
    import os
    os.chdir(HERE)                    # 相對路徑一律以 code\ 為準，從哪裡執行都一樣
    sizes = read_sizes()
    data = {}
    for label, path in INPUTS.items():
        if not Path(path).exists():
            print(f"⚠️ 略過 {label}：找不到 {path}")
            continue
        data[label] = attach(pd.read_excel(path), sizes)
        print(f"\n=== {label}(n={len(data[label])}) ===")
        print(scale_table(data[label]).to_string(index=False))
    if not data:
        raise FileNotFoundError("INPUTS 裡的檔案一個都找不到")

    if not RUN_NOW:
        print(f"\nRUN_NOW=False：只檢查，未寫檔。目前 NORM = {NORM}")
        return

    f = SCALES[NORM]
    out = HERE
    protected = {Path(p).resolve() for p in INPUTS.values()}
    summary = {}
    for label, df in data.items():
        df = df.copy()
        df["像素長度_未正規化"] = df["像素長度"]
        df["像素長度"] = (df["像素長度"] * REF /
                       [f(w, h) for w, h in zip(df["整張寬"], df["整張高"])]).round(4)
        method = f"{label}_{NORM}正規化"
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
        print(f"\n📄 {method}：px/mm CV {cv(df['像素長度_未正規化'] / y):.3f} → "
              f"{cv(df['像素長度'] / y):.3f}，X–Y r {df['像素長度'].corr(y):.3f}")
        print(g.to_string())

    with pd.ExcelWriter(out / "B7_正規化摘要.xlsx", engine="openpyxl") as w:
        for label, df in data.items():
            scale_table(df).to_excel(w, sheet_name=f"{label}_三種尺度"[:31], index=False)
        for method, g in summary.items():
            g.to_excel(w, sheet_name=f"{method}_分組"[:31])
    json.dump({"FULL_IMAGE_DIR": FULL_IMAGE_DIR, "INPUTS": INPUTS, "NORM": NORM, "REF": REF},
              open(out / "B7_config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n📁 輸出在 {out}(跟 B4 的檔案同一層)")
    print("👉 MATLAB：")
    print(f"   cd('{out.as_posix()}');")
    for label in data:
        print(f"   C1 的 METHOD = '{label}_{NORM}正規化'")
    print("   對照：同一份資料未正規化的 C1 結果(mask幾何 1.321 / GT_mask幾何 1.289 mm)")


if __name__ == "__main__":
    print(__doc__)
    main()