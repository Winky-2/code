"""
A1r_check_output.py
===================
驗證 A1r 的輸出「真的是無CLAHE」：比較 輸出圖 vs 原CLAHE資料集圖 的平均灰階差，
並依「配到的原圖是 Roboflow 匯出檔(.rf.) 還是一般原圖」分兩組看。

判讀：
  - 兩組的差都明顯 (≳5 灰階) 且接近 → 兩種來源都是無CLAHE原圖 ✅
  - .rf. 組的差很小 (≈1~2) → Roboflow 那批其實是CLAHE圖，輸出仍是CLAHE ❌
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---------------- 設定（跟 A1r 用的一樣）----------------
CLAHE_SET_DIR = r"data set/cropped_train_A1"
OUTPUT_DIR = r"A1r_train_noCLAHE"


def imread_u(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def main():
    out_root, ds_root = Path(OUTPUT_DIR), Path(CLAHE_SET_DIR)
    df = pd.read_csv(out_root / "A1r_配對QC.csv", encoding="utf-8-sig")
    df = df[df["狀態"].astype(str).str.startswith("✅")].copy()
    diffs = []
    for _, r in df.iterrows():
        o, d = imread_u(out_root / r["資料集圖片"]), imread_u(ds_root / r["資料集圖片"])
        m = d > 5
        diffs.append(np.abs(o.astype(int) - d.astype(int))[m].mean())
    df["平均灰階差"] = diffs
    df["來源"] = np.where(df["對應原圖"].astype(str).str.contains(r"\.rf\."), "Roboflow匯出(.rf.)", "一般原圖")
    print(df.groupby("來源")["平均灰階差"].describe()[["count", "min", "50%", "max"]].round(2))
    low = df[df["平均灰階差"] < 3]
    print(f"\n平均灰階差 < 3 的張數: {len(low)}（這些幾乎等於CLAHE圖，可疑）")
    if len(low):
        print(low[["資料集圖片", "對應原圖", "平均灰階差"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
