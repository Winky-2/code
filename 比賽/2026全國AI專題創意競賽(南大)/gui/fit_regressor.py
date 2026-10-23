#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fit_regressor.py — 產生 GUI 用的 reg_model.json(線性版)
===============================================================
為什麼需要這支：C1 是 MATLAB 的 k-fold 評估，每一摺訓練完就丟掉，
沒有存下可以給新影像用的「最終模型」。GUI 需要一個固定的模型。

這支用 C1 的線性對照組(D 列)同一個公式：Y = b0 + b1·X (+ b2·年齡)，
在「全部」資料上擬合一次存成 JSON。另外照 C1 的 折數fold 做一次 CV，
印出的 MAE 應該跟 C1 輸出表 D 列(線性、未校正)非常接近 → 用來確認讀對檔。

想讓 GUI 用跟論文主結果一樣的 ANN：改跑 MATLAB 的 C1_export_final_ann.m，
它會覆寫同一個 reg_model.json。

用法：python fit_regressor.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ==================== 設定 ====================
RUN_NOW = True
# TODO 5：B7 產生、C1 讀的那份檔
TRAIN_XLSX = "根管填充物像素長度_已配對_26n_mask幾何_長邊正規化.xlsx"
USE_AGE = True                    # 論文主結果 = 長邊正規化 + 年齡
FOLD_COLUMN = "折數fold"
OUT_JSON = "reg_model.json"
OFFSET_MM = 0.5                   # 只用來印校正後指標，跟 C1 一致
# ==============================================


def design(df):
    cols = ["像素長度"] + (["年齡"] if USE_AGE else [])
    return df[cols].to_numpy(float), cols


def fit(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    return b


def main():
    p = Path(TRAIN_XLSX)
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p.resolve()}（B7 的輸出）")
    df = pd.read_excel(p)
    need = ["像素長度", "填充物長度(mm)"] + (["年齡"] if USE_AGE else [])
    df = df.dropna(subset=need)
    X, cols = design(df)
    y = df["填充物長度(mm)"].to_numpy(float)

    # ---- 用 C1 同一組 fold 做 CV，對照 C1 輸出表 D 列 ----
    if FOLD_COLUMN in df.columns:
        fold = df[FOLD_COLUMN].to_numpy(int)
        pred = np.full(len(y), np.nan)
        for k in np.unique(fold):
            te = fold == k
            b = fit(X[~te], y[~te])
            pred[te] = b[0] + X[te] @ b[1:]
        res = pred - y
        off = res - OFFSET_MM
        print(f"[CV，{len(np.unique(fold))} 摺，n={len(y)}] 對照 C1 的 D/E 列(線性)：")
        print(f"  未校正 MAE {np.abs(res).mean():.3f}  RMSE {np.sqrt((res**2).mean()):.3f}  "
              f"R {np.corrcoef(pred, y)[0, 1]:.3f}")
        print(f"  校正後 MAE {np.abs(off).mean():.3f}  理想比率 {(np.abs(off) <= 1.0).mean()*100:.1f}%")

    b = fit(X, y)
    print(f"\n[全資料擬合] 截距 {b[0]:.4f}，" + "，".join(f"{c} {v:.5f}" for c, v in zip(cols, b[1:])))

    if not RUN_NOW:
        print("RUN_NOW=False：未寫檔")
        return
    out = {"kind": "linear", "use_age": USE_AGE, "features": cols,
           "intercept": float(b[0]), "coef": [float(v) for v in b[1:]],
           "n": int(len(y)), "source": p.name,
           "time": datetime.now().isoformat(timespec="seconds")}
    Path(OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已寫出 {OUT_JSON}")


if __name__ == "__main__":
    main()
