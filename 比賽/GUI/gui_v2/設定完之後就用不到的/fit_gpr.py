#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fit_gpr.py — 產生 GUI 用的 gpr_model.json(Python 版，沒有 MATLAB 時用)
===============================================================
✅ 首選：在 MATLAB 跑 C1_export_final_gpr.m，用 fitrgp 本身訓練並匯出，跟論文同一套。
   這支是備案：用 gpr_core.py 重現同一個模型(SE kernel + linear basis + 手動標準化)，
   最佳化器不同，超參數會跟 MATLAB 有些微差異。

1) 用 C1 同一組「折數fold」做 CV，印出跟 C1_pxToMm_GPR.m「C 列」同定義的指標
   → 跟論文數字比，接近才代表設定對。
2) 在全部資料上擬合一次，寫出 gpr_model.json。

設定請跟 C1_pxToMm_GPR.m 保持一致。
用法：python fit_gpr.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import gpr_core as G

HERE = Path(__file__).resolve().parent

# ==================== 設定(= C1_pxToMm_GPR.m) ====================
METHOD = "mask幾何_長邊正規化"            # 論文主結果；對照組用 "mask幾何"
TRAIN_XLSX = f"根管填充物像素長度_已配對_11n_tophat_{METHOD}.xlsx"
USE_AGE = True
GPR_KERNEL = "squaredexponential"
GPR_BASIS = "linear"
FOLD_COLUMN = "折數fold"
OFFSET_MM = 0.5
IDEAL_TOL_MM = 1.0
OUT_JSON = "gpr_model.json"
# =================================================================


def standardize(Xtr):
    mu, sg = Xtr.mean(0), Xtr.std(0, ddof=1)
    sg[sg == 0] = 1
    return mu, sg


def train(X, y):
    mu, sg = standardize(X)
    Xn = (X - mu) / sg
    f = G.fit(Xn, y, GPR_KERNEL, GPR_BASIS)
    return G.Predictor(Xn, y, GPR_KERNEL, GPR_BASIS, f["theta"], f["sigma"], f["beta"], mu, sg), f


def metrics(pred, y, sd):
    off = pred - OFFSET_MM - y
    return {"MAE": np.abs(off).mean(), "RMSE": np.sqrt((off ** 2).mean()), "Bias": off.mean(),
            "R": np.corrcoef(pred, y)[0, 1], "R2": np.corrcoef(pred, y)[0, 1] ** 2,
            "理想比率%": (np.abs(off) <= IDEAL_TOL_MM).mean() * 100,
            "平均預測σ": sd.mean()}


def main():
    p = Path(TRAIN_XLSX)
    p = p if p.is_absolute() else HERE / p
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}\n   → 改 TRAIN_XLSX(B4/B7 的輸出)")
    df = pd.read_excel(p)
    cols = ["像素長度"] + (["年齡"] if USE_AGE else [])
    df = df.dropna(subset=cols + ["填充物長度(mm)"])
    X = df[cols].to_numpy(float)
    y = df["填充物長度(mm)"].to_numpy(float)
    print(f"{p.name}：n={len(y)}，特徵={cols}，kernel={GPR_KERNEL}，basis={GPR_BASIS}")

    cv = {}
    if FOLD_COLUMN in df.columns:
        fold = df[FOLD_COLUMN].to_numpy(int)
        pred, sd = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for k in np.unique(fold):
            te = fold == k
            m, f = train(X[~te], y[~te])
            pred[te], sd[te] = m.predict(X[te])
            print(f"  摺 {k}：σL={np.round(f['theta'][:-1], 3)}  σf={f['theta'][-1]:.3f}  σn={f['sigma']:.3f}")
        cv = {k: round(float(v), 3) for k, v in metrics(pred, y, sd).items()}
        print(f"\n[CV，校正後 −{OFFSET_MM} mm] → 對照 C1_pxToMm_GPR.m 的 C 列：")
        for k, v in cv.items():
            print(f"  {k:10s} {v}")

    model, f = train(X, y)
    print(f"\n[全資料] σL={np.round(f['theta'][:-1], 3)}  σf={f['theta'][-1]:.3f}  σn={f['sigma']:.3f}  β={np.round(f['beta'], 3)}")
    if np.median(f["theta"][:-1]) > 10:
        print("ℹ️ σL > 10：GP 幾乎退化成線性(n 小時常見，不是跑錯)")

    idx = np.linspace(0, len(y) - 1, min(5, len(y))).astype(int)
    m_st, s_st = model.predict(X[idx])
    out = {"kind": "gpr", "trained_by": "python gpr_core", "method": METHOD,
           "normalized": "長邊正規化" in METHOD, "use_age": USE_AGE, "features": cols,
           "kernel": GPR_KERNEL, "basis": GPR_BASIS, "n": int(len(y)),
           "mu": model.mu.tolist(), "sg": model.sg.tolist(), "Xn": model.Xn.tolist(), "y": y.tolist(),
           "theta": f["theta"].tolist(), "sigma": f["sigma"], "beta": f["beta"].tolist(),
           "feat_range": [float(X[:, 0].min()), float(X[:, 0].max())],
           "selftest": [{"x": X[i].tolist(), "mean": float(a), "sd": float(b)} for i, a, b in zip(idx, m_st, s_st)],
           "cv": cv, "source": p.name, "time": datetime.now().isoformat(timespec="seconds")}
    if USE_AGE:
        out["age_range"] = [float(X[:, 1].min()), float(X[:, 1].max())]
    (HERE / OUT_JSON).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"✅ 已寫出 {OUT_JSON}")


if __name__ == "__main__":
    main()
