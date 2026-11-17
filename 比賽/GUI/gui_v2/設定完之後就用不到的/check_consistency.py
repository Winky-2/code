#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_consistency.py — 確認 GUI 算的特徵 = 訓練 Excel 的「像素長度」
===============================================================
GPR 是用 B7 那份 Excel 訓練的。GUI 對同一顆牙算出的特徵如果跟 Excel 不同
(top-hat 參數、補邊比例、scale 換算、正規化分母任何一個不一致)，預測就會
整體偏掉，而且不會報錯。

做法：拿論文用的那 67 張「A4 裁切後、top-hat 前」的單顆牙圖跑 pipeline.measure()，
     正規化分母用對應的整張 X 光長邊(同 B7)，三種 LENGTH_SCALE_MODE 都算，
     跟 Excel 比，印出中位數相對差 → 把 pipeline.py 設成差最小的(應 < 1%)。

只驗證「裁切之後」的部分；偵測與裁切由使用者在 GUI 上點選，跟論文的人工挑牙對應。
個別圖差很大通常是 center 挑到不同區塊(B3 有 GT 時用 gt_iou)，看中位數即可。

用法：python check_consistency.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P

# ==================== 設定 ====================
TRAIN_XLSX = "根管填充物像素長度_已配對_11n_tophat_mask幾何_長邊正規化.xlsx"
CROP_DIR = "cropped_test"          # TODO：A4 裁切 + 人工挑選後、top-hat 之前的單顆牙圖
FULL_IMAGE_DIR = "Data_test"   # TODO：整張 X 光(= B7 的 FULL_IMAGE_DIR)
OUT_XLSX = "check_consistency_結果.xlsx"
OUTLIER_REL = 0.05
# ==============================================

MODES = ("b3_original", "b3_padded", "exact")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HERE = Path(__file__).resolve().parent
NORMALIZED = "長邊正規化" in TRAIN_XLSX


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def index_dir(d):
    out = {}
    for p in resolve(d).rglob("*"):
        if p.suffix.lower() in EXTS:
            out.setdefault(P.name_key(p.name), []).append(p)
    return out


def main():
    xlsx = resolve(TRAIN_XLSX)
    for name, d in (("TRAIN_XLSX", xlsx), ("CROP_DIR", resolve(CROP_DIR)), ("FULL_IMAGE_DIR", resolve(FULL_IMAGE_DIR))):
        if not d.exists():
            raise FileNotFoundError(f"找不到 {name}：{d}（可填絕對路徑）")
    df = pd.read_excel(xlsx)
    crops, fulls = index_dir(CROP_DIR), index_dir(FULL_IMAGE_DIR)
    print(f"Excel {len(df)} 列；裁切圖 {sum(map(len, crops.values()))} 張；整張 {len(fulls)} 張；"
          f"正規化={NORMALIZED}")
    pos_col = "牙位" if "牙位" in df.columns else None

    rows = []
    for _, r in df.iterrows():
        k = P.name_key(r["圖片檔名"])
        rec = {"圖片檔名": r["圖片檔名"], "Excel像素長度": r["像素長度"]}
        if k not in crops or k not in fulls:
            rows.append({**rec, "狀態": "找不到裁切圖" if k not in crops else "找不到整張圖"})
            continue
        full = P.imread_u(fulls[k][0])
        Hf, Wf = full.shape[:2]
        pos = int(r[pos_col]) if pos_col and pd.notna(r[pos_col]) else None
        best = None
        for cp in crops[k]:                       # 同一張有多顆裁切圖時取最接近的(並標記)
            m = P.measure(P.imread_u(cp), pos, normalized=False)
            if not m["ok"]:
                continue
            feats = {}
            for mode in MODES:
                L = P.lb_length_to_orig(m["len_lb"], m["A"], m["B"], m["g"], m["W0"], m["H0"], mode)
                feats[mode] = L * P.NORM_REF / max(Wf, Hf) if NORMALIZED else L
            err = min(abs(v / r["像素長度"] - 1) for v in feats.values())
            if best is None or err < best[0]:
                best = (err, cp, feats, m)
        if best is None:
            rows.append({**rec, "狀態": "GUI 沒分割到牙"})
            continue
        _, cp, feats, m = best
        rec.update({"裁切圖": cp.name, "裁切圖數": len(crops[k]), "mask數": len(m["polys"])})
        for mode, v in feats.items():
            rec[f"GUI_{mode}"] = round(v, 4)
            rec[f"相對差_{mode}"] = round(v / r["像素長度"] - 1, 4)
        rec["狀態"] = "✅" if len(crops[k]) == 1 else "✅(多張裁切圖，取最接近)"
        rows.append(rec)

    out = pd.DataFrame(rows)
    ok = out[out["狀態"].astype(str).str.startswith("✅")]
    print(f"比對 {len(ok)}/{len(out)} 張  (PREPROCESS={P.PREPROCESS}, K={P.TOPHAT_KSIZE}, PAD_TB_RATIO={P.PAD_TB_RATIO})")
    print("  狀態統計：", out["狀態"].value_counts().to_dict(), "\n")
    out.to_excel(resolve(OUT_XLSX), index=False)
    if ok.empty:
        print("❌ 沒有可比對的圖：檢查 CROP_DIR / FULL_IMAGE_DIR 與檔名開頭編號")
        return
    best, best_v = None, np.inf
    for mode in MODES:
        d = ok[f"相對差_{mode}"].abs()
        print(f"  {mode:12s} 中位數相對差 {d.median()*100:6.2f}%   ≤1% 的張數 {(d <= 0.01).sum()}/{len(d)}")
        if d.median() < best_v:
            best, best_v = mode, d.median()
    print(f"\n👉 建議 LENGTH_SCALE_MODE = \"{best}\"（目前 pipeline.py 設 \"{P.LENGTH_SCALE_MODE}\"）")
    if best_v > 0.01:
        print("⚠️ 中位數相對差仍 > 1%：依序檢查 CROP_DIR 是否為 top-hat 前的圖 → TOPHAT 參數 → "
              "PAD_TB_RATIO → SEG_WEIGHTS（一次改一個）。")
    print(f"📄 逐張結果：{OUT_XLSX}")


if __name__ == "__main__":
    main()
