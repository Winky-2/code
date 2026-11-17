#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
check_consistency.py — 確認 GUI 算出的特徵 = 訓練 Excel 的「像素長度」
===============================================================
GUI 的迴歸模型是用 B7 那份 Excel 訓練的。如果 GUI 對同一張原圖算出來的
特徵跟 Excel 不一樣(CLAHE 開關、補邊比例、scale 換算任何一個不同)，
預測值就會整體偏掉，而且不會報錯。

這支對 Excel 裡每一張原圖跑一次 pipeline.measure()，三種 LENGTH_SCALE_MODE
都算，跟 Excel 的「像素長度」比，印出各模式的中位數相對差。
→ 把 pipeline.py 的 LENGTH_SCALE_MODE 設成差最小的那個(應該 < 1%)。

個別圖差很大通常是挑到不同顆牙：B3 有 GT 時用 gt_iou 挑牙，GUI 沒有 GT
只能用 center。這些圖會在輸出表標「疑似挑牙不同」，正常現象，看中位數即可。

用法：python check_consistency.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import pipeline as P

# ==================== 設定 ====================
TRAIN_XLSX = "根管填充物像素長度_已配對_mask幾何_長邊正規化.xlsx"   # 同 fit_regressor.py
RAW_IMAGE_DIR = "Data_test"     # TODO 6：未 CLAHE、未 letterbox 的整張原圖(= B7 的 FULL_IMAGE_DIR)
OUT_XLSX = "check_consistency_結果.xlsx"
OUTLIER_REL = 0.05                       # 單張相對差 > 5% → 疑似挑牙不同
# ==============================================

MODES = ("b3_original", "b3_padded", "exact")


HERE = Path(__file__).resolve().parent


def resolve(p):
    """相對路徑一律以本檔所在資料夾為準(VS Code 的執行目錄常常不是這裡)。"""
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def main():
    xlsx, raw_dir = resolve(TRAIN_XLSX), resolve(RAW_IMAGE_DIR)
    if not xlsx.exists():
        raise FileNotFoundError(f"找不到訓練 Excel：{xlsx}\n   → 改 TRAIN_XLSX(可填絕對路徑)")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"找不到原圖資料夾：{raw_dir}\n   → 改 RAW_IMAGE_DIR(可填絕對路徑)")
    df = pd.read_excel(xlsx)
    raw = {P.name_key(p.name): p for p in raw_dir.rglob("*")
           if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}}
    print(f"Excel {len(df)} 列；原圖資料夾 {len(raw)} 張（{raw_dir}）")
    print(f"  Excel 檔名 key 範例：{[P.name_key(n) for n in df['圖片檔名'].head(5)]}")
    print(f"  原圖檔名 key 範例 ：{sorted(raw)[:5]}")
    pos_col = "牙位" if "牙位" in df.columns else None

    rows = []
    for _, r in df.iterrows():
        k = P.name_key(r["圖片檔名"])
        rec = {"圖片檔名": r["圖片檔名"], "Excel像素長度": r["像素長度"]}
        if k not in raw:
            rows.append({**rec, "狀態": "找不到原圖"})
            continue
        img = P.imread_u(str(raw[k]))
        pos = int(r[pos_col]) if pos_col and pd.notna(r[pos_col]) else None
        m = P.measure(img, pos, scale_mode="b3_padded")
        if not m["ok"]:
            rows.append({**rec, "狀態": "GUI 沒偵測到牙"})
            continue
        for mode in MODES:
            L = P.lb_length_to_orig(m["len_lb"], m["A"], m["B"], m["g"], m["W0"], m["H0"], mode)
            feat = L * P.NORM_REF / max(m["W0"], m["H0"])
            rec[f"GUI_{mode}"] = round(feat, 4)
            rec[f"相對差_{mode}"] = round(feat / r["像素長度"] - 1, 4)
        rec["mask數"] = len(m["polys"])
        rec["狀態"] = "✅"
        rows.append(rec)

    out = pd.DataFrame(rows)
    ok = out[out["狀態"] == "✅"]
    print(f"比對 {len(ok)}/{len(out)} 張  (USE_CLAHE={P.USE_CLAHE}, PAD_TB_RATIO={P.PAD_TB_RATIO})")
    print("  狀態統計：", out["狀態"].value_counts().to_dict(), "\n")
    if ok.empty:
        out.to_excel(resolve(OUT_XLSX), index=False)
        bad = out["狀態"].value_counts().idxmax()
        if bad == "找不到原圖":
            print("❌ 檔名對不上：比較上面兩行 key 範例。RAW_IMAGE_DIR 要指向「整張原圖」，"
                  "不是 A5 的 640 圖或 Roboflow 匯出(檔名帶 _jpg.rf.xxx)。")
        else:
            print("❌ 有讀到圖但 YOLO 都沒偵測到牙：檢查 SEG_WEIGHTS 是不是 seg 權重、"
                  "USE_CLAHE / PAD_TB_RATIO 是否跟訓練一致。")
        return
    best, best_v = None, np.inf
    for mode in MODES:
        d = ok[f"相對差_{mode}"].abs()
        print(f"  {mode:12s} 中位數相對差 {d.median()*100:6.2f}%   ≤1% 的張數 {(d <= 0.01).sum()}/{len(d)}")
        if d.median() < best_v:
            best, best_v = mode, d.median()
    out["疑似挑牙不同"] = out.get(f"相對差_{best}", pd.Series(dtype=float)).abs() > OUTLIER_REL
    out.to_excel(resolve(OUT_XLSX), index=False)

    print(f"\n👉 建議 LENGTH_SCALE_MODE = \"{best}\"（目前 pipeline.py 設 \"{P.LENGTH_SCALE_MODE}\"）")
    if best_v > 0.01:
        print("⚠️ 最好的模式中位數相對差仍 > 1%：前處理參數可能還有不一致，"
              "依序檢查 USE_CLAHE → PAD_TB_RATIO → SEG_WEIGHTS（一次改一個）。")
    print(f"📄 逐張結果：{OUT_XLSX}")


if __name__ == "__main__":
    main()
