#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B2s_seg_inference.py
===============================================================
方案B右支的驗收腳本，跟左支的 B2 對稱：

    左支： B1  訓練 pose  →  B2  推論  →  關鍵點 CSV  →  B3
    右支： B1s 訓練 seg   →  B2s 推論  →  mask 統計 Excel（單獨驗收用）

*** 這支不是 B3 的上游 ***
B3 自己會載入 SEG_WEIGHTS、對影像重跑一次 seg 推論，不吃 B2s 的輸出。
所以 B2s 的定位只有一個：在把 seg 接進 B3 之前，先確認右支模型本身
沒問題（每張圖偵測到幾個 mask、面積合不合理、信心度夠不夠、有沒有
整張漏檢）。確認完就可以放著，改 B3 的幾何邏輯不需要重跑 B2s。

輸出：
  1. OUTPUT_XLSX — 逐張的 mask 統計，給人看
  2. VIS_DIR     — 疊上 mask 輪廓的可視化影像

座標空間：Ultralytics 回傳的 masks.xy 已經換算回「餵進去那張圖」的
實際像素座標，不是 imgsz 縮放後的座標。
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ============================================================
# 第1部分：設定
# ============================================================

SEG_WEIGHTS = "yolo11_seg_run/weights_ready.pt"
IMAGE_DIR = "test_segment/test/images"   # 👈 必須與 B2 的 IMAGE_DIR 完全相同

OUTPUT_XLSX = "B2s_seg推論統計_11.xlsx"
VIS_DIR = "B2s_seg視覺化"
SAVE_VISUALIZATION = True

IMG_SIZE = 640          # 👈 跟 seg 訓練時一致
SEG_CONF = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# 第2部分：工具
# ============================================================

def list_images(image_dir: Path):
    return sorted(p for p in image_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def polygon_area(pts):
    """鞋帶公式算多邊形面積。用來判斷有沒有碎 mask。"""
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def draw(img_path, polygons, confs, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    for poly, conf in zip(polygons, confs):
        pts = np.asarray(poly, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 200, 255), 2)
        top = pts[pts[:, 1].argmin()]
        cv2.putText(img, f"{conf:.2f}", tuple(top + np.int32([4, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ============================================================
# 第3部分：主流程
# ============================================================

def main():
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"找不到影像資料夾：{image_dir}")

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"{image_dir} 裡沒有任何影像")

    model = YOLO(SEG_WEIGHTS)
    print(f"載入 seg 模型：{SEG_WEIGHTS}")
    print(f"待推論：{len(images)} 張，imgsz={IMG_SIZE}，conf={SEG_CONF}\n")

    rows = []

    for img_path in images:
        res = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                            conf=SEG_CONF, save=False, verbose=False)[0]
        H, W = res.orig_shape

        polygons, confs, entries = [], [], []
        if res.masks is not None:
            raw_confs = res.boxes.conf.tolist() if res.boxes is not None else []
            for i, poly in enumerate(res.masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                pts = np.asarray(poly, dtype=np.float64)
                conf = float(raw_confs[i]) if i < len(raw_confs) else float("nan")
                polygons.append(pts)
                confs.append(conf)
                entries.append({
                    "conf": round(conf, 4),
                    "area": round(polygon_area(pts), 2),
                    "n_vertices": int(len(pts)),
                })

        # 面積大的排前面，統計欄位固定看「最大的那顆」
        entries.sort(key=lambda e: e["area"], reverse=True)

        rows.append({
            "圖片檔名": img_path.name,
            "影像尺寸": f"{W}x{H}",
            "mask數": len(entries),
            "最大mask面積px2": entries[0]["area"] if entries else 0,
            "最大mask信心度": entries[0]["conf"] if entries else None,
            "最大mask頂點數": entries[0]["n_vertices"] if entries else 0,
            "平均信心度": round(float(np.mean(confs)), 4) if confs else None,
            "狀態": "✅" if entries else "❌無輸出",
        })

        if SAVE_VISUALIZATION and polygons:
            draw(img_path, polygons, confs, Path(VIS_DIR) / f"{img_path.stem}_seg.jpg")

    # ---- 寫檔 ----
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="seg推論統計", index=False)

    # ---- 摘要 ----
    n_empty = int((df["mask數"] == 0).sum())
    sizes = df["影像尺寸"].unique()

    print(f"✅ 完成，共 {len(df)} 張")
    print(f"📄 {OUTPUT_XLSX}")
    if SAVE_VISUALIZATION:
        print(f"🖼️  {VIS_DIR}/")
    print(f"\n每張平均 mask 數：{df['mask數'].mean():.2f}")
    if n_empty:
        print(f"❌ 完全沒偵測到 mask：{n_empty} 張（這幾張 B3 會直接標記人工複查）")
    if len(sizes) > 1:
        print(f"⚠️ 影像尺寸不一致：{list(sizes)}")
        print(f"   B2 與 B3 假設所有圖同一尺寸，尺寸混雜代表匯出設定有問題，請先確認。")


if __name__ == "__main__":
    main()
