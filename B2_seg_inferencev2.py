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
  1. OUTPUT_XLSX — 逐張的 mask 統計 + 逐張 Mask mAP50-95，給人看
  2. VIS_DIR     — 疊上 mask 輪廓的可視化影像，左上角印該張的 mAP50-95

*** 逐張 mAP50-95 的算法 ***
- GT：LABEL_DIR 下同檔名的 YOLO-seg 多邊形 → 原圖尺寸的二值 mask
- 預測：conf ≥ AP_CONF 的所有 mask（跟 Ultralytics val 一樣用低門檻，
  不是 SEG_CONF；否則 PR 曲線被截尾，AP 偏低）
- IoU 閾值 0.50:0.05:0.95，每個閾值依信心度貪婪配對 → 101 點內插 AP，
  10 個取平均。與 Ultralytics 的 compute_ap 同一套公式，但 mask 是
  在原圖解析度算的，數值不會跟 model.val() 完全一樣。
- test 集只標目標牙：其他牙的預測只要信心度排在目標牙前面就會拉低 AP。

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
IMAGE_DIR = "test_single_seg-67/test/images"   # 👈 必須與 B2 的 IMAGE_DIR 完全相同
LABEL_DIR = "test_single_seg-67/test/labels"   # 👈 GT 多邊形標註（算 mAP 用）

OUTPUT_XLSX = "B2_seg推論統計.xlsx"
VIS_DIR = "B2_seg視覺化"
SAVE_VISUALIZATION = True

IMG_SIZE = 640          # 👈 跟 seg 訓練時一致
SEG_CONF = 0.15         # 統計欄位與畫圖用
AP_CONF = 0.001         # 算 mAP 用，不要改（Ultralytics val 預設）

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
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


def poly_to_mask(pts, H, W):
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def load_gt_masks(label_path: Path, H, W):
    """讀 YOLO-seg 標註：class x1 y1 ... xn yn（正規化座標）。"""
    masks = []
    if not label_path.exists():
        return None                      # 沒標註檔 → mAP 記 NaN
    for line in label_path.read_text(encoding="utf-8").splitlines():
        v = line.split()[1:]
        if len(v) < 6 or len(v) % 2:
            continue
        pts = np.asarray(v, dtype=np.float64).reshape(-1, 2) * [W, H]
        masks.append(poly_to_mask(pts, H, W))
    return masks


def mask_iou(pred_masks, gt_masks):
    iou = np.zeros((len(pred_masks), len(gt_masks)))
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            union = np.logical_or(p, g).sum()
            iou[i, j] = np.logical_and(p, g).sum() / union if union else 0.0
    return iou


def compute_ap(recall, precision):
    """Ultralytics / COCO 101 點內插。"""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(np.interp(x, mrec, mpre), x))


def image_map50_95(pred_masks, pred_confs, gt_masks):
    """單張圖的 Mask mAP50-95。GT 為 None 回 NaN；有 GT 沒預測回 0。"""
    if gt_masks is None or len(gt_masks) == 0:
        return float("nan")
    if len(pred_masks) == 0:
        return 0.0
    order = np.argsort(-np.asarray(pred_confs))
    iou = mask_iou([pred_masks[k] for k in order], gt_masks)
    n_gt = len(gt_masks)

    aps = []
    for t in IOU_THRESHOLDS:
        matched = np.zeros(n_gt, dtype=bool)
        tp = np.zeros(len(order))
        for i in range(len(order)):
            cand = np.where((iou[i] >= t) & ~matched)[0]
            if cand.size:
                j = cand[np.argmax(iou[i, cand])]
                matched[j] = True
                tp[i] = 1
        tpc = np.cumsum(tp)
        fpc = np.cumsum(1 - tp)
        aps.append(compute_ap(tpc / n_gt, tpc / (tpc + fpc)))
    return float(np.mean(aps))


def draw(img_path, polygons, confs, map_val, out_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    for poly, conf in zip(polygons, confs):
        pts = np.asarray(poly, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 200, 255), 2)
        top = pts[pts[:, 1].argmin()]
        cv2.putText(img, f"{conf:.2f}", tuple(top + np.int32([4, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # ---- 左上角印 mAP50-95 ----
    txt = "mAP50-95: N/A" if np.isnan(map_val) else f"mAP50-95: {map_val:.3f}"
    scale = max(0.6, img.shape[1] / 900)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(8 * scale)
    cv2.rectangle(img, (0, 0), (tw + 2 * pad, th + base + 2 * pad), (0, 0, 0), -1)
    cv2.putText(img, txt, (pad, pad + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 255, 0), thick, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ============================================================
# 第3部分：主流程
# ============================================================

def main():
    image_dir, label_dir = Path(IMAGE_DIR), Path(LABEL_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"找不到影像資料夾：{image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"找不到標註資料夾：{label_dir}（算 mAP 需要 GT）")

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"{image_dir} 裡沒有任何影像")

    model = YOLO(SEG_WEIGHTS)
    print(f"載入 seg 模型：{SEG_WEIGHTS}")
    print(f"待推論：{len(images)} 張，imgsz={IMG_SIZE}，"
          f"統計/畫圖 conf={SEG_CONF}，mAP conf={AP_CONF}\n")

    rows = []

    for img_path in images:
        # 用 AP_CONF 推論一次：全部拿去算 mAP，≥SEG_CONF 的才進統計與畫圖
        res = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                            conf=AP_CONF, save=False, verbose=False)[0]
        H, W = res.orig_shape

        all_masks, all_confs = [], []
        polygons, confs, entries = [], [], []
        if res.masks is not None:
            raw_confs = res.boxes.conf.tolist() if res.boxes is not None else []
            for i, poly in enumerate(res.masks.xy):
                if poly is None or len(poly) < 3:
                    continue
                pts = np.asarray(poly, dtype=np.float64)
                conf = float(raw_confs[i]) if i < len(raw_confs) else float("nan")
                all_masks.append(poly_to_mask(pts, H, W))
                all_confs.append(conf)
                if conf < SEG_CONF:
                    continue
                polygons.append(pts)
                confs.append(conf)
                entries.append({
                    "conf": round(conf, 4),
                    "area": round(polygon_area(pts), 2),
                    "n_vertices": int(len(pts)),
                })

        gt_masks = load_gt_masks(label_dir / f"{img_path.stem}.txt", H, W)
        map_val = image_map50_95(all_masks, all_confs, gt_masks)

        # 面積大的排前面，統計欄位固定看「最大的那顆」
        entries.sort(key=lambda e: e["area"], reverse=True)

        rows.append({
            "圖片檔名": img_path.name,
            "影像尺寸": f"{W}x{H}",
            "mask數": len(entries),
            "最大mask面積px2": entries[0]["area"] if entries else 0,
            "最大mask信心度": entries[0]["conf"] if entries else None,
            "最大mask頂點數": entries[0]["n_vertices"] if entries else 0,
            "Mask_mAP50-95": None if np.isnan(map_val) else round(map_val, 4),
            "GT數": None if gt_masks is None else len(gt_masks),
            "狀態": "✅" if entries else "❌無輸出",
        })

        if SAVE_VISUALIZATION:
            draw(img_path, polygons, confs, map_val,
                 Path(VIS_DIR) / f"{img_path.stem}_seg.jpg")

    # ---- 寫檔 ----
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="seg推論統計", index=False)

    # ---- 摘要 ----
    n_empty = int((df["mask數"] == 0).sum())
    n_nogt = int(df["GT數"].isna().sum())
    sizes = df["影像尺寸"].unique()

    print(f"✅ 完成，共 {len(df)} 張")
    print(f"📄 {OUTPUT_XLSX}")
    if SAVE_VISUALIZATION:
        print(f"🖼️  {VIS_DIR}/")
    print(f"\n每張平均 mask 數：{df['mask數'].mean():.2f}")
    print(f"逐張 Mask mAP50-95 平均：{df['Mask_mAP50-95'].mean():.4f}"
          f"（中位數 {df['Mask_mAP50-95'].median():.4f}）")
    print(f"   ⚠️ 這是逐張平均，不等於 model.val() 的整體 mAP（整體是全部預測一起排序）")
    if n_nogt:
        print(f"⚠️ {n_nogt} 張沒有標註檔，mAP 記 N/A")
    if n_empty:
        print(f"❌ 完全沒偵測到 mask：{n_empty} 張（這幾張 B3 會直接標記人工複查）")
    if len(sizes) > 1:
        print(f"⚠️ 影像尺寸不一致：{list(sizes)}")
        print(f"   B2 與 B3 假設所有圖同一尺寸，尺寸混雜代表匯出設定有問題，請先確認。")


if __name__ == "__main__":
    main()
