#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B2_seg_inference_v4.py
===============================================================
seg 模型的驗收腳本（不是 B3 的上游；B3 自己重跑推論，只「讀」這支輸出的
Excel 拿逐張 mAP 印在 B3 的圖上）。

輸出（每個 seed 一份）：
  1. OUTPUT_TMPL — 逐張的 mask 統計 + 逐張 Mask mAP50-95
  2. VIS_TMPL    — 疊上 mask 輪廓的可視化影像，左上角印該張的 mAP50-95
另有 SUMMARY_XLSX：跨 seed 的 mean ± SD，以及逐張 mAP 的跨 seed 寬表。

*** v4 變更：多 seed ***
逐一載入 B1_train_yolo11-seg_v2.py 各 seed 的權重，推論邏輯與 v3 完全相同。
推論本身是確定性的，不需要 seed。找不到某個 seed 的權重會跳過並警告。
⚠️ B3 的 B2_XLSX 要指到對應 seed 的檔案。

*** v3 變更：剔除 mask 碎塊 ***
masks.xy 會把同一個 mask 的所有碎塊串成一條多邊形（碎塊間有細連線，圖上
看起來像牙齒拖著一條線連到遠處的小塊）。REMOVE_FRAGMENTS=True 時，畫圖與
統計欄位只用每個 mask 的最大連通區。
⚠️ mAP 一律用原始 mask 算（評的是模型本身，不受後處理影響）。

*** 逐張 mAP50-95 的算法 ***
- GT：LABEL_DIR 下同檔名的 YOLO-seg 多邊形 → 原圖尺寸的二值 mask
- 預測：conf ≥ AP_CONF 的所有 mask（跟 Ultralytics val 一樣用低門檻）
- IoU 閾值 0.50:0.05:0.95，依信心度貪婪配對 → 101 點內插 AP，10 個取平均。
  mask 在原圖解析度算，數值不會跟 model.val() 完全一樣。
- test 集只標目標牙：其他牙的預測只要信心度排在目標牙前面就會拉低 AP。
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ============================================================
# 第1部分：設定
# ============================================================

TRAIN_SEEDS = [0, 1, 2, 3, 4]                                   # 👈 與 B1 v2 一致
WEIGHTS_TMPL = "yolo11n_seg_run_padded_seed{s}/weights_ready.pt"
IMAGE_DIR = "A5_test_padded/images"   # 👈 必須與 B3 的 IMAGE_DIR 完全相同
LABEL_DIR = "A5_test_padded/labels"   # 👈 GT 多邊形標註（算 mAP 用）

OUTPUT_TMPL = "B2_seg推論統計_11n_padded_seed{s}.xlsx"   # 👈 B3 的 B2_XLSX 要指到對應 seed
VIS_TMPL = "B2_seg視覺化_11n_padded_seed{s}"
SUMMARY_XLSX = "B2_seed彙整_11n_padded.xlsx"
SAVE_VISUALIZATION = True

IMG_SIZE = 640          # 👈 跟 seg 訓練時一致
SEG_CONF = 0.15         # 統計欄位與畫圖用
AP_CONF = 0.001         # 算 mAP 用，不要改（Ultralytics val 預設）

REMOVE_FRAGMENTS = True  # 畫圖/統計只留每個 mask 的最大連通區（不影響 mAP）

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ============================================================
# 第2部分：工具
# ============================================================

def list_images(image_dir: Path):
    return sorted(p for p in image_dir.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def polygon_area(pts):
    """鞋帶公式算多邊形面積。"""
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


def poly_to_mask(pts, H, W):
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def largest_part(polygon, shape):
    """masks.xy 會把同一 mask 的碎塊串成一條多邊形(中間有細連線)。
    填實 → 3x3 開運算切斷細連線 → 留最大連通區 → 取回外輪廓。
    回傳 (多邊形, 丟掉的碎塊數)。與 B3 的同名函式完全相同。"""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return np.asarray(polygon, dtype=np.float64), 0
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lab == big).astype(np.uint8),
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    return c, n - 2


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
# 第3部分：單一 seed 推論（邏輯與 v3 的 main 相同）
# ============================================================

def run_one(seed, images, label_dir):
    """回傳 (逐張 df, 摘要 dict)；找不到權重回 (None, None)。"""
    seg_weights = WEIGHTS_TMPL.format(s=seed)
    output_xlsx = OUTPUT_TMPL.format(s=seed)
    vis_dir = VIS_TMPL.format(s=seed)

    if not Path(seg_weights).exists():
        print(f"⚠️ seed {seed}：找不到 {seg_weights}，跳過")
        return None, None

    model = YOLO(seg_weights)
    print(f"\n################ seed {seed} ################")
    print(f"載入 seg 模型：{seg_weights}")
    print(f"待推論：{len(images)} 張，imgsz={IMG_SIZE}，"
          f"統計/畫圖 conf={SEG_CONF}，mAP conf={AP_CONF}，"
          f"剔除碎塊={REMOVE_FRAGMENTS}\n")

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
                all_masks.append(poly_to_mask(pts, H, W))      # mAP：原始 mask
                all_confs.append(conf)
                if conf < SEG_CONF:
                    continue
                n_drop = 0
                if REMOVE_FRAGMENTS:
                    pts, n_drop = largest_part(pts, (H, W))
                polygons.append(pts)
                confs.append(conf)
                entries.append({
                    "conf": round(conf, 4),
                    "area": round(polygon_area(pts), 2),
                    "n_vertices": int(len(pts)),
                    "n_drop": n_drop,
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
            "最大mask剔除碎塊數": entries[0]["n_drop"] if entries else 0,
            "Mask_mAP50-95": None if np.isnan(map_val) else round(map_val, 4),
            "GT數": None if gt_masks is None else len(gt_masks),
            "狀態": "✅" if entries else "❌無輸出",
        })

        if SAVE_VISUALIZATION:
            draw(img_path, polygons, confs, map_val,
                 Path(vis_dir) / f"{img_path.stem}_seg.jpg")

    # ---- 寫檔 ----
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="seg推論統計", index=False)

    # ---- 摘要 ----
    n_empty = int((df["mask數"] == 0).sum())
    n_nogt = int(df["GT數"].isna().sum())
    n_frag = int((df["最大mask剔除碎塊數"] > 0).sum())
    sizes = df["影像尺寸"].unique()

    print(f"✅ 完成，共 {len(df)} 張")
    print(f"📄 {output_xlsx}")
    if SAVE_VISUALIZATION:
        print(f"🖼️  {vis_dir}/")
    print(f"\n每張平均 mask 數：{df['mask數'].mean():.2f}")
    print(f"逐張 Mask mAP50-95 平均：{df['Mask_mAP50-95'].mean():.4f}"
          f"（中位數 {df['Mask_mAP50-95'].median():.4f}）")
    if REMOVE_FRAGMENTS:
        print(f"✂️  最大 mask 帶碎塊（已剔除）：{n_frag} 張")
    if n_nogt:
        print(f"⚠️ {n_nogt} 張沒有標註檔，mAP 記 N/A")
    if n_empty:
        print(f"❌ 完全沒偵測到 mask：{n_empty} 張（這幾張 B3 會直接標記人工複查）")
    if len(sizes) > 1:
        print(f"⚠️ 影像尺寸不一致：{list(sizes)}")
        print(f"   B2 與 B3 假設所有圖同一尺寸，尺寸混雜代表匯出設定有問題，請先確認。")

    summary = {
        "train_seed": seed,
        "逐張mAP平均": round(df["Mask_mAP50-95"].mean(), 4),
        "逐張mAP中位數": round(df["Mask_mAP50-95"].median(), 4),
        "平均mask數": round(df["mask數"].mean(), 2),
        "無輸出張數": n_empty,
        "帶碎塊張數": n_frag,
    }
    return df, summary


# ============================================================
# 第4部分：主流程（逐 seed 推論 + 跨 seed 彙整）
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

    summaries, per_image = [], {}
    for s in TRAIN_SEEDS:
        df, summ = run_one(s, images, label_dir)
        if df is None:
            continue
        summaries.append(summ)
        per_image[f"seed{s}"] = df.set_index("圖片檔名")["Mask_mAP50-95"]

    if not summaries:
        raise FileNotFoundError("所有 seed 都找不到權重，請先跑 B1_train_yolo11-seg_v2.py")

    sdf = pd.DataFrame(summaries)
    num_cols = [c for c in sdf.columns if c != "train_seed"]
    stat = pd.DataFrame({"mean": sdf[num_cols].mean(),
                         "SD": sdf[num_cols].std(ddof=1),
                         "n_seed": sdf[num_cols].count()})

    # 逐張 mAP 跨 seed：看哪幾張對訓練隨機性特別敏感
    wide = pd.DataFrame(per_image)
    seed_cols = list(wide.columns)
    wide["mean"] = wide[seed_cols].mean(axis=1).round(4)
    wide["SD"] = wide[seed_cols].std(axis=1, ddof=1).round(4)
    wide = wide.sort_values("SD", ascending=False)

    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as w:
        sdf.to_excel(w, sheet_name="各seed", index=False)
        stat.to_excel(w, sheet_name="mean±SD")
        wide.to_excel(w, sheet_name="逐張mAP_跨seed")

    m = sdf["逐張mAP平均"]
    print(f"\n===== 跨 {len(sdf)} 個 seed(test 集) =====")
    print(sdf.to_string(index=False))
    if len(sdf) > 1:
        print(f"\nMask mAP50-95（逐張平均）：{m.mean():.4f} ± {m.std(ddof=1):.4f} (mean ± SD)")
        print(f"跨 seed 最不穩定的 3 張：{', '.join(wide.index[:3])}")
    else:
        print(f"\n⚠️ 只有 1 個 seed 有權重，無法算 SD")
    print(f"📄 {SUMMARY_XLSX}")
    print(f"   ⚠️ 逐張平均 ≠ model.val() 的整體 mAP；最終指標仍是 B3→C1 的長度誤差。")


if __name__ == "__main__":
    main()