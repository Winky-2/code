#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3_mask_assist_and_compare.py
===============================================================
方案B：分割(segmentation)當「輔助 + 防呆」，pose 主線完全不動。

本腳本不重跑 pose，而是讀 B2v2 已經輸出的關鍵點 CSV，
再跑一次 seg 模型，做三件事：

  1) 防呆 QC   : 預測的 A/B 點有沒有落在牙齒 mask 內？離邊界多遠？
  2) 長軸投影  : 把 A/B 投影到 mask 的主軸上再算長度，
                 消掉「垂直於牙軸」方向的定位噪音。
  3) 平行對照  : 同時輸出三種 AB 長度，讓你用數據決定
                 要不要升級成方案A(純幾何)。

     (a) 原始預測      = B2 現在在用的
     (b) 長軸投影      = 預測點投影到 mask 主軸
     (c) mask 幾何端點 = 完全不看關鍵點，純由 mask 推

*** 一張圖有多顆牙時的挑 mask 問題 ***
你們的 seg 訓練集一張圖標了好幾顆牙，所以推論時會吐出多個 mask，
必須挑出「目標牙」那一顆。本腳本提供兩種方式：

  points : 用 pose 預測的 A/B 點去挑(哪顆 mask 把兩點包得最深)。
           不依賴 GT，臨床推論時可用。但 A 點(切端)常落在鄰牙
           交界處，牙列擁擠時有機會挑到隔壁牙。

  gt_iou : 用舊標註的 GT bbox 去挑(mask 外接框與 GT 框 IoU 最高)。
           跟 B2v2 的 select_target_instance() 同一套邏輯，準確但
           依賴 GT，無法用在沒有標註的新病例。

  auto   : 有 GT 就用 gt_iou，沒有就退回 points。預設值。

現階段要驗證的是「分割對長度量測有沒有幫助」，不是「做出能上線
的系統」，所以先用 gt_iou 把牙挑對、把三法比較的數字算乾淨比較
重要。認牙問題等確認方案有效再處理。

不論用哪種方式，都會同時算出另一種方式的結果，並在「兩法選取一致」
欄位回報是否挑到同一顆——這個欄位就是在量「points 模式將來上線時
會錯多少」，別忽略它。

前置需求：
    - 已跑完 B2v2，產生 RAW_KEYPOINT_CSV
    - 已跑完 B1s(或等效的 seg 訓練)，有 best.pt
    - seg 與 pose 使用同一批圖、同樣的 Resize 設定
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ============================================================
# 第1部分：設定
# ============================================================

# --- 輸入 ---
SEG_WEIGHTS ="yolo11_seg_run/weights_ready.pt"
RAW_KEYPOINT_CSV = "yolo關鍵點原始座標_11.csv"   # 👈 B2v2 的輸出
IMAGE_DIR = "test_enhance/test/images"        # 👈 必須與 B2 的 IMAGE_DIR 完全相同

# 可選：B2 的主 Excel，用來併入 GT 長度算誤差。留空則只輸出幾何量(看不到三法比較)。
B2_XLSX = "yolo像素預測_11.xlsx"
B2_SHEET = 0

# 可選：舊標註的 pose GT labels，用於 gt_iou 挑 mask。留空則強制走 points。
GT_LABEL_DIR = "test_enhance/test/labels"
    
# --- 輸出 ---
OUTPUT_XLSX = "B3_mask輔助對照_11.xlsx"
VIS_DIR = "B3_mask視覺化"
SAVE_VISUALIZATION = True

# --- 推論參數 ---
IMG_SIZE = 640                     # 👈 跟 seg 訓練時一致
SEG_CONF = 0.15                    # Precision 高，不需要壓太低；壓低反而增加碎 mask 干擾挑選

# --- 挑 mask 方式 ---
MASK_SELECT_MODE = "auto"          # "auto" | "gt_iou" | "points"
AMBIGUOUS_IOU_GAP = 0.15           # 最佳與次佳 mask 的 IoU 差距小於此值 → 標記選取不明確

# --- QC 門檻 ---
OUTSIDE_MARGIN_PX = 8.0            # 點落在 mask 外超過此距離 → 建議人工複查
MIN_AXIS_RATIO = 0.75              # 長軸解釋比低於此值 → 牙形不夠細長，投影不可靠
ENDPOINT_ERROR_PX = 12.0           # 幾何端點離 GT 超過此距離 → 位置可疑(即使長度剛好)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --- 視覺化圖例 ---
# cv2.putText 只支援 Hershey 字型，畫不出中文(會變成問號)，所以圖上標籤
# 只能用英文，中文對照見下方 LEGEND_ZH 與終端機輸出。
SHOW_LEGEND = True
LEGEND_ITEMS = [                      # (圖上標籤, BGR)
    ("chosen mask (seg)",   (0, 200, 255)),
    ("other masks",         (130, 130, 130)),
    ("geo ends = METHOD 3", (255, 160, 0)),
    ("axis proj = METHOD 2", (255, 255, 0)),
    ("pose A-B = METHOD 1", (0, 255, 0)),
    ("GT annotation",       (255, 0, 255)),
]
LEGEND_ZH = [
    ("橘黃輪廓", "選中的 mask(seg 預測)"),
    ("灰色輪廓", "其他候選 mask，未被選中"),
    ("藍線 geo1-geo2", "方法3 mask幾何 ← 目前打算餵給 C1 的"),
    ("青色虛線", "方法2 長軸投影(預測點投影到 mask 主軸)"),
    ("綠線 A-B / 紅點B", "方法1 原始預測(B2 現在在用的)"),
    ("洋紅空心圈", "GT 人工標註的 A/B 真值"),
]


# ============================================================
# 第2部分：幾何工具
# ============================================================

def euclidean(p1, p2):
    return math.dist(p1, p2)


def yolo_to_corners(cx, cy, w, h):
    """YOLO 中心點格式 → (x0, y0, x1, y1) 角點格式。"""
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def compute_iou(boxA, boxB):
    ax0, ay0, ax1, ay1 = boxA
    bx0, by0, bx1, by1 = boxB
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def polygon_bbox(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()),
            float(pts[:, 0].max()), float(pts[:, 1].max()))


def mask_axis(polygon):
    """對 mask 多邊形做 PCA，回傳 (質心, 長軸單位向量, 長軸解釋比)。

    長軸解釋比 = 第一主成分奇異值 / 兩個奇異值之和。
    牙齒細長 → 接近 1；接近圓形 → 接近 0.5，此時投影方向不可信。
    彎曲牙根也會壓低這個值，是「投影法在這顆牙上能不能用」的指標。
    """
    pts = np.asarray(polygon, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    ratio = float(s[0] / max(s.sum(), 1e-9))
    return centroid, axis, ratio


def axis_extremes(polygon, centroid, axis):
    """mask 沿長軸投影的兩個極值點(方案A會用的「幾何端點」)。"""
    pts = np.asarray(polygon, dtype=np.float64)
    t = (pts - centroid) @ axis
    return centroid + axis * t.min(), centroid + axis * t.max()


def project_points(pA, pB, centroid, axis):
    """A/B 投影到長軸上的落點(方法2 實際量的是這兩點之間的距離)。"""
    a = np.asarray(pA, dtype=np.float64)
    b = np.asarray(pB, dtype=np.float64)
    tA = float((a - centroid) @ axis)
    tB = float((b - centroid) @ axis)
    return centroid + axis * tA, centroid + axis * tB


def project_len(pA, pB, centroid, axis):
    """A/B 投影到長軸後的距離：只保留沿牙軸方向的分量。"""
    qA, qB = project_points(pA, pB, centroid, axis)
    return float(np.linalg.norm(qA - qB))


def signed_dist_to_polygon(polygon, pt):
    """點到多邊形邊界的帶號距離：正=在內部、負=在外部、0=在邊上。"""
    poly32 = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return float(cv2.pointPolygonTest(poly32, (float(pt[0]), float(pt[1])), True))


# ============================================================
# 第3部分：GT 讀取與 mask 挑選
# ============================================================

def load_gt_target(label_path: Path, width: int, height: int):
    """從 pose 格式的 GT label 撈出「目標牙」的 bbox 與 A/B 點(像素空間)。

    目標牙 = A、B 兩個 keypoint 的 visibility 都是 2 的那一行。
    回傳 (bbox, GT_A, GT_B)，找不到則三個都是 None。

    bbox 用於 gt_iou 挑 mask；A/B 點用於視覺化與「端點位置誤差」欄位。
    注意：這裡的座標跟 pose 預測點一樣位於 letterbox(640) 空間。
    """
    if not label_path.exists():
        return None, None, None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                v1, v2 = float(parts[7]), float(parts[10])
            except ValueError:
                continue
            if v1 == 2 and v2 == 2:
                cx, cy, w, h = (float(parts[i]) for i in range(1, 5))
                bbox = yolo_to_corners(cx * width, cy * height,
                                       w * width, h * height)
                gtA = (float(parts[5]) * width, float(parts[6]) * height)
                gtB = (float(parts[8]) * width, float(parts[9]) * height)
                return bbox, gtA, gtB
    return None, None, None


def pick_by_points(polygons, pA, pB):
    """用 pose 預測的 A/B 點挑 mask：兩點中「較差的那個」內部深度最大者勝。

    兩點都在內部 → 分數為正；都在外部 → 為負，仍會選出最接近的一顆，
    由後續 QC 欄位判定可不可信。
    """
    best_idx, best_score = None, -1e18
    for i, poly in enumerate(polygons):
        score = min(signed_dist_to_polygon(poly, pA),
                    signed_dist_to_polygon(poly, pB))
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx, best_score


def pick_by_gt_iou(polygons, gt_bbox):
    """用 GT bbox 挑 mask：mask 外接框與 GT 框 IoU 最高者勝。
    回傳 (最佳索引, 最佳IoU, 次佳IoU)。"""
    ious = [compute_iou(polygon_bbox(p), gt_bbox) for p in polygons]
    order = sorted(range(len(ious)), key=lambda i: ious[i], reverse=True)
    best = order[0]
    second = ious[order[1]] if len(order) > 1 else 0.0
    return best, ious[best], second


def find_image(stem, image_dir: Path):
    for ext in IMAGE_EXTENSIONS:
        p = image_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in image_dir.iterdir():
        if p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS:
            return p
    return None


# ============================================================
# 第4部分：視覺化
# ============================================================

def dashed_line(img, p1, p2, color, thickness=1, dash=8):
    """畫虛線：用來區分方法2，避免跟旁邊的實線混淆。"""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    total = float(np.linalg.norm(p2 - p1))
    if total < 1e-6:
        return
    unit = (p2 - p1) / total
    t = 0.0
    while t < total:
        a = p1 + unit * t
        b = p1 + unit * min(t + dash, total)
        cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), color, thickness)
        t += dash * 2


def draw_legend(img, items, line_h=17, pad=8, box_w=190):
    """左上角畫半透明圖例。標籤只能用英文(Hershey 字型無中文)。"""
    h = line_h * len(items) + pad * 2
    x0, y0 = pad, pad
    x1, y1 = min(x0 + box_w, img.shape[1] - 1), min(y0 + h, img.shape[0] - 1)
    roi = img[y0:y1, x0:x1]
    if roi.size:
        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0)
    for i, (label, color) in enumerate(items):
        y = y0 + pad + line_h * i + 10
        cv2.line(img, (x0 + 6, y - 4), (x0 + 24, y - 4), color, 3)
        cv2.putText(img, label, (x0 + 30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, color, 1, cv2.LINE_AA)


def draw(img_path, chosen, others, pA, pB, end1, end2, out_path,
         gt_pts=None, proj_pts=None):
    """三種方法在圖上的對應(OpenCV 是 BGR，數值看起來會跟顏色名對不上)

        橘黃輪廓 (0,200,255)   = 選中的 mask
        灰色輪廓 (130,130,130) = 其他候選 mask
        藍線 geo1-geo2 (255,160,0)  = 方法3 mask幾何
        青色虛線 (255,255,0)        = 方法2 長軸投影
        綠線 A / 紅點 B             = 方法1 原始預測
        洋紅空心圈 (255,0,255)      = GT 人工標註真值

    翻圖時看兩件事：
      1) 橘黃輪廓有沒有套在正確那顆牙上(認牙對不對)
      2) 藍點下端(根尖側)離洋紅 GT_B 多遠(位置準不準)
         ——長度對但位置偏的情況只有這裡看得出來，Excel 的長度誤差看不到。
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    for poly in others:
        cv2.polylines(img, [np.asarray(poly, dtype=np.int32)], True, (130, 130, 130), 1)
    if chosen is not None:
        cv2.polylines(img, [np.asarray(chosen, dtype=np.int32)], True, (0, 200, 255), 2)
        cv2.line(img, tuple(np.int32(end1)), tuple(np.int32(end2)), (255, 160, 0), 2)
        for p, tag in ((end1, "geo1"), (end2, "geo2")):
            cv2.circle(img, tuple(np.int32(p)), 5, (255, 160, 0), -1)
            cv2.putText(img, tag, tuple(np.int32(p) + np.int32([6, -6])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
    if proj_pts is not None:
        dashed_line(img, proj_pts[0], proj_pts[1], (255, 255, 0), 1)
        for p in proj_pts:
            cv2.circle(img, tuple(np.int32(p)), 3, (255, 255, 0), 1)
    cv2.line(img, tuple(np.int32(pA)), tuple(np.int32(pB)), (0, 255, 0), 1)
    for p, tag, color in ((pA, "A", (0, 255, 0)), (pB, "B", (0, 0, 255))):
        cv2.circle(img, tuple(np.int32(p)), 5, color, -1)
        cv2.putText(img, tag, tuple(np.int32(p) + np.int32([6, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    # GT 最後畫，確保不會被其他圖層蓋掉；空心圈避免遮住底下的預測點
    if gt_pts is not None:
        gtA, gtB = gt_pts
        cv2.line(img, tuple(np.int32(gtA)), tuple(np.int32(gtB)), (255, 0, 255), 1)
        for p, tag in ((gtA, "GT_A"), (gtB, "GT_B")):
            cv2.circle(img, tuple(np.int32(p)), 7, (255, 0, 255), 2)
            cv2.putText(img, tag, tuple(np.int32(p) + np.int32([9, 4])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    if SHOW_LEGEND:
        draw_legend(img, LEGEND_ITEMS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ============================================================
# 第5部分：主流程
# ============================================================

def main():
    kp_csv = Path(RAW_KEYPOINT_CSV)
    if not kp_csv.exists():
        raise FileNotFoundError(f"找不到 B2 的關鍵點 CSV：{kp_csv}(請先跑 B2v2)")

    df_kp = pd.read_csv(kp_csv)
    image_dir = Path(IMAGE_DIR)
    gt_dir = Path(GT_LABEL_DIR) if GT_LABEL_DIR else None
    model = YOLO(SEG_WEIGHTS)

    note = "(有 GT 走 gt_iou，無則退回 points)" if MASK_SELECT_MODE == "auto" else ""
    print(f"挑 mask 模式：{MASK_SELECT_MODE}{note}")

    rows = []
    for _, r in df_kp.iterrows():
        fname = str(r["圖片檔名"])
        stem = Path(fname).stem
        pA = (float(r["A_x_原圖"]), float(r["A_y_原圖"]))
        pB = (float(r["B_x_原圖"]), float(r["B_y_原圖"]))
        scale = float(r.get("縮放比scale", 1.0))
        raw_len = euclidean(pA, pB)

        rec = {
            "圖片檔名": fname,
            "縮放比scale": scale,
            "AB長度_原始預測_letterbox px": round(raw_len, 4),
            "AB長度_原始預測_原圖px": round(raw_len * scale, 4),
        }

        img_path = find_image(stem, image_dir)
        if img_path is None:
            rec.update({"mask狀態": "❌找不到圖片", "建議人工複查": "⚠️是"})
            rows.append(rec)
            continue

        res = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                            conf=SEG_CONF, save=False, verbose=False)[0]
        polygons = [p for p in (res.masks.xy if res.masks is not None else [])
                    if p is not None and len(p) >= 3]

        if not polygons:
            rec.update({"mask狀態": "❌seg無輸出", "偵測到mask數": 0, "建議人工複查": "⚠️是"})
            rows.append(rec)
            continue

        # ---- 挑出目標牙的 mask ----
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        gt_bbox, gtA, gtB = (load_gt_target(gt_dir / f"{stem}.txt", W, H)
                             if gt_dir else (None, None, None))

        idx_pts, _ = pick_by_points(polygons, pA, pB)
        idx_iou, best_iou, second_iou = None, None, None
        if gt_bbox is not None:
            idx_iou, best_iou, second_iou = pick_by_gt_iou(polygons, gt_bbox)

        if MASK_SELECT_MODE == "points" or idx_iou is None:
            chosen_idx, used_mode = idx_pts, "points"
        else:
            chosen_idx, used_mode = idx_iou, "gt_iou"

        poly = polygons[chosen_idx]
        centroid, axis, ratio = mask_axis(poly)
        end1, end2 = axis_extremes(poly, centroid, axis)
        # 讓幾何端點的 A/B 歸屬跟預測點一致，避免對照表出現假性顛倒
        if euclidean(pA, end2) < euclidean(pA, end1):
            end1, end2 = end2, end1

        dA = signed_dist_to_polygon(poly, pA)
        dB = signed_dist_to_polygon(poly, pB)
        proj_len = project_len(pA, pB, centroid, axis)
        proj_pts = project_points(pA, pB, centroid, axis)
        geo_len = euclidean(end1, end2)

        flags = []
        if dA < -OUTSIDE_MARGIN_PX:
            flags.append("A點在mask外")
        if dB < -OUTSIDE_MARGIN_PX:
            flags.append("B點在mask外")
        if ratio < MIN_AXIS_RATIO:
            flags.append("牙形不夠細長")
        if idx_iou is not None and idx_pts != idx_iou:
            flags.append("兩法選到不同mask")
        if (best_iou is not None and second_iou is not None
                and (best_iou - second_iou) < AMBIGUOUS_IOU_GAP):
            flags.append("選取不明確")

        rec.update({
            "mask狀態": "✅",
            "偵測到mask數": len(polygons),
            "選取方式": used_mode,
            "兩法選取一致": ("—" if idx_iou is None else ("是" if idx_pts == idx_iou else "⚠️否")),
            "mask_IoU_vs_GT框": round(best_iou, 3) if best_iou is not None else None,
            "次佳mask_IoU": round(second_iou, 3) if second_iou is not None else None,
            "A點到mask邊界px": round(dA, 2),
            "B點到mask邊界px": round(dB, 2),
            "長軸解釋比": round(ratio, 3),
            "AB長度_長軸投影_letterbox px": round(proj_len, 4),
            "AB長度_mask幾何_letterbox px": round(geo_len, 4),
            "AB長度_長軸投影_原圖px": round(proj_len * scale, 4),
            "AB長度_mask幾何_原圖px": round(geo_len * scale, 4),
        })

        # ---- 端點位置誤差：長度對不代表位置對 ----
        # mask 幾何端點是 PCA 長軸極值，兩端各偏一點但方向相反時，
        # 長度誤差會互相抵銷、看起來很準，實際上根尖位置是錯的。
        # 臨床上真正要準的是根尖(B側)，所以這兩欄要跟長度誤差分開看。
        if gtA is not None:
            e1 = euclidean(end1, gtA)
            e2 = euclidean(end2, gtB)
            rec["幾何端點誤差_A側px"] = round(e1, 2)
            rec["幾何端點誤差_B側px"] = round(e2, 2)
            rec["預測點誤差_A側px"] = round(euclidean(pA, gtA), 2)
            rec["預測點誤差_B側px"] = round(euclidean(pB, gtB), 2)
            # 兩端誤差都大、但長度誤差很小 → 典型的「抵銷式假準確」
            if (min(e1, e2) > ENDPOINT_ERROR_PX
                    and abs(geo_len - euclidean(gtA, gtB)) < ENDPOINT_ERROR_PX):
                flags.append("端點誤差抵銷(長度假準)")

        rec["QC備註"] = "、".join(flags)
        rec["建議人工複查"] = "⚠️是" if flags else "否"
        rows.append(rec)

        if SAVE_VISUALIZATION:
            others = [p for i, p in enumerate(polygons) if i != chosen_idx]
            draw(img_path, poly, others, pA, pB, end1, end2,
                 Path(VIS_DIR) / f"{stem}_mask.jpg",
                 gt_pts=(gtA, gtB) if gtA is not None else None,
                 proj_pts=proj_pts)

    df = pd.DataFrame(rows)

    # ---- 併入 GT 長度，直接算三種方法的誤差 ----
    summary = pd.DataFrame()
    gt_lb = "真實AB像素長度_letterbox px"
    gt_orig = "真實AB像素長度_原圖px"

    if B2_XLSX and Path(B2_XLSX).exists():
        df_b2 = pd.read_excel(B2_XLSX, sheet_name=B2_SHEET)
        keep = [c for c in ("圖片檔名", gt_lb, gt_orig) if c in df_b2.columns]
        if len(keep) > 1:
            df = df.merge(df_b2[keep], on="圖片檔名", how="left")

        items = []
        # 兩個空間都算：letterbox 是模型工作的空間，原圖px 才是 C1 真正吃的東西。
        # 每張圖 scale 不同，換算後排序有可能翻盤，所以不能只看 letterbox 就下結論。
        for space, gcol, suffix in (("letterbox", gt_lb, "letterbox px"),
                                    ("原圖", gt_orig, "原圖px")):
            if gcol not in df.columns:
                continue
            for name in ("原始預測", "長軸投影", "mask幾何"):
                col = f"AB長度_{name}_{suffix}"
                if col not in df.columns:
                    continue
                d = (df[col] - df[gcol]).dropna()
                if d.empty:
                    continue
                items.append({
                    "空間": space,
                    "方法": name,
                    "有效張數": int(len(d)),
                    "長度MAE px": round(d.abs().mean(), 3),
                    "長度偏差(bias) px": round(d.mean(), 3),
                    "誤差標準差 px": round(d.std(ddof=1), 3) if len(d) > 1 else None,
                    "最大絕對誤差 px": round(d.abs().max(), 3),
                })
        summary = pd.DataFrame(items)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="逐顆牙對照", index=False)
        if not summary.empty:
            summary.to_excel(w, sheet_name="三法比較", index=False)
        if SAVE_VISUALIZATION:
            pd.DataFrame(LEGEND_ZH, columns=["圖上顏色", "代表什麼"]).to_excel(
                w, sheet_name="視覺化圖例", index=False)

    # ---- 終端摘要 ----
    print(f"\n✅ 完成，共處理 {len(df)} 張")
    print(f"📄 {OUTPUT_XLSX}")
    if SAVE_VISUALIZATION:
        print(f"🖼️  {VIS_DIR}/")
        print("   圖例(圖上標籤為英文，Hershey 字型無法顯示中文)：")
        for color, meaning in LEGEND_ZH:
            print(f"     {color:<16} {meaning}")

    if "mask狀態" in df.columns:
        n_fail = int((df["mask狀態"] != "✅").sum())
        if n_fail:
            print(f"❌ seg 未成功的有 {n_fail} 張")
    if "建議人工複查" in df.columns:
        print(f"⚠️  建議人工複查：{int((df['建議人工複查'] == '⚠️是').sum())} 張")
    if "兩法選取一致" in df.columns:
        n_diff = int((df["兩法選取一致"] == "⚠️否").sum())
        n_cmp = int(df["兩法選取一致"].isin(["是", "⚠️否"]).sum())
        if n_cmp:
            print(f"🔀 兩法挑到不同 mask：{n_diff}/{n_cmp} 張"
                  f"(這就是 points 模式將來上線時的預期錯誤率)")

    if "幾何端點誤差_B側px" in df.columns:
        sub = df[["幾何端點誤差_A側px", "幾何端點誤差_B側px",
                  "預測點誤差_A側px", "預測點誤差_B側px"]].dropna()
        if not sub.empty:
            print("\n=== 端點位置誤差(letterbox px，中位數/最大) ===")
            for c in sub.columns:
                print(f"  {c:<22} {sub[c].median():>7.2f} / {sub[c].max():>7.2f}")
            print("  ↑ B側(根尖)才是臨床風險所在；長度準但 B 側誤差大代表是抵銷出來的。")

    if not summary.empty:
        print("\n=== 三法比較 ===")
        print(summary.to_string(index=False))
        print("\n判讀：bias 大不要緊(下游 ANN 與 OFFSET_MM 可吸收)，")
        print("      要看的是「誤差標準差」——分散度小的那個方法才是真的比較好。")
        print("      以「原圖」那三列為準，那才是 C1 實際吃到的刻度。")


if __name__ == "__main__":
    main()
