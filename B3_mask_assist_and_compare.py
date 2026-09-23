#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3_mask_assist_and_compare.py  (seg-only 版)
===============================================================
YOLO11-pose 已從流程移除，本腳本只靠 segmentation：

    B1s 訓練 seg → B3 推論 + 幾何量測 → B4 合併 → C1

每張圖做三件事：
  1) 挑出目標牙的 mask
  2) mask 幾何長度：PCA 長軸兩端極值點的距離(原本的「方法3」)
  3) 冠寬：牙冠段垂直於長軸的最大寬度(給 B4 的「冠寬比例尺」)

*** 原本靠 pose 的三件事，改成什麼 ***

  (a) 挑 mask
      舊：pose 的 A/B 點落在哪顆 mask 裡(points 模式)
      新：center 模式 = 質心離影像中心最近的 mask。根尖片通常以目標牙
          置中拍攝。有 GT 標註時仍優先用 gt_iou，並同時算 center 的選擇，
          在「兩法選取一致」欄回報——這就是將來無 GT 上線時的預期錯誤率。

  (b) 哪一端是切端(A)
      舊：離 pose A 點近的那端
      新：ORIENT_MODE="arch"(預設)用醫師表的牙位決定——根尖片標準擺位下，
          上顎牙(1x/2x)牙冠朝下、下顎牙(3x/4x)牙冠朝上，取 y 較大/較小那端。
          查不到牙位時退回「寬度法」：兩端各 30% 長度的平均寬度，寬的是牙冠。
          兩法都算，「寬度法方向一致」欄回報是否吻合，不吻合的圖請人工看。
          (長度本身與方向無關，方向只影響冠寬量在哪一端)

      v1 用「兩端最大寬度」比，mask 端點只要有一根突刺就判反(004、012)，
      所以改成上面兩種。

  (c) scale(letterbox px → 原圖 px)
      舊：直接讀 pose CSV 的 縮放比scale
      新：SCALE_SOURCE 二選一
          "xlsx"     : 讀既有的 B2 輸出 Excel 的 縮放比scale 欄(只當查表用，
                       不需要重跑 pose)
          "original" : 讀原圖尺寸自己算 scale = max(原圖寬,原圖高) ÷ letterbox 邊長
                       (Roboflow「Fit (black edges)」letterbox 的換算)
          兩者都有時會交叉驗證，差超過 1% 在 QC 標出來。

*** 冠寬量法(v2) ***
  v1 取「切片的最外兩點距離」的最大值，mask 邊緣的突刺、碎片會直接變成
  最大寬(025、012)。v2：
    1) 先做形態學開運算 + 只留最大連通區，削掉細突刺與碎片
    2) 每片寬度改用「該片 mask 像素數」而非最外兩點距離(中間有缺口不會被撐寬)
    3) 跳過切端最前面 CROWN_SKIP_FRAC(切端圓角)，平滑窗加大
  最大寬位置正常應落在距切端約 15–35%(接觸點附近)。

*** 移除的輸出 ***
  原始預測、長軸投影兩種長度都需要 pose 點，已刪除。B4 只剩 mask幾何
  (+ 冠寬比例尺、牙位基準兩個衍生方法)。

*** GT 標註 ***
  GT_LABEL_DIR 兩種格式都吃，自動判斷：
    pose 格式(舊標註，有 A/B 點) → bbox 挑 mask + GT 長度 + 端點誤差
    seg 格式(多邊形)            → 只有 bbox 挑 mask，沒有 GT 長度
  seg 的 test 只標了目標牙，所以 seg 格式拿第一個(通常唯一)多邊形當目標。

⚠️ IMAGE_DIR 的前處理(CLAHE / unsharp)必須跟 seg 訓練時一致。
"""

import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ============================================================
# 第1部分：設定
# ============================================================

# --- 輸入 ---
SEG_WEIGHTS = "yolo11_seg_run/weights_ready.pt"
IMAGE_DIR = "test_segment/test/images"       # 👈 seg 的 test 影像(與 B2s 相同)
GT_LABEL_DIR = "test_segment/test/labels"    # 👈 pose 或 seg 格式皆可；留空則強制 center

# --- scale 來源 ---
SCALE_SOURCE = "auto"                        # "auto"(先查 xlsx，查不到用原圖) | "xlsx" | "original"
SCALE_XLSX = "scale表_備份_20260921.xlsx"
SCALE_XLSX_SHEET = "逐顆牙對照"
ORIGINAL_IMAGE_DIR = ""                      # 原始(未 letterbox)影像資料夾；有填就交叉驗證
SCALE_MISMATCH_TOL = 0.01                    # 兩種 scale 相差超過 1% → QC 標記

# --- 輸出 ---
OUTPUT_XLSX = "B3_mask輔助對照_11.xlsx"
VIS_DIR = "B3_mask視覺化"
SAVE_VISUALIZATION = True

# --- 推論參數 ---
IMG_SIZE = 640                     # 👈 跟 seg 訓練時一致
SEG_CONF = 0.15

# --- 挑 mask 方式 ---
MASK_SELECT_MODE = "auto"          # "auto"(有 GT 用 gt_iou，否則 center) | "gt_iou" | "center"
AMBIGUOUS_IOU_GAP = 0.15           # 最佳與次佳 mask 的 IoU 差距小於此值 → 標記選取不明確

# --- 方向判定 ---
ORIENT_MODE = "arch"               # "arch"(牙位決定，查不到退回 width) | "width"
MM_XLSX = "根管充填長度_20260904.xlsx"   # 只讀 圖片檔名 + 牙位
MM_SHEET = "資料填寫"

# --- QC 門檻 ---
MIN_AXIS_RATIO = 0.70              # 長軸解釋比低於此值 → 牙形不夠細長(實測門牙多在 0.70–0.82，0.75 會誤報一半)
ENDPOINT_ERROR_PX = 12.0           # 幾何端點離 GT 超過此距離 → 位置可疑(即使長度剛好)
CROWN_END_MIN_RATIO = 1.10         # 寬度法：兩端平均寬度比小於此值 → 判定不明確

# --- 冠寬 ---
CROWN_FRAC = 0.45                  # 只在距切端 CROWN_SKIP_FRAC ~ 45% 全長內找最大寬度
CROWN_SKIP_FRAC = 0.05             # 跳過切端圓角
OPEN_KERNEL_FRAC = 0.04            # 開運算核大小 = 全長 × 此值(削掉細突刺)
WIDTH_SMOOTH_FRAC = 0.03           # 平滑窗 = 全長 × 此值
WIDTH_POS_RANGE = (0.10, 0.40)     # 最大寬位置落在此範圍外 → QC 標記
WIDTH_RATIO_RANGE = (0.15, 0.50)   # 冠寬 ÷ 全長的合理範圍

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# --- 視覺化圖例 ---
SHOW_LEGEND = True
LEGEND_ITEMS = [                      # (圖上標籤, BGR)
    ("chosen mask (seg)", (0, 200, 255)),
    ("other masks",       (130, 130, 130)),
    ("mask length",       (255, 160, 0)),
    ("crown width",       (255, 255, 255)),
    ("GT annotation",     (255, 0, 255)),
]
LEGEND_ZH = [
    ("橘黃輪廓", "選中的 mask(seg 預測)"),
    ("灰色輪廓", "其他候選 mask，未被選中"),
    ("藍線 A-B", "mask 幾何長度(A=判定的切端、B=根尖端)"),
    ("白線", "冠寬 ← 檢查有沒有黏到鄰牙"),
    ("洋紅空心圈", "GT 人工標註的 A/B(僅 pose 格式標註有)"),
]


# ============================================================
# 第2部分：幾何工具
# ============================================================

def euclidean(p1, p2):
    return math.dist(p1, p2)


def yolo_to_corners(cx, cy, w, h):
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
    牙齒細長 → 接近 1；接近圓形 → 接近 0.5，此時長軸方向不可信。
    """
    pts = np.asarray(polygon, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    return centroid, vt[0], float(s[0] / max(s.sum(), 1e-9))


def axis_extremes(polygon, centroid, axis):
    """mask 沿長軸投影的兩個極值點。"""
    pts = np.asarray(polygon, dtype=np.float64)
    t = (pts - centroid) @ axis
    return centroid + axis * t.min(), centroid + axis * t.max()


def clean_mask(polygon, shape, length_px):
    """多邊形 → 實心 mask，開運算削突刺，只留最大連通區。"""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
    k = max(3, int(round(length_px * OPEN_KERNEL_FRAC)) | 1)
    m2 = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m2, connectivity=8)
    if n <= 1:
        return m                      # 開運算把整個 mask 吃掉 → 退回原 mask
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8)


def width_profile(mask, centroid, axis, a_end):
    """沿長軸從 a_end 那一側起每 1 px 一片。
    回傳 dict：d(片距離，從清理後 mask 的 a_end 側邊界起算)、w(片寬=像素數)、
    s_lo/s_hi(片的法線範圍)、t_edge(a_end 側邊界的 t)、sign、normal、total。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    normal = np.array([-axis[1], axis[0]])
    t = (pts - centroid) @ axis
    s = (pts - centroid) @ normal
    tA = float((np.asarray(a_end, dtype=np.float64) - centroid) @ axis)
    sign = 1.0 if tA <= 0 else -1.0
    t_edge = t.min() if sign > 0 else t.max()   # 清理後 mask 在 a_end 那側的邊界
    d = (t - t_edge) * sign
    total = float(d.max())
    if total <= 0:
        return None
    bins = np.floor(d).astype(int)
    order = np.argsort(bins, kind="stable")
    bins, s = bins[order], s[order]
    uniq, idx, cnt = np.unique(bins, return_index=True, return_counts=True)
    return {
        "d": uniq, "w": cnt.astype(float),
        "s_lo": np.minimum.reduceat(s, idx), "s_hi": np.maximum.reduceat(s, idx),
        "t_edge": float(t_edge), "sign": sign, "normal": normal, "total": total,
    }


def crown_width(mask, centroid, axis, a_end):
    """從切端起 CROWN_SKIP_FRAC ~ CROWN_FRAC 全長內的最大片寬(平滑後)。
    回傳 dict(width, pos_ratio, p1, p2)；失敗回傳 None。"""
    pf = width_profile(mask, centroid, axis, a_end)
    if pf is None:
        return None
    d, w, total = pf["d"], pf["w"], pf["total"]
    win = max(3, int(round(total * WIDTH_SMOOTH_FRAC)))
    smoothed = pd.Series(w).rolling(win, center=True, min_periods=1).median().values
    cand = np.where((d >= CROWN_SKIP_FRAC * total) & (d <= CROWN_FRAC * total))[0]
    if len(cand) == 0:
        return None
    k = cand[int(np.argmax(smoothed[cand]))]
    t_k = pf["t_edge"] + pf["sign"] * (d[k] + 0.5)
    mid = (pf["s_lo"][k] + pf["s_hi"][k]) / 2.0
    half = smoothed[k] / 2.0
    nrm = pf["normal"]
    return {
        "width": float(smoothed[k]),
        "pos_ratio": float((d[k] + 0.5) / total),
        "p1": centroid + axis * t_k + nrm * (mid - half),
        "p2": centroid + axis * t_k + nrm * (mid + half),
    }


def end_mean_width(mask, centroid, axis, end, frac=0.30):
    """從 end 那側起 frac 全長內的平均片寬(寬度法判方向用)。"""
    pf = width_profile(mask, centroid, axis, end)
    if pf is None:
        return float("nan")
    sel = pf["d"] <= frac * pf["total"]
    return float(pf["w"][sel].mean()) if sel.any() else float("nan")


def orient(mask, centroid, axis, end1, end2, position):
    """決定哪端是切端。回傳 (A端, B端, 依據, 寬度法兩端比, 寬度法是否同意)。

    arch ：上顎(1x/2x)牙冠朝下 → y 大的是 A；下顎(3x/4x)牙冠朝上 → y 小的是 A。
    width：兩端各 30% 長度的平均寬度，寬的是 A。
    """
    w1 = end_mean_width(mask, centroid, axis, end1)
    w2 = end_mean_width(mask, centroid, axis, end2)
    width_A_is_1 = w1 >= w2
    ratio = max(w1, w2) / min(w1, w2) if min(w1, w2) > 0 else float("nan")

    quadrant = position // 10 if position else None
    if ORIENT_MODE == "arch" and quadrant in (1, 2, 3, 4):
        end1_is_lower = end1[1] >= end2[1]
        A_is_1 = end1_is_lower if quadrant in (1, 2) else not end1_is_lower
        basis = "牙位(上顎冠朝下)" if quadrant in (1, 2) else "牙位(下顎冠朝上)"
    else:
        A_is_1 = width_A_is_1
        basis = "寬度法"
    A, B = (end1, end2) if A_is_1 else (end2, end1)
    return A, B, basis, ratio, (A_is_1 == width_A_is_1)


# ============================================================
# 第3部分：GT 讀取與 mask 挑選
# ============================================================

def load_gt_target(label_path: Path, width: int, height: int):
    """讀 GT 標註，自動判斷格式。回傳 (bbox, GT_A, GT_B, 格式)。

    pose 格式：class cx cy w h Ax Ay Av Bx By Bv → 取 A、B visibility 都是 2 的那行
    seg 格式 ：class x1 y1 x2 y2 ...(偶數個座標) → 取第一個多邊形
    """
    if not label_path.exists():
        return None, None, None, None
    seg_poly = None
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 11:
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
                    return bbox, gtA, gtB, "pose"
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0 and seg_poly is None:
                try:
                    xy = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
                except ValueError:
                    continue
                seg_poly = xy * [width, height]
    if seg_poly is not None:
        return polygon_bbox(seg_poly), None, None, "seg"
    return None, None, None, None


def pick_by_center(polygons, shape):
    """質心離影像中心最近的 mask。根尖片通常以目標牙置中。"""
    H, W = shape[:2]
    c = np.array([W / 2.0, H / 2.0])
    d = [float(np.linalg.norm(np.asarray(p, dtype=np.float64).mean(axis=0) - c))
         for p in polygons]
    return int(np.argmin(d))


def pick_by_gt_iou(polygons, gt_bbox):
    """mask 外接框與 GT 框 IoU 最高者勝。回傳 (最佳索引, 最佳IoU, 次佳IoU)。"""
    ious = [compute_iou(polygon_bbox(p), gt_bbox) for p in polygons]
    order = sorted(range(len(ious)), key=lambda i: ious[i], reverse=True)
    second = ious[order[1]] if len(order) > 1 else 0.0
    return order[0], ious[order[0]], second


# ============================================================
# 第4部分：scale
# ============================================================

_RF_SUFFIX = re.compile(r"_(?:jpg|jpeg|png)\.rf\.[0-9a-z]+$", re.I)


def base_key(name):
    """xxx_jpg.rf.<hash>.jpg → xxx，用來把 letterbox 圖對回原圖。"""
    s = _RF_SUFFIX.sub("", Path(str(name)).stem)
    return s.lower()


def load_scale_table():
    """讀既有 B2 Excel 的 縮放比scale 欄 → {base_key: scale}。

    用 base_key(剝掉 Roboflow 後綴)對，因為 seg 與 pose 是不同次匯出，
    同一張圖的 .rf.<hash> 不一樣，用完整檔名會全部對不上。
    """
    p = Path(SCALE_XLSX) if SCALE_XLSX else None
    if p is None or not p.exists():
        print(f"ℹ️  scale 表不存在：{p.resolve() if p else '(未設定)'}")
        return {}
    df = pd.read_excel(p, sheet_name=SCALE_XLSX_SHEET)
    if "縮放比scale" not in df.columns or "圖片檔名" not in df.columns:
        print(f"⚠️ {p} 沒有 圖片檔名/縮放比scale 欄，現有欄位：{list(df.columns)}")
        return {}
    return {base_key(r["圖片檔名"]): float(r["縮放比scale"])
            for _, r in df.iterrows() if pd.notna(r["縮放比scale"])}


def index_original_images():
    """原圖資料夾 → {base_key: 路徑}。"""
    if not ORIGINAL_IMAGE_DIR or not Path(ORIGINAL_IMAGE_DIR).exists():
        return {}
    return {base_key(p.name): p for p in Path(ORIGINAL_IMAGE_DIR).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS}


def scale_from_original(orig_path, lb_shape):
    """letterbox：原圖長邊縮到 letterbox 邊長，所以 scale = 原圖長邊 ÷ letterbox 邊長。"""
    im = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, None, None
    h0, w0 = im.shape[:2]
    return max(h0, w0) / float(max(lb_shape[:2])), w0, h0


def load_positions():
    """醫師表 → {base_key: 牙位}。只用來決定方向，不進模型。"""
    p = Path(MM_XLSX) if MM_XLSX else None
    if p is None or not p.exists():
        print(f"ℹ️  找不到醫師表 {p}，方向判定全部用寬度法")
        return {}
    raw = pd.read_excel(p, sheet_name=MM_SHEET, header=None, nrows=15)
    hdr = 0
    for i in range(len(raw)):
        cells = [str(c) for c in raw.iloc[i].tolist() if pd.notna(c)]
        if any("檔名" in c for c in cells) and any("牙位" in c for c in cells):
            hdr = i
            break
    df = pd.read_excel(p, sheet_name=MM_SHEET, header=hdr)
    ncol = next((c for c in df.columns if "檔名" in str(c)), None)
    pcol = next((c for c in df.columns if "牙位" in str(c)), None)
    if ncol is None or pcol is None:
        print("⚠️ 醫師表找不到 檔名/牙位 欄，方向判定全部用寬度法")
        return {}
    out = {}
    for _, r in df.iterrows():
        try:
            out[base_key(r[ncol])] = int(float(r[pcol]))
        except (ValueError, TypeError):
            pass
    return out


# ============================================================
# 第5部分：視覺化
# ============================================================

def draw_legend(img, items, line_h=17, pad=8, box_w=170):
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


def draw(img, chosen, others, endA, endB, out_path, gt_pts=None, width_pts=None):
    """翻圖時看三件事：
      1) 橘黃輪廓套在正確那顆牙上(認牙)
      2) A 在切端、B 在根尖(方向判定)，B 離洋紅 GT_B 多遠
      3) 白線兩端落在自己這顆牙的近遠心面，沒伸進鄰牙
    """
    img = img.copy()
    for poly in others:
        cv2.polylines(img, [np.asarray(poly, dtype=np.int32)], True, (130, 130, 130), 1)
    cv2.polylines(img, [np.asarray(chosen, dtype=np.int32)], True, (0, 200, 255), 2)
    cv2.line(img, tuple(np.int32(endA)), tuple(np.int32(endB)), (255, 160, 0), 2)
    for p, tag in ((endA, "A"), (endB, "B")):
        cv2.circle(img, tuple(np.int32(p)), 5, (255, 160, 0), -1)
        cv2.putText(img, tag, tuple(np.int32(p) + np.int32([6, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)
    if width_pts is not None:
        w1, w2 = width_pts
        cv2.line(img, tuple(np.int32(w1)), tuple(np.int32(w2)), (255, 255, 255), 2)
        for p in (w1, w2):
            cv2.circle(img, tuple(np.int32(p)), 3, (255, 255, 255), -1)
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
# 第6部分：主流程
# ============================================================

def main():
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"找不到影像資料夾：{image_dir}")
    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    gt_dir = Path(GT_LABEL_DIR) if GT_LABEL_DIR else None
    model = YOLO(SEG_WEIGHTS)

    scale_tab = load_scale_table()
    positions = load_positions()
    orig_idx = index_original_images()
    if not scale_tab and not orig_idx:
        raise FileNotFoundError(
            "兩個 scale 來源都拿不到：\n"
            f"   SCALE_XLSX = {SCALE_XLSX!r}(工作目錄 {Path.cwd()})\n"
            f"   ORIGINAL_IMAGE_DIR = {ORIGINAL_IMAGE_DIR!r}\n"
            "   至少要有一個：B2 的舊 Excel(只查表，不用重跑 pose)，或原始未 letterbox 的影像資料夾。")
    if SCALE_SOURCE == "xlsx" and not scale_tab:
        raise FileNotFoundError(f"SCALE_SOURCE='xlsx' 但讀不到 {SCALE_XLSX}；改 'auto' 或 'original'。")
    if SCALE_SOURCE == "original" and not orig_idx:
        raise FileNotFoundError("SCALE_SOURCE='original' 但 ORIGINAL_IMAGE_DIR 沒填或是空的。")

    note = "(有 GT 走 gt_iou，無則 center)" if MASK_SELECT_MODE == "auto" else ""
    print(f"挑 mask 模式：{MASK_SELECT_MODE}{note}")
    print(f"scale 來源：{SCALE_SOURCE}"
          f"{'(並用原圖交叉驗證)' if SCALE_SOURCE == 'xlsx' and orig_idx else ''}")
    print(f"待處理：{len(images)} 張\n")

    rows = []
    for img_path in images:
        fname = img_path.name
        rec = {"圖片檔名": fname}
        flags = []

        img = cv2.imread(str(img_path))
        if img is None:
            rec.update({"mask狀態": "❌讀不到圖片", "建議人工複查": "⚠️是"})
            rows.append(rec)
            continue
        H, W = img.shape[:2]

        # ---- scale ----
        s_xlsx = scale_tab.get(base_key(fname))
        s_orig, w0, h0 = None, None, None
        op = orig_idx.get(base_key(fname))
        if op is not None:
            s_orig, w0, h0 = scale_from_original(op, img.shape)
        if SCALE_SOURCE == "xlsx":
            scale = s_xlsx
        elif SCALE_SOURCE == "original":
            scale = s_orig
        else:
            scale = s_xlsx if s_xlsx is not None else s_orig
        rec.update({"縮放比scale": scale, "原圖寬": w0, "原圖高": h0})
        if s_xlsx is not None and s_orig is not None:
            rel = abs(s_xlsx - s_orig) / s_orig
            rec["scale驗證_相對差"] = round(rel, 4)
            if rel > SCALE_MISMATCH_TOL:
                flags.append("scale兩來源不一致")
        if scale is None:
            rec.update({"mask狀態": "❌缺scale", "建議人工複查": "⚠️是"})
            rows.append(rec)
            continue

        # ---- seg 推論 ----
        res = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                            conf=SEG_CONF, save=False, verbose=False)[0]
        polygons = [p for p in (res.masks.xy if res.masks is not None else [])
                    if p is not None and len(p) >= 3]
        if not polygons:
            rec.update({"mask狀態": "❌seg無輸出", "偵測到mask數": 0, "建議人工複查": "⚠️是"})
            rows.append(rec)
            continue

        # ---- 挑目標牙 ----
        gt_bbox, gtA, gtB, gt_fmt = (load_gt_target(gt_dir / f"{img_path.stem}.txt", W, H)
                                     if gt_dir else (None, None, None, None))
        idx_ctr = pick_by_center(polygons, img.shape)
        idx_iou, best_iou, second_iou = None, None, None
        if gt_bbox is not None:
            idx_iou, best_iou, second_iou = pick_by_gt_iou(polygons, gt_bbox)

        if MASK_SELECT_MODE == "center" or idx_iou is None:
            chosen_idx, used_mode = idx_ctr, "center"
        else:
            chosen_idx, used_mode = idx_iou, "gt_iou"

        if idx_iou is not None and idx_ctr != idx_iou:
            flags.append("兩法選到不同mask")
        if (best_iou is not None and second_iou is not None
                and (best_iou - second_iou) < AMBIGUOUS_IOU_GAP):
            flags.append("選取不明確")

        # ---- 幾何 ----
        poly = polygons[chosen_idx]
        centroid, axis, ratio = mask_axis(poly)
        e1, e2 = axis_extremes(poly, centroid, axis)
        geo_len = euclidean(e1, e2)          # 長度仍用原始 mask，跟舊版可比
        position = positions.get(base_key(fname))

        mask = clean_mask(poly, img.shape, geo_len)
        endA, endB, basis, end_ratio, agree = orient(mask, centroid, axis, e1, e2, position)
        cw = crown_width(mask, centroid, axis, endA)

        if ratio < MIN_AXIS_RATIO:
            flags.append("牙形不夠細長")
        if basis == "寬度法" and not (end_ratio >= CROWN_END_MIN_RATIO):
            flags.append("切端判定不明確")
        if basis != "寬度法" and not agree:
            flags.append("牙位與寬度法方向不一致")
        if cw is None:
            flags.append("冠寬量測失敗")
        else:
            wr = cw["width"] / geo_len if geo_len > 0 else float("nan")
            lo, hi = WIDTH_RATIO_RANGE
            if not (lo <= wr <= hi):
                flags.append("冠寬比例異常(疑似黏鄰牙或mask殘缺)")
            plo, phi = WIDTH_POS_RANGE
            if not (plo <= cw["pos_ratio"] <= phi):
                flags.append("冠寬位置異常")

        rec.update({
            "mask狀態": "✅",
            "偵測到mask數": len(polygons),
            "選取方式": used_mode,
            "GT格式": gt_fmt or "—",
            "兩法選取一致": ("—" if idx_iou is None else ("是" if idx_ctr == idx_iou else "⚠️否")),
            "mask_IoU_vs_GT框": round(best_iou, 3) if best_iou is not None else None,
            "次佳mask_IoU": round(second_iou, 3) if second_iou is not None else None,
            "長軸解釋比": round(ratio, 3),
            "牙位": position,
            "方向依據": basis,
            "寬度法方向一致": "是" if agree else "⚠️否",
            "兩端寬度比": round(end_ratio, 3) if end_ratio == end_ratio else None,
            "AB長度_mask幾何_letterbox px": round(geo_len, 4),
            "AB長度_mask幾何_原圖px": round(geo_len * scale, 4),
        })
        if cw is not None:
            rec.update({
                "冠寬_letterbox px": round(cw["width"], 4),
                "冠寬_原圖px": round(cw["width"] * scale, 4),
                "冠寬位置_距A端比例": round(cw["pos_ratio"], 3),
                "長寬比_mask幾何÷冠寬": round(geo_len / cw["width"], 4),
            })

        # ---- 有 pose 格式 GT 時：GT 長度、端點誤差、方向驗證 ----
        if gtA is not None:
            gt_len = euclidean(gtA, gtB)
            rec["真實AB像素長度_letterbox px"] = round(gt_len, 4)
            rec["真實AB像素長度_原圖px"] = round(gt_len * scale, 4)
            eA, eB = euclidean(endA, gtA), euclidean(endB, gtB)
            rec["幾何端點誤差_A側px"] = round(eA, 2)
            rec["幾何端點誤差_B側px"] = round(eB, 2)
            same_dir = (euclidean(endA, gtA) + euclidean(endB, gtB)
                        <= euclidean(endA, gtB) + euclidean(endB, gtA))
            rec["A端判定與GT一致"] = "是" if same_dir else "⚠️否"
            if not same_dir:
                flags.append("切端方向判反")
            if (min(eA, eB) > ENDPOINT_ERROR_PX
                    and abs(geo_len - gt_len) < ENDPOINT_ERROR_PX):
                flags.append("端點誤差抵銷(長度假準)")

        rec["QC備註"] = "、".join(flags)
        rec["建議人工複查"] = "⚠️是" if flags else "否"
        rows.append(rec)

        if SAVE_VISUALIZATION:
            others = [p for i, p in enumerate(polygons) if i != chosen_idx]
            draw(img, poly, others, endA, endB,
                 Path(VIS_DIR) / f"{img_path.stem}_mask.jpg",
                 gt_pts=(gtA, gtB) if gtA is not None else None,
                 width_pts=(cw["p1"], cw["p2"]) if cw is not None else None)

    df = pd.DataFrame(rows)

    # ---- mask幾何 vs GT 長度 ----
    items = []
    for space, gcol, col in (
            ("letterbox", "真實AB像素長度_letterbox px", "AB長度_mask幾何_letterbox px"),
            ("原圖", "真實AB像素長度_原圖px", "AB長度_mask幾何_原圖px")):
        if gcol not in df.columns or col not in df.columns:
            continue
        d = (df[col] - df[gcol]).dropna()
        if d.empty:
            continue
        items.append({
            "空間": space, "方法": "mask幾何", "有效張數": int(len(d)),
            "長度MAE px": round(d.abs().mean(), 3),
            "長度偏差(bias) px": round(d.mean(), 3),
            "誤差標準差 px": round(d.std(ddof=1), 3) if len(d) > 1 else None,
            "最大絕對誤差 px": round(d.abs().max(), 3),
        })
    summary = pd.DataFrame(items)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="逐顆牙對照", index=False)
        if not summary.empty:
            summary.to_excel(w, sheet_name="長度vsGT", index=False)
        if SAVE_VISUALIZATION:
            pd.DataFrame(LEGEND_ZH, columns=["圖上顏色", "代表什麼"]).to_excel(
                w, sheet_name="視覺化圖例", index=False)

    # ---- 終端摘要 ----
    print(f"✅ 完成，共處理 {len(df)} 張")
    print(f"📄 {OUTPUT_XLSX}")
    if SAVE_VISUALIZATION:
        print(f"🖼️  {VIS_DIR}/")
        for color, meaning in LEGEND_ZH:
            print(f"     {color:<12} {meaning}")

    n_fail = int((df["mask狀態"] != "✅").sum())
    if n_fail:
        print(f"❌ 未成功：{n_fail} 張(見 mask狀態 欄)")
    print(f"⚠️  建議人工複查：{int((df['建議人工複查'] == '⚠️是').sum())} 張")

    if "兩法選取一致" in df.columns:
        n_diff = int((df["兩法選取一致"] == "⚠️否").sum())
        n_cmp = int(df["兩法選取一致"].isin(["是", "⚠️否"]).sum())
        if n_cmp:
            print(f"🔀 center 與 gt_iou 挑到不同 mask：{n_diff}/{n_cmp} 張"
                  f"(= 無 GT 上線時的預期認牙錯誤率)")
    if "A端判定與GT一致" in df.columns:
        n_rev = int((df["A端判定與GT一致"] == "⚠️否").sum())
        n_chk = int(df["A端判定與GT一致"].notna().sum())
        print(f"↕️  切端方向判反：{n_rev}/{n_chk} 張")
    if "scale驗證_相對差" in df.columns:
        rel = df["scale驗證_相對差"].dropna()
        print(f"📏 scale 兩來源相對差：最大 {rel.max():.4f}"
              f"(>{SCALE_MISMATCH_TOL} 的 {int((rel > SCALE_MISMATCH_TOL).sum())} 張)")
    if "冠寬_letterbox px" in df.columns:
        wcol = df["冠寬_letterbox px"].dropna()
        n_bad = int(df["QC備註"].fillna("").str.contains("冠寬").sum())
        print(f"\n=== 冠寬 ===")
        print(f"  成功 {len(wcol)} 張，letterbox px 中位數 {wcol.median():.1f}"
              f"(範圍 {wcol.min():.1f}–{wcol.max():.1f})；相關 QC 旗標 {n_bad} 張")

    if not summary.empty:
        print("\n=== mask幾何 vs GT 長度 ===")
        print(summary.to_string(index=False))
        print("看「誤差標準差」；bias 由下游 C1 吸收。以「原圖」列為準。")
    elif gt_dir:
        print("\n(GT 標註是 seg 格式，沒有 A/B 點，無法算長度誤差——C1 用醫師 mm 驗證即可)")


if __name__ == "__main__":
    main()