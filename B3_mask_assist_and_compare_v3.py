#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B3_mask_assist_and_compare_v3.py  (seg-only 版)
===============================================================
YOLO11-pose 已從流程移除，本腳本只靠 segmentation：

    B1s 訓練 seg → B3 推論 + 幾何量測 → B4 合併 → C1

每張圖做三件事：
  1) 挑出目標牙的 mask
  2) mask 幾何長度：PCA 長軸兩端極值點的距離(原本的「方法3」)
  3) 冠寬：牙冠段垂直於長軸的最大寬度(給 B4 的「冠寬比例尺」)

*** v3 變更 ***
  (4) 新增 sheet「逐張長度誤差」：每張圖 預測AB長度 / GT AB長度 / 誤差(有號) /
      絕對誤差 / 相對誤差%，原圖 px 與 letterbox px 都給，依絕對誤差由大到小排。
      (單張的 MAE 就是該張的絕對誤差；整體 MAE = 這欄的平均)
  (5)「長度vsGT」摘要多分 全部 / 上顎 / 下顎 三組，並加中位數絕對誤差。
  (6) 逐顆牙對照多一欄「上下顎」(由牙位第一碼：1/2 上顎、3/4 下顎)。
  (7) 終端印出絕對誤差最大的 TOP_N_WORST 張。

*** v2 變更 ***
  (1) 剔除碎塊 REMOVE_FRAGMENTS
      masks.xy 會把同一個 mask 的所有碎塊串成一條多邊形(碎塊間有細連線)，
      碎塊會把 PCA 長軸與 B 端拉歪。開啟時每個預測 mask 只留最大連通區；
      GT 多邊形也走同一個函式(填實→取輪廓)，避免兩邊處理不同造成 ~1px 系統偏差。
      對照實驗：先 False 跑一次當 baseline，再 True 跑一次比「長度vsGT」。
  (2) seg 格式 GT 也算 AB 長度誤差
      對 GT 多邊形套「同一套」幾何(PCA 極值點 + 牙位定方向)得到 GT_A/GT_B。
      這量的是「seg 輪廓誤差造成的長度誤差」，不是跟人工解剖學 A/B 點比。
      (seg 格式下「A端判定與GT一致」兩邊都用牙位定方向，幾乎恆為「是」，參考價值低)
  (3) 圖上左上角印 B2 的逐張 mAP50-95(讀 B2_XLSX) 與 |ΔAB|；圖例只列有畫出來的項目。

*** 挑 mask ***
  center 模式 = 質心離影像中心最近的 mask；有 GT 時優先用 gt_iou，並同時算
  center 的選擇，在「兩法選取一致」欄回報(= 無 GT 上線時的預期認牙錯誤率)。

*** 方向(哪端是切端 A) ***
  ORIENT_MODE="arch"：上顎(1x/2x)牙冠朝下、下顎(3x/4x)牙冠朝上。
  查不到牙位時退回「寬度法」：兩端各 30% 長度的平均寬度，寬的是牙冠。

*** scale(letterbox px → 原圖 px) ***
  "xlsx" 讀舊 Excel 的 縮放比scale 欄；"original" 用原圖長邊 ÷ letterbox 邊長；
  兩者都有時交叉驗證，差超過 1% 在 QC 標出來。

*** 冠寬量法(v2) ***
  開運算 + 最大連通區 → 每片寬度用像素數 → 跳過切端圓角、平滑。
  最大寬位置正常應落在距切端約 15–35%。

*** GT 標註 ***
  GT_LABEL_DIR 兩種格式都吃，自動判斷：
    pose 格式(A/B 點) → bbox 挑 mask + GT 長度 + 端點誤差
    seg 格式(多邊形)  → bbox 挑 mask + 由 GT 多邊形幾何算出的 GT 長度 + 端點誤差

⚠️ IMAGE_DIR 的前處理(CLAHE / unsharp)必須跟 seg 訓練時一致。
⚠️ 先跑 B2 再跑 B3(B3 讀 B2 的 Excel 拿 mAP；讀不到只是圖上不印 mAP)。
"""

import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ============================================================
# 第1部分：設定
# ============================================================

# --- 輸入 ---
SEG_WEIGHTS = "yolo11n_seg_run_padded_noCLAHE/weights_ready.pt"
IMAGE_DIR = "A5_test_padded_noCLAHE/images"       # 👈 seg 的 test 影像(與 B2 相同)
GT_LABEL_DIR = "A5_test_padded_noCLAHE/labels"    # 👈 pose 或 seg 格式皆可；留空則強制 center
B2_XLSX = "B2_seg推論統計_11n_padded_noCLAHE.xlsx"                  # 👈 B2 的輸出，只拿 Mask_mAP50-95 印在圖上

# --- scale 來源 ---
SCALE_SOURCE = "auto"                      # "auto"(先查 xlsx，查不到用原圖) | "xlsx" | "original"
SCALE_XLSX = ""
SCALE_XLSX_SHEET = "逐顆牙對照"
ORIGINAL_IMAGE_DIR = "test_single_seg-67_no_resize/test/images"  # 原始(未 letterbox)影像資料夾；有填就交叉驗證
SCALE_MISMATCH_TOL = 0.01                  # 兩種 scale 相差超過 1% → QC 標記

# --- 輸出 ---
OUTPUT_XLSX = "B3_mask輔助對照_11n_padded_noCLAHE.xlsx"
VIS_DIR = "B3_mask視覺化_11n_padded_noCLAHE"
SAVE_VISUALIZATION = True
TOP_N_WORST = 10                   # 終端印出絕對誤差最大的前 N 張(0 = 不印)

# --- 推論參數 ---
IMG_SIZE = 640                     # 👈 跟 seg 訓練時一致
SEG_CONF = 0.15
REMOVE_FRAGMENTS = True            # 👈 對照實驗開關：False = baseline(原始多邊形)

# --- 挑 mask 方式 ---
MASK_SELECT_MODE = "auto"          # "auto"(有 GT 用 gt_iou，否則 center) | "gt_iou" | "center"
AMBIGUOUS_IOU_GAP = 0.15           # 最佳與次佳 mask 的 IoU 差距小於此值 → 標記選取不明確

# --- 方向判定 ---
ORIENT_MODE = "arch"               # "arch"(牙位決定，查不到退回 width) | "width"
MM_XLSX = "根管充填長度_20260921-67.xlsx"   # 只讀 圖片檔名 + 牙位
MM_SHEET = "資料填寫"

# --- QC 門檻 ---
MIN_AXIS_RATIO = 0.70              # 長軸解釋比低於此值 → 牙形不夠細長
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

# --- 視覺化圖例(只列該張圖有畫出來的項目) ---
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
    ("洋紅空心圈", "GT 的 A/B(pose 標註點，或由 GT 多邊形以同一套幾何算出)"),
    ("左上綠字", "B2 的逐張 mAP50-95、預測/GT 長度與 |ΔAB| = |預測長度 − GT 長度|(原圖 px)"),
]

# 逐張長度誤差 sheet 的欄位(原圖 px 在前，為準)
PER_IMAGE_COLS = [
    "圖片檔名", "牙位", "上下顎",
    "AB長度_mask幾何_原圖px", "真實AB像素長度_原圖px",
    "AB長度誤差_預測減GT_原圖px", "AB長度絕對誤差_原圖px", "相對誤差%",
    "AB長度_mask幾何_letterbox px", "真實AB像素長度_letterbox px",
    "AB長度誤差_預測減GT_letterbox px", "AB長度絕對誤差_letterbox px",
    "幾何端點誤差_A側px", "幾何端點誤差_B側px",
    "建議人工複查", "QC備註",
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


def largest_part(polygon, shape):
    """masks.xy 會把同一 mask 的碎塊串成一條多邊形(中間有細連線)。
    填實 → 3x3 開運算切斷細連線 → 留最大連通區 → 取回外輪廓。
    回傳 (多邊形, 丟掉的碎塊數)。與 B2 的同名函式完全相同。"""
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


def mask_axis(polygon):
    """對 mask 多邊形做 PCA，回傳 (質心, 長軸單位向量, 長軸解釋比)。"""
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
    """沿長軸從 a_end 那一側起每 1 px 一片。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    normal = np.array([-axis[1], axis[0]])
    t = (pts - centroid) @ axis
    s = (pts - centroid) @ normal
    tA = float((np.asarray(a_end, dtype=np.float64) - centroid) @ axis)
    sign = 1.0 if tA <= 0 else -1.0
    t_edge = t.min() if sign > 0 else t.max()
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
    """從切端起 CROWN_SKIP_FRAC ~ CROWN_FRAC 全長內的最大片寬(平滑後)。"""
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
    """決定哪端是切端。回傳 (A端, B端, 依據, 寬度法兩端比, 寬度法是否同意)。"""
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


def geometry_AB(poly, shape, position):
    """預測與 GT 共用的幾何：PCA 極值點 + 定方向。回傳 (A, B, 長度, 其他)。"""
    centroid, axis, ratio = mask_axis(poly)
    e1, e2 = axis_extremes(poly, centroid, axis)
    length = euclidean(e1, e2)
    mask = clean_mask(poly, shape, length)
    A, B, basis, end_ratio, agree = orient(mask, centroid, axis, e1, e2, position)
    return A, B, length, dict(centroid=centroid, axis=axis, ratio=ratio, mask=mask,
                              basis=basis, end_ratio=end_ratio, agree=agree)


def arch_of(position):
    """牙位第一碼 → 上顎 / 下顎；查不到回 None。"""
    if not position:
        return None
    q = int(position) // 10
    return "上顎" if q in (1, 2) else ("下顎" if q in (3, 4) else None)


# ============================================================
# 第3部分：GT 讀取與 mask 挑選
# ============================================================

def load_gt_target(label_path: Path, width: int, height: int):
    """讀 GT 標註，自動判斷格式。回傳 (bbox, GT_A, GT_B, 格式, seg多邊形)。"""
    if not label_path.exists():
        return None, None, None, None, None
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
                    return bbox, gtA, gtB, "pose", None
            elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0 and seg_poly is None:
                try:
                    xy = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
                except ValueError:
                    continue
                seg_poly = xy * [width, height]
    if seg_poly is not None:
        return polygon_bbox(seg_poly), None, None, "seg", seg_poly
    return None, None, None, None, None


def pick_by_center(polygons, shape):
    """質心離影像中心最近的 mask。"""
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
# 第4部分：scale / 外部表
# ============================================================

_RF_SUFFIX = re.compile(r"_(?:jpg|jpeg|png)\.rf\.[0-9a-z]+$", re.I)
_LEADING_NUM = re.compile(r"^\s*0*(\d+)")


def base_key(name):
    """取檔名開頭的編號當 key：060_xxx.jpg / 60-yyy.png → '60'。
    開頭不是數字的檔名才退回舊規則(剝 Roboflow 後綴)。"""
    stem = Path(str(name)).stem
    m = _LEADING_NUM.match(stem)
    if m:
        return str(int(m.group(1)))
    return _RF_SUFFIX.sub("", stem).lower()


def load_scale_table():
    """讀既有 Excel 的 縮放比scale 欄 → {base_key: scale}。"""
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
    """scale = 原圖長邊 ÷ letterbox 邊長。"""
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


def load_b2_map():
    """B2 輸出 → {圖片檔名: Mask_mAP50-95}。B2 與 B3 用同一個 IMAGE_DIR，檔名一致。"""
    p = Path(B2_XLSX) if B2_XLSX else None
    if p is None or not p.exists():
        print(f"ℹ️  找不到 B2 輸出 {p}，圖上不印 mAP(先跑 B2)")
        return {}
    df = pd.read_excel(p)
    if "圖片檔名" not in df.columns or "Mask_mAP50-95" not in df.columns:
        print(f"⚠️ {p} 沒有 圖片檔名/Mask_mAP50-95 欄，圖上不印 mAP")
        return {}
    return dict(zip(df["圖片檔名"].astype(str), df["Mask_mAP50-95"]))


# ============================================================
# 第5部分：視覺化
# ============================================================

def draw_legend(img, items, y0=8, line_h=17, pad=8, box_w=170):
    if not items:
        return
    h = line_h * len(items) + pad * 2
    x0 = pad
    x1, y1 = min(x0 + box_w, img.shape[1] - 1), min(y0 + h, img.shape[0] - 1)
    roi = img[y0:y1, x0:x1]
    if roi.size:
        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.25, np.zeros_like(roi), 0.75, 0)
    for i, (label, color) in enumerate(items):
        y = y0 + pad + line_h * i + 10
        cv2.line(img, (x0 + 6, y - 4), (x0 + 24, y - 4), color, 3)
        cv2.putText(img, label, (x0 + 30, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, color, 1, cv2.LINE_AA)


def draw(img, chosen, others, endA, endB, out_path, gt_pts=None, width_pts=None, info=()):
    """翻圖時看三件事：
      1) 橘黃輪廓套在正確那顆牙上(認牙)
      2) A 在切端、B 在根尖(方向判定)，B 離洋紅 GT_B 多遠
      3) 白線兩端落在自己這顆牙的近遠心面，沒伸進鄰牙
    """
    img = img.copy()
    used = {"chosen mask (seg)", "mask length"}
    for poly in others:
        cv2.polylines(img, [np.asarray(poly, dtype=np.int32)], True, (130, 130, 130), 1)
    if others:
        used.add("other masks")
    cv2.polylines(img, [np.asarray(chosen, dtype=np.int32)], True, (0, 200, 255), 2)
    cv2.line(img, tuple(np.int32(endA)), tuple(np.int32(endB)), (255, 160, 0), 2)
    for p, tag in ((endA, "A"), (endB, "B")):
        cv2.circle(img, tuple(np.int32(p)), 5, (255, 160, 0), -1)
        cv2.putText(img, tag, tuple(np.int32(p) + np.int32([6, -6])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)
    if width_pts is not None:
        used.add("crown width")
        w1, w2 = width_pts
        cv2.line(img, tuple(np.int32(w1)), tuple(np.int32(w2)), (255, 255, 255), 2)
        for p in (w1, w2):
            cv2.circle(img, tuple(np.int32(p)), 3, (255, 255, 255), -1)
    if gt_pts is not None:
        used.add("GT annotation")
        gtA, gtB = gt_pts
        cv2.line(img, tuple(np.int32(gtA)), tuple(np.int32(gtB)), (255, 0, 255), 1)
        for p, tag in ((gtA, "GT_A"), (gtB, "GT_B")):
            cv2.circle(img, tuple(np.int32(p)), 7, (255, 0, 255), 2)
            cv2.putText(img, tag, tuple(np.int32(p) + np.int32([9, 4])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

    # ---- 左上角：mAP、長度、|ΔAB|，圖例接在下面 ----
    y = 8
    for txt in info:
        (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (0, y - 4), (tw + 16, y + th + base + 4), (0, 0, 0), -1)
        cv2.putText(img, txt, (8, y + th), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += th + base + 10
    if SHOW_LEGEND:
        draw_legend(img, [it for it in LEGEND_ITEMS if it[0] in used], y0=y)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


# ============================================================
# 第6部分：彙整(逐張誤差表 / 摘要)
# ============================================================

def build_per_image(df):
    """逐張長度誤差：只留算得出 GT 長度的圖，依原圖 px 絕對誤差由大到小。"""
    key = "AB長度絕對誤差_原圖px"
    if key not in df.columns:
        return pd.DataFrame()
    t = df[df[key].notna()].copy()
    if t.empty:
        return t
    t["相對誤差%"] = (t[key] / t["真實AB像素長度_原圖px"] * 100).round(2)
    if "牙位" in t.columns:
        t["牙位"] = pd.to_numeric(t["牙位"], errors="coerce").astype("Int64")
    t = t[[c for c in PER_IMAGE_COLS if c in t.columns]]
    return t.sort_values(key, ascending=False).reset_index(drop=True)


def build_summary(df):
    """長度 MAE 摘要：全部 / 上顎 / 下顎 × letterbox / 原圖。"""
    groups = [("全部", df)]
    if "上下顎" in df.columns:
        for g in ("上顎", "下顎"):
            sub = df[df["上下顎"] == g]
            if len(sub):
                groups.append((g, sub))
    items = []
    for gname, sub in groups:
        for space, gcol, col in (
                ("原圖", "真實AB像素長度_原圖px", "AB長度_mask幾何_原圖px"),
                ("letterbox", "真實AB像素長度_letterbox px", "AB長度_mask幾何_letterbox px")):
            if gcol not in sub.columns or col not in sub.columns:
                continue
            d = (sub[col] - sub[gcol]).dropna()
            if d.empty:
                continue
            rel = (d.abs() / sub.loc[d.index, gcol] * 100)
            items.append({
                "牙弓": gname, "空間": space, "方法": "mask幾何",
                "剔除碎塊": REMOVE_FRAGMENTS,
                "有效張數": int(len(d)),
                "長度MAE px": round(d.abs().mean(), 3),
                "中位數絕對誤差 px": round(d.abs().median(), 3),
                "平均相對誤差%": round(rel.mean(), 2),
                "長度偏差(bias) px": round(d.mean(), 3),
                "誤差標準差 px": round(d.std(ddof=1), 3) if len(d) > 1 else None,
                "最大絕對誤差 px": round(d.abs().max(), 3),
            })
    return pd.DataFrame(items)


# ============================================================
# 第7部分：主流程
# ============================================================

def main():
    from ultralytics import YOLO

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
    b2_map = load_b2_map()
    if not scale_tab and not orig_idx:
        raise FileNotFoundError(
            "兩個 scale 來源都拿不到：\n"
            f"   SCALE_XLSX = {SCALE_XLSX!r}(工作目錄 {Path.cwd()})\n"
            f"   ORIGINAL_IMAGE_DIR = {ORIGINAL_IMAGE_DIR!r}\n"
            "   至少要有一個：舊 Excel(只查表)，或原始未 letterbox 的影像資料夾。")
    if SCALE_SOURCE == "xlsx" and not scale_tab:
        raise FileNotFoundError(f"SCALE_SOURCE='xlsx' 但讀不到 {SCALE_XLSX}；改 'auto' 或 'original'。")
    if SCALE_SOURCE == "original" and not orig_idx:
        raise FileNotFoundError("SCALE_SOURCE='original' 但 ORIGINAL_IMAGE_DIR 沒填或是空的。")

    note = "(有 GT 走 gt_iou，無則 center)" if MASK_SELECT_MODE == "auto" else ""
    print(f"挑 mask 模式：{MASK_SELECT_MODE}{note}")
    print(f"scale 來源：{SCALE_SOURCE}"
          f"{'(並用原圖交叉驗證)' if SCALE_SOURCE == 'xlsx' and orig_idx else ''}")
    print(f"剔除碎塊：{REMOVE_FRAGMENTS}")
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

        # ---- seg 推論(+ 剔除碎塊) ----
        res = model.predict(source=str(img_path), imgsz=IMG_SIZE,
                            conf=SEG_CONF, save=False, verbose=False)[0]
        raw = [np.asarray(p, dtype=np.float64)
               for p in (res.masks.xy if res.masks is not None else [])
               if p is not None and len(p) >= 3]
        if REMOVE_FRAGMENTS:
            cleaned = [largest_part(p, img.shape) for p in raw]
            polygons = [c for c, _ in cleaned]
            n_frag = [k for _, k in cleaned]
        else:
            polygons, n_frag = raw, [0] * len(raw)
        if not polygons:
            rec.update({"mask狀態": "❌seg無輸出", "偵測到mask數": 0, "建議人工複查": "⚠️是"})
            rows.append(rec)
            continue

        # ---- 挑目標牙 ----
        gt_bbox, gtA, gtB, gt_fmt, gt_poly = (
            load_gt_target(gt_dir / f"{img_path.stem}.txt", W, H)
            if gt_dir else (None,) * 5)
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
        position = positions.get(base_key(fname))
        endA, endB, geo_len, g = geometry_AB(poly, img.shape, position)
        centroid, axis, ratio, mask = g["centroid"], g["axis"], g["ratio"], g["mask"]
        basis, end_ratio, agree = g["basis"], g["end_ratio"], g["agree"]
        cw = crown_width(mask, centroid, axis, endA)

        # ---- seg 格式 GT：對 GT 多邊形套同一套幾何 → GT_A / GT_B ----
        if gt_fmt == "seg" and gt_poly is not None:
            gp = largest_part(gt_poly, img.shape)[0] if REMOVE_FRAGMENTS else gt_poly
            gtA, gtB, _, _ = geometry_AB(gp, img.shape, position)

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
            "剔除碎塊數": n_frag[chosen_idx],
            "選取方式": used_mode,
            "GT格式": gt_fmt or "—",
            "兩法選取一致": ("—" if idx_iou is None else ("是" if idx_ctr == idx_iou else "⚠️否")),
            "mask_IoU_vs_GT框": round(best_iou, 3) if best_iou is not None else None,
            "次佳mask_IoU": round(second_iou, 3) if second_iou is not None else None,
            "長軸解釋比": round(ratio, 3),
            "牙位": position,
            "上下顎": arch_of(position),
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

        # ---- 有 GT A/B 時：GT 長度、逐張誤差、端點誤差、方向驗證 ----
        if gtA is not None:
            gt_len = euclidean(gtA, gtB)
            err = geo_len - gt_len                       # 有號：正 = 預測偏長
            rec["真實AB像素長度_letterbox px"] = round(gt_len, 4)
            rec["真實AB像素長度_原圖px"] = round(gt_len * scale, 4)
            rec["AB長度誤差_預測減GT_letterbox px"] = round(err, 4)
            rec["AB長度誤差_預測減GT_原圖px"] = round(err * scale, 4)
            rec["AB長度絕對誤差_letterbox px"] = round(abs(err), 4)
            rec["AB長度絕對誤差_原圖px"] = round(abs(err) * scale, 4)
            eA, eB = euclidean(endA, gtA), euclidean(endB, gtB)
            rec["幾何端點誤差_A側px"] = round(eA, 2)
            rec["幾何端點誤差_B側px"] = round(eB, 2)
            same_dir = (euclidean(endA, gtA) + euclidean(endB, gtB)
                        <= euclidean(endA, gtB) + euclidean(endB, gtA))
            rec["A端判定與GT一致"] = "是" if same_dir else "⚠️否"
            if not same_dir:
                flags.append("切端方向判反")
            if (min(eA, eB) > ENDPOINT_ERROR_PX
                    and abs(err) < ENDPOINT_ERROR_PX):
                flags.append("端點誤差抵銷(長度假準)")

        rec["QC備註"] = "、".join(flags)
        rec["建議人工複查"] = "⚠️是" if flags else "否"
        rows.append(rec)

        if SAVE_VISUALIZATION:
            info = []
            mv = b2_map.get(fname)
            if mv is not None and pd.notna(mv):
                info.append(f"mAP50-95: {float(mv):.3f}")
            if "AB長度絕對誤差_原圖px" in rec:
                info.append(f"pred {rec['AB長度_mask幾何_原圖px']:.1f} / "
                            f"GT {rec['真實AB像素長度_原圖px']:.1f} px (orig)")
                info.append(f"|dAB|: {rec['AB長度絕對誤差_原圖px']:.1f} px (orig)")
            others = [p for i, p in enumerate(polygons) if i != chosen_idx]
            draw(img, poly, others, endA, endB,
                 Path(VIS_DIR) / f"{img_path.stem}_mask.jpg",
                 gt_pts=(gtA, gtB) if gtA is not None else None,
                 width_pts=(cw["p1"], cw["p2"]) if cw is not None else None,
                 info=info)

    df = pd.DataFrame(rows)
    per_img = build_per_image(df)
    summary = build_summary(df)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="逐顆牙對照", index=False)
        if not per_img.empty:
            per_img.to_excel(w, sheet_name="逐張長度誤差", index=False)
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

    n_fail = int((df["mask狀態"] != "✅").sum())
    if n_fail:
        print(f"❌ 未成功：{n_fail} 張(見 mask狀態 欄)")
    print(f"⚠️  建議人工複查：{int((df['建議人工複查'] == '⚠️是').sum())} 張")
    if REMOVE_FRAGMENTS and "剔除碎塊數" in df.columns:
        print(f"✂️  選中的 mask 帶碎塊(已剔除)：{int((df['剔除碎塊數'].fillna(0) > 0).sum())} 張")

    if "兩法選取一致" in df.columns:
        n_diff = int((df["兩法選取一致"] == "⚠️否").sum())
        n_cmp = int(df["兩法選取一致"].isin(["是", "⚠️否"]).sum())
        if n_cmp:
            print(f"🔀 center 與 gt_iou 挑到不同 mask：{n_diff}/{n_cmp} 張"
                  f"(= 無 GT 上線時的預期認牙錯誤率)")
    if "A端判定與GT一致" in df.columns:
        n_rev = int((df["A端判定與GT一致"] == "⚠️否").sum())
        n_chk = int(df["A端判定與GT一致"].notna().sum())
        print(f"↕️  切端方向判反：{n_rev}/{n_chk} 張(seg 格式 GT 下幾乎恆為 0，參考價值低)")
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

    if not per_img.empty and TOP_N_WORST:
        show = ["圖片檔名", "上下顎", "AB長度_mask幾何_原圖px", "真實AB像素長度_原圖px",
                "AB長度誤差_預測減GT_原圖px", "相對誤差%"]
        show = [c for c in show if c in per_img.columns]
        print(f"\n=== 長度誤差最大的 {min(TOP_N_WORST, len(per_img))} 張(原圖 px，全表見 sheet「逐張長度誤差」) ===")
        print(per_img[show].head(TOP_N_WORST).rename(columns={
            "AB長度_mask幾何_原圖px": "預測", "真實AB像素長度_原圖px": "GT",
            "AB長度誤差_預測減GT_原圖px": "預測-GT"}).to_string(index=False))

    if not summary.empty:
        print("\n=== mask幾何 vs GT 長度 ===")
        print(summary.to_string(index=False))
        print("看「誤差標準差」；bias 由下游 C1 吸收。以「原圖」列為準。")
    elif gt_dir:
        print("\n(沒有任何一張算得出 GT 長度——檢查 GT_LABEL_DIR 的檔名是否跟影像對得上)")


if __name__ == "__main__":
    main()
