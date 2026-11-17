#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py — GUI 用的單張推論流程
===============================================================
  輸入     整張根尖 X 光
  A3/A4   牙齒偵測 → 使用者點選目標牙 → 依偵測框裁切(padding 1%)
  前處理   top-hat(主，參數同 A1_tophat_before_annotation.py)，做在裁切圖上
  A5      上下補 10% 黑邊 → letterbox 640x640
  B2/B3   YOLO11n-seg 推論(conf 0.15)、剔除碎塊、center 挑牙
  B3      PCA 長軸極值點距離 = AB 長度
  B7      長邊正規化：X = 像素長度 × 1200 ÷「整張 X 光」長邊
  C       GPR(同 C1_pxToMm_GPR.m：SE kernel + linear basis) → mm ± 預測區間，再 −0.5 mm

⚠️ 前處理與 letterbox 參數必須跟「訓練 YOLO 與 GPR 的資料」產生時一致。
⚠️ 搜尋「TODO」找出需要你們確認的地方。
"""
from __future__ import annotations

import math
import re
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# ==================== 設定區 ====================
DET_WEIGHTS = HERE / "detector.pt"        # A3 牙齒偵測器(= A4 的 train_crop_yolov11_run/weights_ready.pt)
SEG_WEIGHTS = HERE / "weights_ready.pt"   # top-hat 版 YOLO11n-seg 權重(沿用 pipeline_v2)
GPR_MODEL = HERE / "gpr_model.json"       # C1_export_final_gpr.m(首選)或 fit_gpr.py 產生

# ---- A3/A4 偵測與裁切(值抄自 A4_crop_teeth_yolov11.py) ----
DET_IMG_SIZE = 640
DET_CONF = 0.15
CROP_PADDING_RATIO = 0.01

# ---- 前處理(做在 A4 裁切後的單顆牙圖上、letterbox 之前) ----
PREPROCESS = "tophat"                     # "tophat"(論文主設定) | "clahe"(對照) | "none"
# top-hat：參數照抄 A1_tophat_before_annotation.py(改那邊就要同步這裡)
TOPHAT_MODE = "contrast"                  # "contrast" | "tophat" | "th_clahe"
TOPHAT_KSIZE = 31
TOPHAT_SHAPE = "ellipse"                  # "ellipse" | "rect" | "cross"
TOPHAT_STRENGTH = 1.0
# 前處理做在 letterbox 之前還是之後(K=31 是像素數，兩者結果不同)：
#   "before"：對原始整張圖做，再 A5 letterbox(A1t 檔頭「跟 A1 同位置」= 標註前、A5 前)
#   "after" ：先 A5 letterbox 成 640，再對 640 圖做
PREPROCESS_STAGE = "before"               # 已確認：top-hat 在 letterbox 之前
# CLAHE(值抄自 A1_CLAHE_before_annotation.py)
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)
BLACK_BORDER_THRESHOLD = 5

# ---- A5 letterbox ----
PAD_TB_RATIO = 0.10
OUT_SIZE = 640
PAD_VALUE = 0

# ---- B2/B3 推論 ----
IMG_SIZE = 640
SEG_CONF = 0.15
REMOVE_FRAGMENTS = True

# ---- B3 幾何 ----
ORIENT_MODE = "arch"
CROWN_END_FRAC = 0.30
OPEN_KERNEL_FRAC = 0.04

# ---- letterbox px → 原圖 px(沿用 pipeline_v2 / check_consistency 選出的值) ----
LENGTH_SCALE_MODE = "b3_padded"

# ---- B7 / C ----
NORM_REF = 1200                           # 要不要做長邊正規化由 gpr_model.json 的 normalized 決定
OFFSET_MM = 0.5
PI_Z = 1.96                               # 95% 預測區間

TOOTH_POSITIONS = [11, 12, 21, 22, 31, 32, 41, 42]
# =================================================


# ============================================================
# 讀圖：cv2.imread 在 Windows 遇到中文路徑會回傳 None，改用 imdecode
# ============================================================
def imread_u(path, flags=cv2.IMREAD_COLOR):
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(buf, flags) if buf.size else None


# ============================================================
# A1：CLAHE(與 A1_CLAHE_before_annotation.py 的 enhance_image 相同)
# ============================================================
def enhance_image(img, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    if img is None:
        return img
    is_color = img.ndim == 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img.copy()
    ys, xs = np.where(gray > BLACK_BORDER_THRESHOLD)
    if len(ys) == 0:
        return img
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    out = gray.copy()
    out[y0:y1, x0:x1] = clahe.apply(gray[y0:y1, x0:x1])
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) if is_color else out


# ============================================================
# A5：補邊 + letterbox(與 A5_pad_letterbox_640.py 相同)
# ============================================================
def compute_geom(W, H, ratio=PAD_TB_RATIO, out_size=OUT_SIZE):
    p = int(round(H * ratio))
    Hp = H + 2 * p
    s = out_size / max(W, Hp)
    nw, nh = max(1, int(round(W * s))), max(1, int(round(Hp * s)))
    return {"pad_tb_px": p, "scale_x": nw / W, "scale_y": nh / Hp,
            "new_w": nw, "new_h": nh,
            "pad_left": (out_size - nw) // 2, "pad_top": (out_size - nh) // 2}


def pad_and_letterbox(img, g, out_size=OUT_SIZE, pad_value=PAD_VALUE):
    p = g["pad_tb_px"]
    ch = img.shape[2] if img.ndim == 3 else 1
    padded = cv2.copyMakeBorder(img, p, p, 0, 0, cv2.BORDER_CONSTANT, value=[pad_value] * ch)
    shrink = g["new_w"] < padded.shape[1]
    resized = cv2.resize(padded, (g["new_w"], g["new_h"]),
                         interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    canvas = np.full((out_size, out_size) + ((ch,) if img.ndim == 3 else ()), pad_value, img.dtype)
    y0, x0 = g["pad_top"], g["pad_left"]
    canvas[y0:y0 + g["new_h"], x0:x0 + g["new_w"]] = resized
    return canvas


def lb_to_orig(pts, g):
    """640 → 原圖像素(A5 檔頭公式)。"""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x = (pts[:, 0] - g["pad_left"]) / g["scale_x"]
    y = (pts[:, 1] - g["pad_top"]) / g["scale_y"] - g["pad_tb_px"]
    return np.column_stack([x, y])


# ============================================================
# B3：碎塊、挑牙、幾何(函式與 B3_mask_assist_and_compare_v3.py 相同)
# ============================================================
def largest_part(polygon, shape):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if n <= 1:
        return np.asarray(polygon, dtype=np.float64), 0
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cs, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float64), n - 2


def pick_by_center(polygons, shape):
    H, W = shape[:2]
    c = np.array([W / 2.0, H / 2.0])
    d = [float(np.linalg.norm(np.asarray(p, dtype=np.float64).mean(axis=0) - c)) for p in polygons]
    return int(np.argmin(d))


def mask_axis(polygon):
    pts = np.asarray(polygon, dtype=np.float64)
    centroid = pts.mean(axis=0)
    _, s, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    return centroid, vt[0], float(s[0] / max(s.sum(), 1e-9))


def axis_extremes(polygon, centroid, axis):
    t = (np.asarray(polygon, dtype=np.float64) - centroid) @ axis
    return centroid + axis * t.min(), centroid + axis * t.max()


def clean_mask(polygon, shape, length_px):
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(polygon)).astype(np.int32)], 1)
    k = max(3, int(round(length_px * OPEN_KERNEL_FRAC)) | 1)
    m2 = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m2, connectivity=8)
    if n <= 1:
        return m
    return (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def end_mean_width(mask, centroid, axis, end, frac=CROWN_END_FRAC):
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return float("nan")
    pts = np.column_stack([xs, ys]).astype(np.float64)
    t = (pts - centroid) @ axis
    sign = 1.0 if float((np.asarray(end) - centroid) @ axis) <= 0 else -1.0
    d = (t - (t.min() if sign > 0 else t.max())) * sign
    total = float(d.max())
    if total <= 0:
        return float("nan")
    bins = np.floor(d).astype(int)
    uniq, cnt = np.unique(bins, return_counts=True)
    sel = uniq <= frac * total
    return float(cnt[sel].mean()) if sel.any() else float("nan")


def orient(mask, centroid, axis, e1, e2, position):
    """回傳 (A=切端, B=根尖, 依據)。上顎冠朝下、下顎冠朝上；沒牙位用寬度法。"""
    q = position // 10 if position else None
    if ORIENT_MODE == "arch" and q in (1, 2, 3, 4):
        e1_lower = e1[1] >= e2[1]
        a_is_1 = e1_lower if q in (1, 2) else not e1_lower
        basis = "牙位"
    else:
        a_is_1 = end_mean_width(mask, centroid, axis, e1) >= end_mean_width(mask, centroid, axis, e2)
        basis = "寬度法"
    return (e1, e2, basis) if a_is_1 else (e2, e1, basis)


def geometry_AB(poly, shape, position):
    centroid, axis, ratio = mask_axis(poly)
    e1, e2 = axis_extremes(poly, centroid, axis)
    length = math.dist(e1, e2)
    A, B, basis = orient(clean_mask(poly, shape, length), centroid, axis, e1, e2, position)
    return A, B, length, {"axis_ratio": ratio, "basis": basis}


def lb_length_to_orig(len_lb, A, B, g, W0, H0, mode=LENGTH_SCALE_MODE):
    if mode == "b3_original":
        return len_lb * max(W0, H0) / float(OUT_SIZE)
    if mode == "b3_padded":
        return len_lb
    if mode == "exact":
        a, b = lb_to_orig([A, B], g)
        return math.dist(a, b)
    raise ValueError(f"LENGTH_SCALE_MODE 只能是 b3_original / b3_padded / exact，目前是 {mode!r}")



# ============================================================
# top-hat 前處理
# ============================================================
_SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}


def _signal_bbox(gray):
    ys, xs = np.where(gray > BLACK_BORDER_THRESHOLD)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _tophat_contrast(roi, se, strength):
    th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, se).astype(np.float32)
    bh = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    return np.clip(roi.astype(np.float32) + strength * (th - bh), 0, 255).astype(np.uint8)


def tophat_image(img, mode=None, k=None, shape=None, strength=None):
    """與 A1_tophat_before_annotation.py 的 enhance_image 相同(只在非黑邊 ROI 內處理)。"""
    if img is None:
        return img
    mode = mode or TOPHAT_MODE
    strength = TOPHAT_STRENGTH if strength is None else strength
    is_color = img.ndim == 3
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img.copy()
    bb = _signal_bbox(gray)
    if bb is None:
        return img
    y0, y1, x0, x1 = bb
    roi = gray[y0:y1, x0:x1]
    se = cv2.getStructuringElement(_SHAPES[shape or TOPHAT_SHAPE], ((k or TOPHAT_KSIZE) | 1,) * 2)
    if mode == "contrast":
        out = _tophat_contrast(roi, se, strength)
    elif mode == "tophat":
        out = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, se)
    elif mode == "th_clahe":
        out = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE).apply(
            _tophat_contrast(roi, se, strength))
    else:
        raise ValueError(f"TOPHAT_MODE 只能是 contrast / tophat / th_clahe，目前是 {mode!r}")
    g2 = gray.copy()
    g2[y0:y1, x0:x1] = out
    return cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR) if is_color else g2


def preprocess(img, mode=None):
    mode = mode or PREPROCESS
    if mode == "tophat":
        return tophat_image(img)
    if mode == "clahe":
        return enhance_image(img)
    if mode == "none":
        return img.copy()
    raise ValueError(f"PREPROCESS 只能是 tophat / clahe / none，目前是 {mode!r}")


def lb_content_to_orig(lb, g, W0, H0):
    """把 640 letterbox 圖裡的影像區塊切回原圖尺寸(只用來顯示 after 模式的前處理結果)。"""
    y0, x0 = g["pad_top"], g["pad_left"]
    c = lb[y0:y0 + g["new_h"], x0:x0 + g["new_w"]]
    p = int(round(g["pad_tb_px"] * g["scale_y"]))
    c = c[p:c.shape[0] - p] if p else c
    return cv2.resize(c, (W0, H0), interpolation=cv2.INTER_LINEAR)


PREPROCESS_LABEL = {"tophat": "Top-hat 形態學增強", "clahe": "CLAHE 對比增強", "none": "不處理"}


# ============================================================
# GPR(gpr_model.json；數學見 gpr_core.py，與 MATLAB fitrgp 相同)
# ============================================================
class GPR:
    def __init__(self, path=GPR_MODEL):
        import json
        import gpr_core
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path.name}：MATLAB 跑 C1_export_final_gpr.m，或 python fit_gpr.py")
        d = self.d = json.loads(path.read_text(encoding="utf-8"))
        self.use_age = bool(d["use_age"])
        self.normalized = bool(d.get("normalized", True))
        self.m = gpr_core.Predictor(d["Xn"], d["y"], d["kernel"], d["basis"], d["theta"],
                                    d["sigma"], d["beta"], d["mu"], d["sg"])
        st = d.get("selftest") or []
        for row in (st if isinstance(st, list) else [st]):
            mu, sd = self.m.predict([np.ravel(row["x"])])
            if abs(mu[0] - row["mean"]) > 1e-4 or abs(sd[0] - row["sd"]) > 1e-4:
                raise RuntimeError(f"GPR 自我驗證失敗：Python ({mu[0]:.5f}, {sd[0]:.5f}) ≠ "
                                   f"{d.get('trained_by')} ({row['mean']:.5f}, {row['sd']:.5f})")

    def feature(self, len_px, W0, H0):
        """訓練時的 X：長邊正規化版 = px × 1200 ÷ 整張長邊；否則直接用 px。"""
        return len_px * NORM_REF / max(W0, H0) if self.normalized else len_px

    def predict(self, feat, age=None):
        """回傳 (平均 mm, 預測標準差 mm)。"""
        x = [feat]
        if self.use_age:
            if age is None:
                raise ValueError("GPR 模型有用年齡，請輸入年齡")
            x.append(float(age))
        mu, sd = self.m.predict([x])
        return float(mu[0]), float(sd[0])

    def range_warnings(self, feat, age=None):
        out = []
        lo, hi = self.d["feat_range"]
        if not lo <= feat <= hi:
            out.append(f"長度特徵 {feat:.0f} 超出訓練範圍 {lo:.0f}–{hi:.0f}")
        if self.use_age and age is not None and "age_range" in self.d:
            lo, hi = self.d["age_range"]
            if not lo <= age <= hi:
                out.append(f"年齡 {age:g} 超出訓練範圍 {lo:g}–{hi:g} 歲")
        return out

    def info(self):
        d = self.d
        feats = d.get("features")
        return {"kind": "Gaussian Process Regression", "kernel": d.get("kernel"), "basis": d.get("basis"),
                "method": d.get("method"), "trained_by": d.get("trained_by"),
                "features": feats if isinstance(feats, list) else [feats], "n": d.get("n"),
                "cv": d.get("cv", {})}


# ============================================================
# A3/A4：偵測與裁切
# ============================================================
def _np(t):
    return t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t)


def detect_teeth(full_bgr):
    """整張 X 光 → 偵測框 list(正規化 x1,y1,x2,y2 + conf)，依框中心由左到右排序(同 A4)。"""
    res = get_det().predict(full_bgr, imgsz=DET_IMG_SIZE, conf=DET_CONF, save=False, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    xy = _np(res.boxes.xyxyn).reshape(-1, 4)
    cf = _np(res.boxes.conf).reshape(-1)
    order = sorted(range(len(xy)), key=lambda i: xy[i][0] + xy[i][2])
    return [{"box": [float(v) for v in xy[i]], "conf": float(cf[i])} for i in order]


def crop_tooth(full_bgr, box, padding_ratio=CROP_PADDING_RATIO):
    """與 A4 crop_and_save 相同的裁切(含 padding 與邊界 clip)。回傳 (裁切圖, (x1,y1,x2,y2) 像素)。"""
    h, w = full_bgr.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, x2 = x1 - bw * padding_ratio, x2 + bw * padding_ratio
    y1, y2 = y1 - bh * padding_ratio, y2 + bh * padding_ratio
    x1, y1 = max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1))
    x2, y2 = max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))
    px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    if px2 <= px1 or py2 <= py1:
        raise ValueError("裁切框座標異常")
    return full_bgr[py1:py2, px1:px2].copy(), (px1, py1, px2, py2)


# ============================================================
# 模型載入(只載一次)
# ============================================================
_det = None
_seg = None
_gpr = None


def get_det():
    global _det
    if _det is None:
        from ultralytics import YOLO
        if not Path(DET_WEIGHTS).exists():
            raise FileNotFoundError(f"找不到牙齒偵測器權重 {DET_WEIGHTS}")
        _det = YOLO(str(DET_WEIGHTS))
    return _det


def get_seg():
    global _seg
    if _seg is None:
        from ultralytics import YOLO
        if not Path(SEG_WEIGHTS).exists():
            raise FileNotFoundError(f"找不到 seg 權重 {SEG_WEIGHTS}")
        _seg = YOLO(str(SEG_WEIGHTS))
    return _seg


def get_gpr():
    global _gpr
    if _gpr is None:
        _gpr = GPR()
    return _gpr


def device_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU"


# ============================================================
# 主流程
# ============================================================
def measure(bgr, position=None, scale_mode=LENGTH_SCALE_MODE, timings=None, stage=None, normalized=True,
            full_wh=None):
    """單顆牙裁切圖 → 長度特徵(不經迴歸)。check_consistency.py 也用這個。
    full_wh：整張 X 光的 (寬, 高)，長邊正規化的分母(同 B7)；None 時才退回用裁切圖尺寸。"""
    if bgr is None:
        raise ValueError("影像讀取失敗(檔案損毀或格式不支援)")
    t = timings if timings is not None else {}
    H0, W0 = bgr.shape[:2]
    stage = stage or PREPROCESS_STAGE

    t0 = time.perf_counter()
    g = compute_geom(W0, H0)
    if stage == "before":
        enh = preprocess(bgr)
        lb = pad_and_letterbox(enh, g)
    elif stage == "after":
        lb = preprocess(pad_and_letterbox(bgr, g))
        enh = lb_content_to_orig(lb, g, W0, H0)
    else:
        raise ValueError(f"PREPROCESS_STAGE 只能是 before / after，目前是 {stage!r}")
    t["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    res = get_seg().predict(lb, imgsz=IMG_SIZE, conf=SEG_CONF, save=False, verbose=False)[0]
    raw, confs = [], []
    if res.masks is not None:
        all_conf = res.boxes.conf.tolist() if getattr(res, "boxes", None) is not None else []
        for i, p in enumerate(res.masks.xy):
            if p is not None and len(p) >= 3:
                raw.append(np.asarray(p, dtype=np.float64))
                confs.append(float(all_conf[i]) if i < len(all_conf) else float("nan"))
    t["segment"] = time.perf_counter() - t0
    if not raw:
        return {"ok": False, "enh": enh, "msg": "沒有偵測到牙齒"}

    t0 = time.perf_counter()
    polys = [largest_part(p, lb.shape)[0] for p in raw] if REMOVE_FRAGMENTS else raw
    idx = pick_by_center(polys, lb.shape)
    A, B, len_lb, geo = geometry_AB(polys[idx], lb.shape, position)
    len_orig = lb_length_to_orig(len_lb, A, B, g, W0, H0, scale_mode)
    denom = max(full_wh) if full_wh else max(W0, H0)
    feat = len_orig * NORM_REF / denom if normalized else len_orig
    t["measure"] = time.perf_counter() - t0
    return {"ok": True, "enh": enh, "g": g, "polys": polys, "confs": confs, "idx": idx,
            "A": A, "B": B, "len_lb": len_lb, "len_orig": len_orig, "feat": feat,
            "W0": W0, "H0": H0, **geo}


def draw_overlay(m):
    """選中的牙：半透明填色 + 輪廓 + 長軸 A–B；其他候選：細灰線。"""
    vis = m["enh"].copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if not m["ok"]:
        return vis
    g = m["g"]
    t = max(2, int(round(max(m["W0"], m["H0"]) / 350)))
    cyan, amber = (230, 200, 40), (40, 180, 255)          # BGR
    for i, p in enumerate(m["polys"]):
        if i == m["idx"]:
            continue
        pts = np.round(lb_to_orig(p, g)).astype(np.int32)
        cv2.polylines(vis, [pts], True, (150, 150, 150), max(1, t // 2), cv2.LINE_AA)
    pts = np.round(lb_to_orig(m["polys"][m["idx"]], g)).astype(np.int32)
    fill = vis.copy()
    cv2.fillPoly(fill, [pts], cyan)
    vis = cv2.addWeighted(vis, 0.72, fill, 0.28, 0)
    cv2.polylines(vis, [pts], True, cyan, t, cv2.LINE_AA)
    a, b = np.round(lb_to_orig([m["A"], m["B"]], g)).astype(int)
    cv2.line(vis, tuple(a), tuple(b), amber, t, cv2.LINE_AA)
    for p, tag in ((a, "A"), (b, "B")):
        cv2.circle(vis, tuple(p), 3 * t, amber, -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(p), 3 * t, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
        cv2.putText(vis, tag, tuple(p + np.array([4 * t, -3 * t])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * t, (255, 255, 255), max(1, t // 2 + 1), cv2.LINE_AA)
    return vis


def run(full_bgr, box, age=None, position=None, det_conf=None):
    """GUI 入口。full_bgr：整張 X 光；box：使用者選的偵測框(正規化 x1,y1,x2,y2)。
    回傳 dict(影像為 BGR numpy；數值已四捨五入)。"""
    timings = {}
    position = int(position) if position not in (None, "", "0") else None
    gpr = get_gpr()
    Hf, Wf = full_bgr.shape[:2]

    t0 = time.perf_counter()
    crop, cpx = crop_tooth(full_bgr, box)
    timings["crop"] = time.perf_counter() - t0

    m = measure(crop, position, timings=timings, normalized=gpr.normalized, full_wh=(Wf, Hf))
    if not m["ok"]:
        return {"ok": False, "msg": m["msg"], "original": crop, "enhanced": m["enh"],
                "overlay": m["enh"], "timings": {k: round(v * 1000, 1) for k, v in timings.items()}}

    t0 = time.perf_counter()
    mu, sd = gpr.predict(m["feat"], age)
    timings["regress"] = time.perf_counter() - t0
    pred = mu - OFFSET_MM

    notes = gpr.range_warnings(m["feat"], age)
    if len(m["polys"]) > 1:
        notes.append(f"裁切圖內分割出 {len(m['polys'])} 個區塊，已取最靠近中心者，請確認橘框是目標牙")
    if m["basis"] != "牙位":
        notes.append("未指定牙位，切端/根尖方向以寬度法判斷")

    return {
        "ok": True, "original": crop, "enhanced": m["enh"], "overlay": draw_overlay(m),
        "pred_mm": round(pred, 2), "raw_mm": round(mu, 2), "sd_mm": round(sd, 3),
        "pi_low": round(pred - PI_Z * sd, 2), "pi_high": round(pred + PI_Z * sd, 2),
        "len_px": round(m["len_orig"], 1), "feat": round(m["feat"], 2),
        "conf": round(m["confs"][m["idx"]], 3) if m["confs"] else None,
        "det_conf": None if det_conf is None else round(det_conf, 3),
        "n_masks": len(m["polys"]), "size": f"{Wf}×{Hf}", "crop_size": f"{m['W0']}×{m['H0']}",
        "crop_box": list(cpx), "basis": m["basis"], "notes": notes,
        "timings": {k: round(v * 1000, 1) for k, v in timings.items()},
    }


# ============================================================
# 檔名 key(check_consistency.py 用)
# ============================================================
_LEADING_NUM = re.compile(r"^\s*0*(\d+)")


def name_key(name):
    stem = Path(str(name)).stem
    m = _LEADING_NUM.match(stem)
    return str(int(m.group(1))) if m else stem.lower()
