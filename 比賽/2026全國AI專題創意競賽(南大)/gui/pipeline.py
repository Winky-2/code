#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline.py — GUI 用的單張推論流程（不訓練、不讀 GT）
===============================================================
把你們的離線腳本串成一個函式 run()，順序與參數都照原腳本：

  A1  enhance_image      CLAHE(clip 1.5, 8x8, 排除黑邊)
  A5  pad + letterbox    上下補 10% 黑邊 → 640x640
  B2/B3 YOLO-seg 推論    conf 0.15，剔除碎塊(最大連通區)
  B3  挑牙 + 幾何        center 挑牙 → PCA 長軸極值點距離 = AB 長度
  B7  長邊正規化         X = 原圖px × 1200 ÷ 整張長邊
  C1  迴歸 + offset      (X, 年齡) → mm，再 −0.5 mm；|誤差|≤1 mm 為理想

⚠️ 所有參數都必須跟「訓練迴歸模型用的那份 Excel」產生時一致。
   改完後一定要跑 check_consistency.py 驗證(見 README)。
⚠️ 搜尋「TODO」找出需要你們確認的地方。
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent

# ==================== 設定區 ====================
SEG_WEIGHTS = HERE / "best.pt"          # TODO 1：換成論文用的 seg 權重
REG_MODEL = HERE / "reg_model.json"     # fit_regressor.py 或 C1_export_final_ann.m 產生

# ---- A1 CLAHE(值抄自 A1_CLAHE_before_annotation.py) ----
USE_CLAHE = True                        # TODO 2：B2/B3 推論的 test 圖若沒做 CLAHE，改 False
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)
BLACK_BORDER_THRESHOLD = 5

# ---- A5 letterbox(值抄自 A5_pad_letterbox_640.py) ----
PAD_TB_RATIO = 0.10                     # TODO 3：test 集跑 A5 時用的值
OUT_SIZE = 640
PAD_VALUE = 0

# ---- B2/B3 推論 ----
IMG_SIZE = 640
SEG_CONF = 0.15
REMOVE_FRAGMENTS = True

# ---- B3 幾何(只影響方向與畫圖；AB 長度 = 兩極值點距離，與方向無關) ----
ORIENT_MODE = "arch"
CROWN_END_FRAC = 0.30
OPEN_KERNEL_FRAC = 0.04

# ---- letterbox px → 原圖 px 的換算方式 ----
# TODO 4：用 check_consistency.py 決定，選跟訓練 Excel「像素長度」對得上的那個
#   "b3_original"：B3 SCALE_SOURCE=original，ORIGINAL_IMAGE_DIR 指向未 letterbox 原圖
#                 → scale = 原圖長邊 ÷ 640(B3 原寫法，沒扣掉 A5 的上下補邊)
#   "b3_padded"  ：ORIGINAL_IMAGE_DIR 指向 A5 輸出(檔內預設值) → scale = 1
#   "exact"      ：照 A5 檔頭公式把端點逆換回原圖座標(數學上最正確)
LENGTH_SCALE_MODE = "b3_original"

# ---- B7 正規化 ----
NORM_REF = 1200

# ---- C1 臨床設定 ----
OFFSET_MM = 0.5
IDEAL_TOLERANCE_MM = 1.0

TOOTH_POSITIONS = [11, 12, 21, 22, 31, 32, 41, 42]
# =================================================


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
# C1：迴歸(讀 reg_model.json)
# ============================================================
class Regressor:
    """kind="linear"：fit_regressor.py 產生
       kind="matlab_ann"：C1_export_final_ann.m 產生(fitnet 前向傳遞，載入時自我驗證)"""

    def __init__(self, path=REG_MODEL):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path.name}：先跑 fit_regressor.py(或 MATLAB 的 C1_export_final_ann.m)")
        self.d = json.loads(path.read_text(encoding="utf-8"))
        self.kind = self.d["kind"]
        self.use_age = bool(self.d["use_age"])
        if self.kind == "matlab_ann":
            self._selftest()

    def _vec(self, feat, age):
        x = [feat]
        if self.use_age:
            if age is None:
                raise ValueError("這個迴歸模型有用年齡，請輸入年齡")
            x.append(float(age))
        return np.asarray(x, dtype=np.float64)

    def _ann(self, x):
        d = self.d
        v = lambda k, src=d: np.atleast_1d(np.asarray(src[k], dtype=np.float64)).reshape(-1)
        x = np.atleast_1d(np.asarray(x, dtype=np.float64)).reshape(-1)
        xn = (x - v("mu")) / v("sg")
        pi = d["in_ps"]
        xi = (xn - v("xoffset", pi)) * v("gain", pi) + float(np.ravel(pi["ymin"])[0])
        b1 = v("b1")
        IW = np.asarray(d["IW"], dtype=np.float64).reshape(len(b1), -1)      # MATLAB 1 欄時會被攤平
        h = np.tanh(IW @ xi + b1)                                            # tansig == tanh
        o = float(np.asarray(d["LW"]).reshape(-1) @ h + float(np.asarray(d["b2"]).reshape(-1)[0]))
        po = d["out_ps"]
        return (o - po["ymin"]) / float(np.ravel(po["gain"])[0]) + float(np.ravel(po["xoffset"])[0])

    def _selftest(self):
        for row in self.d.get("selftest", []):
            got = self._ann(np.asarray(row["x"], dtype=np.float64))
            if abs(got - row["y"]) > 1e-4:
                raise RuntimeError(f"ANN 自我驗證失敗：Python {got:.6f} ≠ MATLAB {row['y']:.6f}")

    def predict(self, feat, age=None):
        x = self._vec(feat, age)
        if self.kind == "linear":
            return float(self.d["intercept"] + np.dot(self.d["coef"], x))
        if self.kind == "matlab_ann":
            return float(self._ann(x))
        raise ValueError(f"未知的模型種類 {self.kind}")

    def describe(self):
        feats = "長邊正規化像素長度" + (" + 年齡" if self.use_age else "")
        return f"{self.kind}（{feats}，n={self.d.get('n', '?')}）"


def judge(pred_off_mm, true_mm):
    if true_mm is None or not (true_mm > 0):
        return "—（未輸入實際充填長度）", None
    err = pred_off_mm - true_mm
    if abs(err) <= IDEAL_TOLERANCE_MM:
        return f"🟢 理想（誤差 {err:+.2f} mm，容許 ±{IDEAL_TOLERANCE_MM} mm）", err
    if err > 0:
        return f"🔴 過長（{err:+.2f} mm）", err
    return f"🟡 過短（{err:+.2f} mm）", err


# ============================================================
# 模型載入(只載一次)
# ============================================================
_seg = None
_reg = None


def get_seg():
    global _seg
    if _seg is None:
        from ultralytics import YOLO
        if not Path(SEG_WEIGHTS).exists():
            raise FileNotFoundError(f"找不到 seg 權重 {SEG_WEIGHTS}")
        _seg = YOLO(str(SEG_WEIGHTS))
    return _seg


def get_reg():
    global _reg
    if _reg is None:
        _reg = Regressor()
    return _reg


# ============================================================
# 主流程
# ============================================================
def measure(bgr, position=None, scale_mode=LENGTH_SCALE_MODE):
    """影像 → 長度特徵(不經迴歸)。check_consistency.py 也用這個。"""
    H0, W0 = bgr.shape[:2]
    enh = enhance_image(bgr) if USE_CLAHE else bgr.copy()
    g = compute_geom(W0, H0)
    lb = pad_and_letterbox(enh, g)

    res = get_seg().predict(lb, imgsz=IMG_SIZE, conf=SEG_CONF, save=False, verbose=False)[0]
    raw = [np.asarray(p, dtype=np.float64)
           for p in (res.masks.xy if res.masks is not None else []) if p is not None and len(p) >= 3]
    if not raw:
        return {"ok": False, "enh": enh, "msg": "⚠️ 沒有偵測到牙齒"}
    polys = [largest_part(p, lb.shape)[0] for p in raw] if REMOVE_FRAGMENTS else raw
    idx = pick_by_center(polys, lb.shape)

    A, B, len_lb, geo = geometry_AB(polys[idx], lb.shape, position)
    len_orig = lb_length_to_orig(len_lb, A, B, g, W0, H0, scale_mode)
    feat = len_orig * NORM_REF / max(W0, H0)
    return {"ok": True, "enh": enh, "g": g, "polys": polys, "idx": idx, "A": A, "B": B,
            "len_lb": len_lb, "len_orig": len_orig, "feat": feat, "W0": W0, "H0": H0, **geo}


def draw_result(m):
    vis = m["enh"].copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if not m["ok"]:
        return vis
    g = m["g"]
    t = max(2, int(round(max(m["W0"], m["H0"]) / 400)))
    for i, p in enumerate(m["polys"]):
        pts = np.round(lb_to_orig(p, g)).astype(np.int32)
        if i == m["idx"]:
            fill = vis.copy()
            cv2.fillPoly(fill, [pts], (0, 200, 255))
            vis = cv2.addWeighted(vis, 0.75, fill, 0.25, 0)
            cv2.polylines(vis, [pts], True, (0, 200, 255), t)
        else:
            cv2.polylines(vis, [pts], True, (130, 130, 130), max(1, t // 2))
    a, b = np.round(lb_to_orig([m["A"], m["B"]], g)).astype(int)
    cv2.line(vis, tuple(a), tuple(b), (255, 160, 0), t)
    for p, tag in ((a, "A"), (b, "B")):
        cv2.circle(vis, tuple(p), 3 * t, (255, 160, 0), -1)
        cv2.putText(vis, tag, tuple(p + np.array([4 * t, -4 * t])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t, (255, 160, 0), t, cv2.LINE_AA)
    return vis


def run(img_rgb, age=None, position=None, true_mm=None):
    """GUI 入口。img_rgb：Gradio 給的 RGB(或灰階) numpy。
    回傳 (原圖, CLAHE, 結果疊圖, 校正後預測mm, 判讀, 明細dict)，影像皆為 RGB。"""
    if img_rgb.ndim == 2:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(img_rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    position = int(position) if position not in (None, "", "不指定") else None

    m = measure(bgr, position)
    to_rgb = lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    if not m["ok"]:
        return to_rgb(bgr), to_rgb(m["enh"]), to_rgb(m["enh"]), None, m["msg"], {}

    reg = get_reg()
    pred = reg.predict(m["feat"], age)
    pred_off = pred - OFFSET_MM
    status, err = judge(pred_off, true_mm)

    info = {
        "影像尺寸": f"{m['W0']}x{m['H0']}",
        "偵測到 mask 數": len(m["polys"]),
        "AB 長度 (640 px)": round(m["len_lb"], 2),
        "AB 長度 (原圖 px)": round(m["len_orig"], 2),
        "長邊正規化特徵": round(m["feat"], 3),
        "年齡": age,
        "方向依據": m["basis"],
        "迴歸模型": reg.describe(),
        "模型預測 (mm)": round(pred, 2),
        f"校正後預測 (−{OFFSET_MM} mm)": round(pred_off, 2),
    }
    if err is not None:
        info["誤差 (mm)"] = round(err, 2)
    if len(m["polys"]) > 1:
        info["提醒"] = "偵測到多顆牙，已選最靠近影像中心的那顆（橘色），請確認是目標牙"
    return to_rgb(bgr), to_rgb(m["enh"]), to_rgb(draw_result(m)), round(pred_off, 2), status, info


# ============================================================
# 醫師表(批次用；找標頭的邏輯同 B4)
# ============================================================
_LEADING_NUM = re.compile(r"^\s*0*(\d+)")


def name_key(name):
    stem = Path(str(name)).stem
    m = _LEADING_NUM.match(stem)
    return str(int(m.group(1))) if m else stem.lower()


def read_doctor_table(path):
    import pandas as pd
    path = str(path)
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raw = pd.read_excel(path, header=None, nrows=15)
        hdr = 0
        for i in range(len(raw)):
            cells = [str(c) for c in raw.iloc[i].tolist() if pd.notna(c)]
            if any("檔名" in c for c in cells):
                hdr = i
                break
        df = pd.read_excel(path, header=hdr)
    col = lambda kw, ex=(): next((c for c in df.columns
                                  if kw in str(c) and not any(e in str(c) for e in ex)), None)
    ncol, ycol = col("檔名"), col("mm", ("像素", "px"))
    acol, pcol = col("年齡"), col("牙位")
    if ncol is None:
        raise KeyError(f"表格找不到「檔名」欄，現有欄位：{list(df.columns)}")
    out = pd.DataFrame({"檔名": df[ncol]})
    out["實際長度_mm"] = pd.to_numeric(df[ycol], errors="coerce") if ycol else np.nan
    out["年齡"] = pd.to_numeric(df[acol], errors="coerce") if acol else np.nan
    out["牙位"] = pd.to_numeric(df[pcol], errors="coerce") if pcol else np.nan
    out = out.dropna(subset=["檔名"])
    out["_key"] = out["檔名"].map(name_key)
    return out.drop_duplicates("_key").set_index("_key")
