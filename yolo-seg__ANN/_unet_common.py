"""
_unet_common.py
===============================================================
UNet 分支（B1u / B2u）共用工具。刻意不在模組層 import torch：
資料收集、切分、letterbox、指標在沒裝 torch 的機器上也能跑。

資料收集與切分邏輯與 B1_train_yolo11-seg.py 一致（同檔名池順序 + seed 42），
讓 UNet 與 YOLO11-seg 用同一組 train/val —— 唯一變因是模型。
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
PAD_VALUE = 114                                   # 與 Ultralytics letterbox 相同的灰邊
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ============================================================
# 資料收集與切分（與 B1 相同）
# ============================================================

def collect_seg_dataset(data_dir):
    pool, seen = [], {}
    root = Path(data_dir)
    for split_name in ("train", "valid", "test", "images"):
        if split_name == "images":
            img_dir, lbl_dir = root / "images", root / "labels"
        else:
            img_dir, lbl_dir = root / split_name / "images", root / split_name / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists() or img_path.stem in seen:
                continue
            seen[img_path.stem] = split_name
            pool.append({"img_id": img_path.stem, "img_path": img_path,
                         "label_path": label_path})
    return pool


def split_train_val(pool, val_ratio, seed):
    """等同 B1：random.seed(seed) 後對 pool 做一次 shuffle。"""
    shuffled = pool[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def check_split_against_yolo(val_items, yolo_val_dir):
    """核對 val 集是否與 YOLO 訓練時實際用的完全相同。"""
    d = Path(yolo_val_dir)
    if not d.exists():
        print(f"ℹ️  找不到 {d}，無法核對切分是否與 YOLO 相同（不影響訓練）")
        return None
    yolo = {p.stem for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    mine = {it["img_id"] for it in val_items}
    if yolo == mine:
        print(f"✅ val 切分與 YOLO 完全相同（{len(mine)} 張）")
        return True
    print(f"❌ val 切分與 YOLO 不同：只在UNet {len(mine - yolo)} 張 / "
          f"只在YOLO {len(yolo - mine)} 張 → 這樣就不是單一變因對照，先查原因")
    return False


# ============================================================
# 影像 / 標註 I/O（np.fromfile 讀寫，Windows 中文路徑也可）
# ============================================================

def imread_rgb(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"讀不到影像：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path, img_bgr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img_bgr)
    if ok:
        buf.tofile(str(path))


def label_to_mask(label_path, H, W):
    """YOLO-seg 多邊形 → 原圖尺寸的二值 mask（所有 instance 聯集）。
    回傳 (mask uint8, 有效多邊形數)。"""
    m = np.zeros((H, W), np.uint8)
    if label_path is None or not Path(label_path).exists():
        return m, 0
    n = 0
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        v = line.split()[1:]
        if len(v) < 6 or len(v) % 2:
            continue
        pts = np.asarray(v, np.float64).reshape(-1, 2) * [W, H]
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        n += 1
    return m, n


# ============================================================
# letterbox（置中補灰邊，與 Ultralytics 同概念）
# ============================================================

def letterbox(img, size, pad_value, interp=cv2.INTER_LINEAR):
    H, W = img.shape[:2]
    r = min(size / H, size / W)
    nh, nw = int(round(H * r)), int(round(W * r))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size) + img.shape[2:], pad_value, dtype=img.dtype)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return out, (top, left, nh, nw)


def unletterbox(arr, meta, H, W):
    """letterbox 空間的機率圖 → 原圖空間（先插值機率，再閾值）。"""
    top, left, nh, nw = meta
    return cv2.resize(arr[top:top + nh, left:left + nw], (W, H),
                      interpolation=cv2.INTER_LINEAR)


def to_input(img_rgb):
    """uint8 HWC RGB → float32 CHW，ImageNet 標準化。"""
    x = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def random_affine(img, mask, rng, degrees, translate, scale):
    """旋轉/平移/縮放，強度對齊 B1（degrees=10, translate=0.05, scale=0.1）。"""
    S = img.shape[0]
    M = cv2.getRotationMatrix2D((S / 2, S / 2), rng.uniform(-degrees, degrees),
                                rng.uniform(1 - scale, 1 + scale))
    M[:, 2] += rng.uniform(-translate, translate, 2) * S
    img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR,
                         borderValue=(PAD_VALUE,) * 3)
    mask = cv2.warpAffine(mask, M, (S, S), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, mask


# ============================================================
# 連通區 / 指標
# ============================================================

def _cc(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, lab, stats


def largest_cc(mask):
    n, lab, stats = _cc(mask)
    if n <= 1:
        return np.zeros(mask.shape, bool), 0
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), n - 1


def best_overlap_cc(mask, gt):
    """oracle：與 GT 重疊最多的連通區（只供診斷，不可當成績）。"""
    n, lab, _ = _cc(mask)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    cnt = np.bincount(lab[gt.astype(bool)], minlength=n)
    cnt[0] = 0
    return lab == int(np.argmax(cnt)) if cnt.max() > 0 else np.zeros(mask.shape, bool)


def iou_dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    inter, ps, gs = (p & g).sum(), p.sum(), g.sum()
    iou = inter / (ps + gs - inter) if (ps + gs - inter) else np.nan
    dice = 2 * inter / (ps + gs) if (ps + gs) else np.nan
    return float(iou), float(dice)


def pca_length(mask):
    """mask 沿自身第一主軸的延伸長度(px)。簡化版，不保證等同 B3 方法3。"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return np.nan
    pts = np.stack([xs, ys], 1).astype(np.float64)
    pts -= pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    proj = pts @ vt[0]
    return float(proj.max() - proj.min())


# ============================================================
# 其他
# ============================================================

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dataset_fingerprint(items):
    """所有影像+標註檔 SHA256 的總雜湊：資料有沒有被動過一眼看出。"""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda d: d["img_id"]):
        h.update(sha256_file(it["img_path"]).encode())
        h.update(sha256_file(it["label_path"]).encode())
    return h.hexdigest()


def build_model(encoder, encoder_weights):
    import segmentation_models_pytorch as smp          # 重依賴延後匯入
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=3, classes=1)
