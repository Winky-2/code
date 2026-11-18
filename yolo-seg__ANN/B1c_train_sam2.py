"""
B1s2_train_sam2.py
===============================================================
分割模型對照 ③：SAM 2.1（Hiera），框提示來自 B1 的 YOLO11n-seg。
與 B1sm（MedSAM）的訓練迴圈完全相同，差別只在骨幹：
  MedSAM = SAM1 ViT-B + 醫學影像預訓練、輸入 1024、mask 256 再放大
  SAM2   = Hiera 骨幹 + 自然影像/影片預訓練、mask decoder 有高解析度特徵（對細尖端可能較有利）

刻意改變的變因：輪廓模型（YOLO mask head → SAM2）
沿用 B1：資料夾、切分、增強強度、不翻轉、BATCH 4、EPOCHS 150、best 依 val fitness
無法對齊：AdamW 1e-4、輸入放大到 1024（SAM2 固定）、mosaic 不做、框抖動

操作（PowerShell）：
1. 安裝：torch ≥ 2.5.1（pytorch.org），再
       $env:SAM2_BUILD_CUDA = "0"      # Windows 上略過 CUDA 擴充編譯（只影響補洞後處理）
       pip install git+https://github.com/facebookresearch/sam2.git
       pip install ultralytics pandas openpyxl pyyaml opencv-python
   ⚠️ 官方建議 Windows 用 WSL；上面的方式我沒在 Windows 實測過
2. 下載權重 sam2.1_hiera_small.pt 放本資料夾：
       https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
   （要換 tiny / base_plus / large，SAM2_CFG 也要跟著換）
3. 先跑完 B1；先直接執行（RUN_NOW=False）檢查，再改 RUN_NOW=True
產出：sam2_run_<DATA_TAG>/ 結構同 B1

單檔版：資料切分、評估器、輸出格式、推論轉接都在本檔裡（五支 B1 各有一份相同的共用區塊），
不需要其他輔助檔。
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import yaml


# ==================== 共用設定（五支 B1 各有一份、內容相同；要改請五支一起改） ====================
SEG_DATA_DIR = "padded_train-308_tophat"            # 👈 與 B1_train_yolo11-seg.py 相同
DATA_TAG = "train-308_tophat"                       # WORK_DIR 名稱用：{模型}_run_{DATA_TAG}
YOLO_WORK_DIR = "yolo11n_seg_run_train-308_tophat"  # 👈 B1 的 WORK_DIR（核對切分、SAM 的框來源）
B1_FILE = "B1_train_yolo11-seg.py"                  # 👈 用來做 AST 比對（找不到就略過）

# 以下與 B1 對齊，不要單獨調整
IMG_SIZE = 640
VAL_RATIO = 0.2
RANDOM_SEED = 42            # 切分用（與 B1 相同）
TRAIN_SEED = 0              # 權重初始化 / 增強用（Ultralytics 預設 seed=0）
CLASS_NAMES = ["tooth"]
BATCH = 4
AMP = False
AUG = dict(degrees=10, translate=0.05, scale=0.1)   # 同 B1；fliplr=0 不翻轉；mosaic 不做（B1 是 0.3）

# 評估（對齊 Ultralytics val 預設）
VAL_CONF = 0.001
VAL_MAX_DET = 300
# =================================================================================


def work_dir_of(model_tag):
    return f"{model_tag}_run_{DATA_TAG}"


ARCH = "segx_sam2"
# ==================== 可手動修改 ====================
RUN_NOW = False

INIT_CKPT = "sam2.1_hiera_small.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"           # 與 INIT_CKPT 對應
PROMPT_WEIGHTS = f"{YOLO_WORK_DIR}/weights_ready.pt"     # 👈 B1 的 YOLO 權重（框來源）
EPOCHS = 150
LR0 = 1e-4
LRF = 0.01
WEIGHT_DECAY = 0.01
BOX_JITTER = 0.05
MAX_PROMPTS = 16
EVAL_EVERY = 5
DEVICE = None
# ===================================================

SETTINGS = dict(RUN_NOW=RUN_NOW, ARCH="segx_sam2", MODEL_NAME="SAM2.1 Hiera-S + YOLO 框",
                WORK_DIR=work_dir_of("sam2"), INIT_CKPT=INIT_CKPT, SAM2_CFG=SAM2_CFG,
                PROMPT_WEIGHTS=PROMPT_WEIGHTS, EPOCHS=EPOCHS, LR0=LR0, LRF=LRF,
                WEIGHT_DECAY=WEIGHT_DECAY, BOX_JITTER=BOX_JITTER, MAX_PROMPTS=MAX_PROMPTS,
                TRAIN_ENCODER=False, EVAL_EVERY=EVAL_EVERY, DEVICE=DEVICE)


# ######################################################################
# 共用區塊 A：切分（逐字複製自 B1）、增強、評估器（同 Ultralytics）、輸出（同 B1）
# ######################################################################
# =================================================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PAD_VALUE = 114              # Ultralytics letterbox 灰邊
METRIC_KEYS = ["Box_precision", "Box_recall", "Box_mAP50", "Box_mAP50-95",
               "Mask_precision", "Mask_recall", "Mask_mAP50", "Mask_mAP50-95"]


# ============================================================
# 第1部分：逐字複製自 B1（不要改；check_same_as_b1 會比對）
# ============================================================

def collect_seg_dataset():
    """相容 Roboflow 標準 train/valid/test 匯出或扁平 images/+labels/ 結構，
    全部 pool 起來，由這支腳本自己重新切一次 train/val。"""
    pool = []
    seen_ids = {}
    for split_name in ("train", "valid", "test", "images"):
        if split_name == "images":
            img_dir = Path(SEG_DATA_DIR) / "images"
            lbl_dir = Path(SEG_DATA_DIR) / "labels"
        else:
            img_dir = Path(SEG_DATA_DIR) / split_name / "images"
            lbl_dir = Path(SEG_DATA_DIR) / split_name / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"⚠️ {img_path.name} 沒有對應的標註檔，跳過")
                continue
            if img_path.stem in seen_ids:
                print(f"⚠️ 圖片ID重複：'{img_path.stem}'，已出現在 "
                      f"{seen_ids[img_path.stem]}，這次({split_name})跳過")
                continue
            seen_ids[img_path.stem] = split_name
            pool.append({"img_id": img_path.stem, "img_path": img_path,
                         "label_path": label_path})

    print(f"\n✅ 分割資料 pool 完成，總共 {len(pool)} 張X光片")
    if len(pool) < 20:
        print(f"⚠️ 只有 {len(pool)} 張圖有標註，資料量偏少，訓練結果可能不穩定。")
    return pool


def split_train_val(pool, val_ratio):
    shuffled = pool[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_items, train_items = shuffled[:n_val], shuffled[n_val:]
    print(f"📊 train/val 切分：train {len(train_items)} 張 / val {len(val_items)} 張"
          f"(val_ratio={val_ratio}, seed={RANDOM_SEED})")
    return train_items, val_items


def copy_items(items, dst_img_dir, dst_lbl_dir):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        shutil.copy2(item["img_path"], dst_img_dir / item["img_path"].name)
        shutil.copy2(item["label_path"], dst_lbl_dir / item["label_path"].name)


def count_polygons_per_image(items):
    """統計每張圖有幾個 instance、多邊形頂點數，並抓出格式異常的行。

    頂點數是分割標註品質的直接指標：Roboflow 的 Smart Polygon 產出的
    點數通常較多，手繪的較少。點數過少(例如 <8)代表輪廓被簡化得太粗，
    牙根尖端的形狀可能已經被抹平——而那正是你唯一在乎的區域。
    """
    records = []
    for item in items:
        n_inst, verts, n_bad = 0, [], 0
        with open(item["label_path"], "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for line in lines:
            parts = line.split()
            coords = parts[1:]
            # 合法的多邊形：偶數個座標，且至少 3 個點
            if len(coords) < 6 or len(coords) % 2 != 0:
                n_bad += 1
                continue
            n_inst += 1
            verts.append(len(coords) // 2)
        records.append({
            "檔名": item["img_path"].name,
            "instance數": n_inst,
            "最少頂點數": min(verts) if verts else 0,
            "平均頂點數": round(sum(verts) / len(verts), 1) if verts else 0,
            "格式異常行數": n_bad,
        })
    return records


def report_annotation_quality(df: pd.DataFrame):
    if df.empty:
        return
    n_bad = int(df["格式異常行數"].sum())
    if n_bad:
        print(f"❌ 發現 {n_bad} 行格式異常的標註(座標數不是偶數或少於3點)。")
        print(f"   最可能的原因：匯出成了 pose/bbox 格式而不是 Segmentation 格式。")

    mean_inst = df["instance數"].mean()
    if mean_inst < 1.5:
        print(f"ℹ️  每張圖平均 {mean_inst:.1f} 個 instance → 判定為「A 案：只標目標牙」")
        print(f"   這是預期的。模型 confidence 會偏低，B3 的 SEG_CONF 建議設 0.10~0.15。")
    else:
        print(f"ℹ️  每張圖平均 {mean_inst:.1f} 個 instance → 判定為「B 案：標多顆牙」")

    thin = df[df["最少頂點數"] < 8]
    if not thin.empty:
        print(f"⚠️ 有 {len(thin)} 張圖的多邊形頂點數少於 8，輪廓可能過度簡化：")
        print(f"   {', '.join(thin['檔名'].head(5).tolist())}")
        print(f"   → 牙根尖端的形狀是這個方案的重點，頂點太少會把它抹平，建議補標。")


B1_COPIED = ["collect_seg_dataset", "split_train_val", "copy_items",
             "count_polygons_per_image", "report_annotation_quality"]


def check_same_as_b1():
    """AST 比對：上面 5 個函式與 B1 原檔是否完全相同（排版/註解不算差異）。"""
    p = Path(B1_FILE)
    if not p.exists():
        print(f"ℹ️  找不到 {B1_FILE}，略過 AST 比對（不影響訓練）")
        return None

    def defs(src):
        tree = ast.parse(src)
        return {n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.FunctionDef)}

    theirs = defs(p.read_text(encoding="utf-8"))
    mine = defs(Path(__file__).read_text(encoding="utf-8"))
    diff = [n for n in B1_COPIED if theirs.get(n) != mine.get(n)]
    if diff:
        print(f"❌ 與 B1 不同的函式：{diff} → 切分/統計可能不一致，先查原因")
        return False
    print(f"✅ 切分與標註統計函式與 {B1_FILE} 完全相同（AST 比對 {len(B1_COPIED)} 個）")
    return True


def check_split_against_yolo(val_items):
    d = Path(YOLO_WORK_DIR) / "data" / "val" / "images"
    if not d.exists():
        print(f"ℹ️  找不到 {d}，無法核對 val 是否與 YOLO 相同（不影響訓練）")
        return None
    yolo = {q.stem for q in d.iterdir() if q.suffix.lower() in IMAGE_EXTENSIONS}
    mine = {it["img_id"] for it in val_items}
    if yolo == mine:
        print(f"✅ val 切分與 YOLO 完全相同（{len(mine)} 張）")
        return True
    print(f"❌ val 切分與 YOLO 不同：只在這次 {len(mine - yolo)} 張 / 只在YOLO {len(yolo - mine)} 張"
          f" → 這樣就不是單一變因對照，先查原因")
    return False


def prepare_split():
    """B1 的 train_seg_model() 前半段：pool → 切分 → 標註品質。回傳 (train, val, poly_df)。"""
    check_same_as_b1()
    pool = collect_seg_dataset()
    if not pool:
        raise FileNotFoundError(f"在 {SEG_DATA_DIR} 裡找不到任何已標註的資料")
    random.seed(RANDOM_SEED)            # 等同 B1 模組層的 random.seed，且緊接著 shuffle
    train_items, val_items = split_train_val(pool, VAL_RATIO)
    check_split_against_yolo(val_items)
    poly_df = pd.DataFrame(count_polygons_per_image(train_items))
    print("\n=== 標註品質 ===")
    report_annotation_quality(poly_df)
    return train_items, val_items, poly_df


def copy_data_like_b1(work_dir, train_items, val_items):
    """同 B1：WORK_DIR/data/{train,val}/{images,labels} + data.yaml。"""
    data_root = Path(work_dir) / "data"
    copy_items(train_items, data_root / "train" / "images", data_root / "train" / "labels")
    copy_items(val_items, data_root / "val" / "images", data_root / "val" / "labels")
    data_yaml = {"path": str(data_root.resolve()), "train": "train/images",
                 "val": "val/images", "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    with open(data_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True)
    return data_root


# ============================================================
# 第2部分：影像 / 標註 / 增強
# ============================================================

def imread_rgb(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"讀不到影像：{path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite(path, img):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        raise IOError(f"寫不出影像：{path}")
    buf.tofile(str(path))


def load_instances(label_path, H, W):
    """YOLO-seg 多邊形 → 每個 instance 一張 bool mask。"""
    masks = []
    for line in Path(label_path).read_text(encoding="utf-8").splitlines():
        v = line.split()[1:]
        if len(v) < 6 or len(v) % 2:
            continue
        pts = np.asarray(v, np.float64).reshape(-1, 2) * [W, H]
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if m.any():
            masks.append(m.astype(bool))
    return masks


def augment(img, masks, rng, degrees=10, translate=0.05, scale=0.1):
    """旋轉/平移/縮放，強度同 B1，不翻轉；所有 instance 用同一個矩陣。"""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-degrees, degrees),
                                rng.uniform(1 - scale, 1 + scale))
    M[:, 2] += rng.uniform(-translate, translate, 2) * [W, H]
    img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderValue=(PAD_VALUE,) * 3)
    masks = [cv2.warpAffine(m.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST,
                            borderValue=0).astype(bool) for m in masks]
    return img, masks


def drop_small(masks, min_area=50):
    """增強後被推出畫面的 instance 丟掉。"""
    return [m for m in masks if m.sum() >= min_area]


def mask_box(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


# ============================================================
# 第3部分：評估器（定義同 Ultralytics val）
# ============================================================

IOUV = np.linspace(0.5, 0.95, 10)


def match_predictions(iou):
    """Ultralytics DetectionValidator.match_predictions 的單一類別版。
    iou: (n_gt, n_pred) → correct: (n_pred, 10) bool"""
    correct = np.zeros((iou.shape[1], len(IOUV)), bool)
    for i, thr in enumerate(IOUV):
        matches = np.array(np.nonzero(iou >= thr)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).prod(1)
    ab = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (aa[:, None] + ab[None] - inter + 1e-9)


def mask_iou(gts, prs, gboxes, pboxes):
    """只在兩個外接框的交集區算交集，省記憶體。"""
    iou = np.zeros((len(gts), len(prs)))
    ga = [g.sum() for g in gts]
    pa = [p.sum() for p in prs]
    for j, p in enumerate(prs):
        for i, g in enumerate(gts):
            x0 = int(max(gboxes[i, 0], pboxes[j, 0]))
            y0 = int(max(gboxes[i, 1], pboxes[j, 1]))
            x1 = int(min(gboxes[i, 2], pboxes[j, 2]))
            y1 = int(min(gboxes[i, 3], pboxes[j, 3]))
            if x1 <= x0 or y1 <= y0:
                continue
            inter = np.count_nonzero(g[y0:y1, x0:x1] & p[y0:y1, x0:x1])
            iou[i, j] = inter / (ga[i] + pa[j] - inter + 1e-9)
    return iou


def _ap_summary(tp, conf, n_gt_total):
    if n_gt_total == 0:
        return 0.0, 0.0, 0.0, 0.0
    if len(conf) == 0:
        return 0.0, 0.0, 0.0, 0.0
    from ultralytics.utils.metrics import ap_per_class      # 重依賴延後匯入
    tcls = np.zeros(n_gt_total)
    res = ap_per_class(tp, conf, np.zeros(len(conf)), tcls)
    p, r, ap = res[2], res[3], res[5]
    return float(p.mean()), float(r.mean()), float(ap[:, 0].mean()), float(ap.mean())


def evaluate(predict_fn, items, conf=VAL_CONF, max_det=VAL_MAX_DET, progress=False):
    """predict_fn(img_rgb, conf) → (masks: list[bool HxW], scores: array)
    回傳 B1 同名的 8 個指標 + fitness（Ultralytics：Σ(0.1·mAP50 + 0.9·mAP50-95)，Box+Mask）。"""
    tp_b, tp_m, confs, n_gt = [], [], [], 0
    for k, it in enumerate(items):
        img = imread_rgb(it["img_path"])
        H, W = img.shape[:2]
        gts = load_instances(it["label_path"], H, W)
        masks, scores = predict_fn(img, conf)
        scores = np.asarray(scores, np.float64)
        keep = [i for i in np.argsort(-scores) if masks[i].any()][:max_det]
        masks = [masks[i] for i in keep]
        scores = scores[keep]
        n_gt += len(gts)
        if not masks:
            continue
        gb = np.stack([mask_box(g) for g in gts]) if gts else np.zeros((0, 4))
        pb = np.stack([mask_box(m) for m in masks])
        if gts:
            tp_b.append(match_predictions(box_iou(gb, pb)))
            tp_m.append(match_predictions(mask_iou(gts, masks, gb, pb)))
        else:
            tp_b.append(np.zeros((len(masks), 10), bool))
            tp_m.append(np.zeros((len(masks), 10), bool))
        confs.append(scores)
        if progress and (k + 1) % 20 == 0:
            print(f"   評估 {k + 1}/{len(items)}")
    conf_all = np.concatenate(confs) if confs else np.zeros(0)
    b = _ap_summary(np.concatenate(tp_b) if tp_b else np.zeros((0, 10), bool), conf_all, n_gt)
    m = _ap_summary(np.concatenate(tp_m) if tp_m else np.zeros((0, 10), bool), conf_all, n_gt)
    out = {k: round(v, 4) for k, v in zip(METRIC_KEYS, b + m)}
    out["fitness"] = round(0.1 * b[2] + 0.9 * b[3] + 0.1 * m[2] + 0.9 * m[3], 4)
    return out


# ============================================================
# 第4部分：輸出（同 B1）
# ============================================================

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dataset_fingerprint(items):
    h = hashlib.sha256()
    for it in sorted(items, key=lambda d: d["img_id"]):
        h.update(sha256_file(it["img_path"]).encode())
        h.update(sha256_file(it["label_path"]).encode())
    return h.hexdigest()


def run_paths(work_dir):
    wd = Path(work_dir)
    return dict(work=wd, save=wd / "tooth_seg", weights=wd / "tooth_seg" / "weights",
                best=wd / "tooth_seg" / "weights" / "best.pt",
                last=wd / "tooth_seg" / "weights" / "last.pt",
                results=wd / "tooth_seg" / "results.csv",
                ready=wd / "weights_ready.pt", config=wd / "config.json",
                summary=wd / "訓練摘要.xlsx")


def write_config(work_dir, arch, cfg, train_items, val_items):
    p = run_paths(work_dir)
    p["work"].mkdir(parents=True, exist_ok=True)
    shared = dict(SEG_DATA_DIR=SEG_DATA_DIR, IMG_SIZE=IMG_SIZE, VAL_RATIO=VAL_RATIO,
                  RANDOM_SEED=RANDOM_SEED, TRAIN_SEED=TRAIN_SEED, BATCH=BATCH, AMP=AMP,
                  AUG=AUG, VAL_CONF=VAL_CONF, VAL_MAX_DET=VAL_MAX_DET)
    data = {"ARCH": arch, "SCRIPT": Path(__file__).name, **shared, **cfg, "n_train": len(train_items), "n_val": len(val_items),
            "train_fingerprint": dataset_fingerprint(train_items),
            "val_fingerprint": dataset_fingerprint(val_items),
            "split": {"train": [it["img_id"] for it in train_items],
                      "val": [it["img_id"] for it in val_items]}}
    p["config"].write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str),
                           encoding="utf-8")


def write_summary(work_dir, train_items, val_items, poly_df, val_metrics, model_name):
    """與 B1 train_seg_model() 結尾完全相同的 訓練摘要.xlsx。"""
    p = run_paths(work_dir)
    total_inst = int(poly_df["instance數"].sum()) if not poly_df.empty else 0
    metrics = {k: val_metrics[k] for k in METRIC_KEYS if k in val_metrics}
    summary_df = pd.DataFrame([{
        "train張數": len(train_items),
        "val張數": len(val_items),
        "train_instance總數": total_inst,
        "每張平均instance數": round(poly_df["instance數"].mean(), 2) if not poly_df.empty else 0,
        "每張平均頂點數": round(poly_df["平均頂點數"].mean(), 1) if not poly_df.empty else 0,
        **metrics,
    }])
    with pd.ExcelWriter(p["summary"], engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="訓練摘要", index=False)
        poly_df.to_excel(writer, sheet_name="train每張標註統計", index=False)

    print(f"📍 訓練摘要：{p['summary']}")
    print(f"   {metrics}")
    print(f"   ⚠️ 這是 val 集(約 {len(val_items)} 張)的表現，只能看訓練有沒有學起來。")
    print(f"      Mask mAP 高 ≠ 根尖畫得準(根尖只佔全牙面積極小比例)，")
    print(f"      真正的指標是 B3 對 test 圖算出來的長度誤差。")
    print(f"\n=======================================================")
    print(f"✨ {model_name} 牙齒分割模型訓練完成！")
    print(f"📍 權重路徑: {p['ready']}")
    print(f"=======================================================")
    print(f"👉 接下來：B1x_compare_val.py（同評估器對照）→ B234x_run_segx.py（B2→B3→B4）")


def finish_weights(work_dir):
    p = run_paths(work_dir)
    if not p["best"].exists():
        raise FileNotFoundError(f"找不到 {p['best']}，訓練可能中途失敗")
    shutil.copy2(p["best"], p["ready"])
    print(f"\n✅ 訓練完成，模型存成: {p['ready']}")


def already_trained(work_dir):
    p = run_paths(work_dir)
    if p["ready"].exists():
        print(f"⚠️ 偵測到已訓練的模型 {p['ready']}，不重新訓練。若要重新訓練請先刪除 {p['work']}")
        return True
    return False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def fitness_of(m):
    return m.get("fitness", -1.0)


# ######################################################################
# 共用區塊 B：模型建立 / 推論 / Ultralytics 介面轉接（B2/B3 原檔不用改）
# ######################################################################


# ============================================================
# 共用小工具
# ============================================================

def pick_device(device=None):
    import torch
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_dir_of(path):
    p = Path(path)
    return p if p.is_dir() else p.parent


def read_cfg(path):
    c = run_dir_of(path) / "config.json"
    if not c.exists():
        return None
    try:
        return json.loads(c.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def mask_to_xy(mask):
    """二值 mask → 多邊形 (N,2)。多個連通區依面積大到小串接（同 _maskrcnn_common），
    B2/B3 的 REMOVE_FRAGMENTS 才會跟 YOLO 時一樣生效。"""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if len(c) >= 3]
    if not cs:
        return None
    cs.sort(key=cv2.contourArea, reverse=True)
    return np.concatenate([c.reshape(-1, 2) for c in cs]).astype(np.float32)


class SegxModel:
    """model.predict(source=..., imgsz=..., conf=...)[0] → .orig_shape / .masks.xy / .boxes.conf/.xyxy"""

    def __init__(self, path, device=None):
        self.cfg = read_cfg(path)
        self.backend = build_backend(path, device)
        print(f"🔁 segx 轉接：{path}（{self.cfg['ARCH']}）")

    def predict(self, source, imgsz=None, conf=0.25, save=False, verbose=False, **_):
        img = source if isinstance(source, np.ndarray) else imread_rgb(source)
        if isinstance(source, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # Ultralytics 慣例：陣列是 BGR
        H, W = img.shape[:2]
        masks, scores = self.backend.predict_instances(img, float(conf))
        polys, confs, boxes = [], [], []
        for m, s in zip(masks, scores):
            xy = mask_to_xy(m)
            if xy is None:
                continue
            polys.append(xy)
            confs.append(float(s))
            boxes.append([xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()])
        return [SimpleNamespace(
            orig_shape=(H, W),
            masks=SimpleNamespace(xy=polys) if polys else None,
            boxes=SimpleNamespace(conf=np.asarray(confs, np.float32),
                                  xyxy=np.asarray(boxes, np.float32).reshape(-1, 4)))]


def dice_bce(logits, target):
    """logits / target: (N,H,W)。MedSAM / SAM2 微調用的 Dice + BCE。"""
    import torch.nn.functional as F
    t = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, t)
    p = logits.sigmoid()
    inter = (p * t).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)
    return bce + dice.mean()


# ============================================================
# YOLO 框提示（MedSAM / SAM2 用）
# ============================================================

class YoloPrompter:
    """用 B1 的 YOLO11n-seg 偵測框當 SAM 的提示；分數 = YOLO conf。
    這樣 SAM 系列只換掉「輪廓」，偵測與分數跟 YOLO 相同。"""

    def __init__(self, weights, imgsz=640):
        import ultralytics
        yolo_cls = getattr(ultralytics.YOLO, "_segx_orig", ultralytics.YOLO)
        self.model = yolo_cls(str(weights))
        self.imgsz = imgsz
        self._cache = {}

    def __call__(self, img_rgb, conf, key=None):
        if key is not None and key in self._cache:
            boxes, scores = self._cache[key]
        else:
            r = self.model.predict(source=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                                   imgsz=self.imgsz, conf=min(conf, 0.001), max_det=300,
                                   verbose=False)[0]
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = r.boxes.conf.cpu().numpy().astype(np.float32)
            if key is not None:
                self._cache[key] = (boxes, scores)
        keep = scores >= conf
        return boxes[keep], scores[keep]


# ============================================================
# SAM2（官方 sam2 套件）
# ============================================================

class SAM2Net:
    """沿用 SAM2ImagePredictor 的前處理；decode 重寫 _predict 的框分支（box → 標籤 2/3 的兩個點）。
    ⚠️ 用到 predictor 的內部屬性（_features、_transforms、_orig_hw），sam2 改版可能要跟著改。"""

    def __init__(self, model_cfg, init_ckpt=None, device=None):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device = pick_device(device)
        self.model = build_sam2(model_cfg, init_ckpt, device=self.device)
        self.pred = SAM2ImagePredictor(self.model)

    def trainable_parameters(self, train_encoder=False):
        if train_encoder:
            raise NotImplementedError("SAM2 這裡只微調 mask decoder（set_image 在 no_grad 下跑）")
        for p in self.model.parameters():
            p.requires_grad = False
        ps = list(self.model.sam_mask_decoder.parameters())
        for p in ps:
            p.requires_grad = True
        return ps

    def train(self, flag=True):
        self.model.eval()
        self.model.sam_mask_decoder.train(flag)

    def prepare(self, img_rgb, grad=False):
        self.pred.set_image(img_rgb)

    def decode(self, boxes_xyxy):
        import torch
        b = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=self.device)
        hw = self.pred._orig_hw[-1]
        coords = self.pred._transforms.transform_boxes(b, normalize=True, orig_hw=hw).reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int, device=self.device).repeat(coords.shape[0], 1)
        sparse, dense = self.model.sam_prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        hr = [f[-1].unsqueeze(0) for f in self.pred._features["high_res_feats"]]
        low, _, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self.pred._features["image_embed"][-1].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False, repeat_image=coords.shape[0] > 1, high_res_features=hr)
        return self.pred._transforms.postprocess_masks(low, hw)[:, 0]

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


class SamPredictor:
    """SAM 系列推論：YOLO 框 → SAM 輪廓。"""

    def __init__(self, net, prompter):
        self.net, self.prompter = net, prompter

    def predict_instances(self, img_rgb, conf, key=None):
        import hashlib

        import torch
        if key is None:                    # 同一張圖的 YOLO 框只算一次（val 每次評估都重用）
            key = hashlib.md5(np.ascontiguousarray(img_rgb).tobytes()).hexdigest()
        boxes, scores = self.prompter(img_rgb, conf, key=key)
        if len(boxes) == 0:
            return [], np.zeros(0)
        self.net.train(False)
        masks = []
        with torch.no_grad():
            self.net.prepare(img_rgb)
            for i in range(0, len(boxes), 32):                 # 分批，避免框太多爆記憶體
                masks += list((self.net.decode(boxes[i:i + 32]) > 0).cpu().numpy())
        return masks, scores


def build_sam_net(cfg, device=None, init=True):
    """init=True 用原始預訓練權重；False 只建架構（之後 load 微調權重）。"""
    return SAM2Net(cfg["SAM2_CFG"], cfg["INIT_CKPT"] if init else None, device)


def build_backend(path, device=None):
    """SAM2 推論器：讀 WORK_DIR/config.json + weights_ready.pt。"""
    cfg = read_cfg(path)
    if not cfg or cfg.get("ARCH") != ARCH:
        raise ValueError(f"{path} 不是 {ARCH} 的權重（WORK_DIR 裡要有 config.json）")
    import torch
    net = build_sam_net(cfg, device, init=False)
    net.load_state_dict(torch.load(run_dir_of(path) / "weights_ready.pt", map_location=net.device))
    return SamPredictor(net, YoloPrompter(cfg["PROMPT_WEIGHTS"], cfg["IMG_SIZE"]))


# ######################################################################
# 訓練迴圈
# ######################################################################


def make_cfg(s):
    keys = ["ARCH", "INIT_CKPT", "SAM2_CFG", "PROMPT_WEIGHTS", "EPOCHS", "LR0", "LRF",
            "WEIGHT_DECAY", "BOX_JITTER", "MAX_PROMPTS", "TRAIN_ENCODER", "EVAL_EVERY", "DEVICE"]
    cfg = {k: s[k] for k in keys if k in s}
    cfg["PROMPT_WEIGHTS"] = str(Path(s["PROMPT_WEIGHTS"]).resolve())
    cfg["INIT_CKPT"] = str(Path(s["INIT_CKPT"]).resolve())
    return cfg


def jitter_boxes(masks, rng, frac, H, W):
    out = []
    for m in masks:
        b = mask_box(m)
        w, h = b[2] - b[0], b[3] - b[1]
        b = b + rng.uniform(-frac, frac, 4) * np.array([w, h, w, h])
        out.append(np.clip(b, 0, [W, H, W, H]))
    return np.asarray(out, np.float32)


def run(s):
    """s：入口腳本的設定 dict（見 B1sm / B1s2）。"""
    work_dir = s["WORK_DIR"]
    train_items, val_items, poly_df = prepare_split()
    for k in ("INIT_CKPT", "PROMPT_WEIGHTS"):
        if not Path(s[k]).exists():
            print(f"❌ 找不到 {k} = {s[k]}")
            if s["RUN_NOW"]:
                raise FileNotFoundError(s[k])
    if not s["RUN_NOW"]:
        print(f"\nRUN_NOW=False：只檢查，未訓練。輸出會在 {work_dir}/")
        return
    if already_trained(work_dir):
        return

    import pandas as pd
    import torch

    cfg = make_cfg(s)
    copy_data_like_b1(work_dir, train_items, val_items)
    write_config(work_dir, s["ARCH"], cfg, train_items, val_items)
    P = run_paths(work_dir)
    P["weights"].mkdir(parents=True, exist_ok=True)

    set_seed(TRAIN_SEED)
    rng = np.random.default_rng(TRAIN_SEED)
    net = build_sam_net({**cfg, "ARCH": s["ARCH"]}, s["DEVICE"], init=True)
    print(f"🖥️  device = {net.device}")
    params = net.trainable_parameters(s["TRAIN_ENCODER"])
    print(f"🔧 可訓練參數 {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=s["LR0"], weight_decay=s["WEIGHT_DECAY"])
    steps_per_ep = math.ceil(len(train_items) / BATCH)
    total = s["EPOCHS"] * steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 1 - (1 - s["LRF"]) * i / max(total, 1))
    pred = SamPredictor(net, YoloPrompter(s["PROMPT_WEIGHTS"], IMG_SIZE))

    history, best = [], -1.0
    print(f"\n{'epoch':>5} {'lr':>9} {'loss':>8} {'Mask_mAP50-95':>14} {'fitness':>8}")
    for ep in range(s["EPOCHS"]):
        t0 = time.time()
        net.train(True)
        losses, k = [], 0
        opt.zero_grad(set_to_none=True)
        order = rng.permutation(len(train_items))
        for j, idx in enumerate(order):
            it = train_items[idx]
            img = imread_rgb(it["img_path"])
            H, W = img.shape[:2]
            masks = load_instances(it["label_path"], H, W)
            img, masks = augment(img, masks, rng, **AUG)
            masks = drop_small(masks)
            if masks:
                if len(masks) > s["MAX_PROMPTS"]:
                    masks = [masks[i] for i in rng.choice(len(masks), s["MAX_PROMPTS"], replace=False)]
                boxes = jitter_boxes(masks, rng, s["BOX_JITTER"], H, W)
                net.prepare(img, grad=s["TRAIN_ENCODER"])
                logits = net.decode(boxes)
                gt = torch.from_numpy(np.stack(masks)).to(logits.device)
                loss = dice_bce(logits, gt)
                if not math.isfinite(loss.item()):
                    raise RuntimeError(f"epoch {ep + 1} loss = {loss.item()}，訓練發散；先把 LR0 減半")
                (loss / BATCH).backward()
                losses.append(loss.item())
            k += 1
            if k % BATCH == 0 or j == len(order) - 1:
                opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()

        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"],
                   train_loss=float(np.mean(losses)) if losses else float("nan"))
        last = ep == s["EPOCHS"] - 1
        mark = ""
        if (ep + 1) % s["EVAL_EVERY"] == 0 or last:
            m = evaluate(pred.predict_instances, val_items)
            row.update(m)
            if m["fitness"] > best:
                best = m["fitness"]
                torch.save(net.state_dict(), P["best"])
                mark = " ★"
            print(f"{ep + 1:>5} {row['lr']:>9.2e} {row['train_loss']:>8.4f}"
                  f" {m['Mask_mAP50-95']:>14.4f} {m['fitness']:>8.4f}{mark}")
        else:
            print(f"{ep + 1:>5} {row['lr']:>9.2e} {row['train_loss']:>8.4f}")
        row["sec"] = round(time.time() - t0, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(P["results"], index=False, encoding="utf-8-sig")

    torch.save(net.state_dict(), P["last"])
    finish_weights(work_dir)
    print("\n=== val 評估（best.pt，與 B1 同定義的 Box/Mask mAP）===")
    m = evaluate(build_backend(P["ready"], s["DEVICE"]).predict_instances, val_items, progress=True)
    write_summary(work_dir, train_items, val_items, poly_df, m, s["MODEL_NAME"])


def print_header(doc, s):
    print(doc)
    print(f"待執行設定：{json.dumps({k: v for k, v in s.items() if k != 'RUN_NOW'}, ensure_ascii=False, default=str)}\n")


if __name__ == "__main__":
    print_header(__doc__, SETTINGS)
    run(SETTINGS)
