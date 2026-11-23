"""
B1d_train_pointrend.py
===============================================================
分割模型對照 ④：Mask R-CNN + PointRend（detectron2 官方實作，COCO 預訓練 R50-FPN 3x）。

為什麼要 PointRend：Mask R-CNN 每顆牙只出 28×28 的 mask 再放大（640 圖上每格約 18px），
根尖這種細尖端容易被抹鈍；PointRend 在「不確定的邊界點」上逐步細修到 224×224。

USE_POINTREND = False → 同一支腳本跑 detectron2 的「純 Mask R-CNN」。
建議兩個都跑：PointRend 的效果 = 兩者的差（同框架、同配方，唯一變因是 mask head）。
（跟 B1m 的 torchvision Mask R-CNN 比，框架與預設細節不同，不是單一變因。）

沿用 B1：資料夾、切分、增強強度（旋轉±10°/平移5%/縮放10%、不翻轉）、BATCH 4、AMP 關、輸入維持 640
沿用 B1m：SGD 0.005 / momentum 0.9 / wd 1e-4 + warmup 200 iter，線性衰減到 ×0.01，EPOCHS 50
best 依 val fitness（0.1·mAP50 + 0.9·mAP50-95，Box+Mask，同 Ultralytics）挑

操作：
1. detectron2 在 Windows 要自己編譯，PointRend 也需要 detectron2 的原始碼資料夾 → 建議用 WSL2 或 Colab。
   WSL / Linux（bash）：
       git clone https://github.com/facebookresearch/detectron2.git
       pip install -e detectron2
       pip install ultralytics pandas openpyxl pyyaml opencv-python
   Windows 原生（PowerShell，需 Visual Studio Build Tools + 與 torch 相同版本的 CUDA Toolkit）：
       git clone https://github.com/facebookresearch/detectron2.git
       pip install -e .\\detectron2
   ⚠️ Windows 原生編譯我沒實測過，失敗率不低
2. DETECTRON2_ROOT 指向上面 clone 的資料夾；第一次執行會自動下載 COCO 權重
3. 先直接執行（RUN_NOW=False）檢查，再改 RUN_NOW=True
產出：pointrend_r50fpn_run_<DATA_TAG>/（或 d2maskrcnn_r50fpn_run_…）結構同 B1，另有 d2_config.yaml

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


ARCH = "segx_d2"
# ==================== 可手動修改 ====================
RUN_NOW = True                  # False = 只檢查、產出 config.json 與訓練摘要.xlsx

USE_POINTREND = True
DETECTRON2_ROOT = "detectron2"          # 👈 detectron2 git clone 的路徑
PRETRAINED = True                       # COCO 預訓練
EPOCHS = 150                             # 同 B1m
LR0 = 0.005                             # 同 B1m（0.02×4/16 線性縮放）
LRF = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
WARMUP_ITERS = 200
EVAL_EVERY = 1
DEVICE = None
# ===================================================

MODEL_TAG = "pointrend_r50fpn" if USE_POINTREND else "d2maskrcnn_r50fpn"
CFG = dict(USE_POINTREND=USE_POINTREND, DETECTRON2_ROOT=str(DETECTRON2_ROOT), PRETRAINED=PRETRAINED,
           EPOCHS=EPOCHS, LR0=LR0, LRF=LRF, MOMENTUM=MOMENTUM, WEIGHT_DECAY=WEIGHT_DECAY,
           WARMUP_ITERS=WARMUP_ITERS, EVAL_EVERY=EVAL_EVERY)
SETTINGS = dict(RUN_NOW=RUN_NOW, ARCH="segx_d2", CFG=CFG, WORK_DIR=work_dir_of(MODEL_TAG),
                MODEL_NAME=("Mask R-CNN + PointRend" if USE_POINTREND else "Mask R-CNN") + " (detectron2)",
                OPTIM="sgd", SCHED="linear", DEVICE=DEVICE, **CFG)


class D2Wrapper:
    def __init__(self):
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.modeling import build_model
        self.device = pick_device(DEVICE)
        self.cfg, init = d2_build_cfg(USE_POINTREND, DETECTRON2_ROOT, IMG_SIZE, self.device)
        self.model = build_model(self.cfg)
        if PRETRAINED:
            DetectionCheckpointer(self.model).load(init)      # 類別數不同的 head 會自動略過

    def param_groups(self):
        return [{"params": [p for p in self.model.parameters() if p.requires_grad]}]

    def loss(self, imgs, mlist):
        import torch
        from detectron2.structures import BitMasks, Boxes, Instances
        batch = []
        for img, masks in zip(imgs, mlist):
            if not masks:
                continue
            H, W = img.shape[:2]
            inst = Instances((H, W))
            inst.gt_boxes = Boxes(torch.from_numpy(np.stack([mask_box(m) for m in masks])))
            inst.gt_classes = torch.zeros(len(masks), dtype=torch.int64)
            inst.gt_masks = BitMasks(torch.from_numpy(np.stack(masks)))
            batch.append({"image": d2_input(img, self.device), "instances": inst,
                          "height": H, "width": W})
        if not batch:
            return None, {}
        from detectron2.utils.events import EventStorage
        with EventStorage():                                   # d2 訓練時要有 storage 記 log
            ld = self.model(batch)
        loss = sum(ld.values())
        return loss, {**{k: v.item() for k, v in ld.items()}, "loss": loss.item()}

    def predict_instances(self, img, conf):
        return d2_predict(self.model, img, conf, self.device)

    def train(self, flag):
        self.model.train(flag)

    def state_dict(self):
        return self.model.state_dict()

    def save_extra(self, work_dir):
        from pathlib import Path
        (Path(work_dir) / "d2_config.yaml").write_text(self.cfg.dump(), encoding="utf-8")


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


# ============================================================
# detectron2：Mask R-CNN / PointRend
# ============================================================

D2_MASKRCNN_CFG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
D2_POINTREND_CFG = "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
D2_POINTREND_URL = ("https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/"
                    "pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl")


def _import_point_rend(d2_root):
    try:
        from detectron2.projects import point_rend          # 原始碼 / editable 安裝時可用
        return point_rend
    except ImportError:
        pr = Path(d2_root) / "projects" / "PointRend"
        if not pr.exists():
            raise ImportError(f"找不到 PointRend：{pr}（DETECTRON2_ROOT 要指向 detectron2 的 git clone）")
        sys.path.insert(0, str(pr))
        import point_rend
        return point_rend


def d2_build_cfg(use_pointrend, d2_root, img_size, device, from_yaml=None):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    cfg = get_cfg()
    if use_pointrend:
        _import_point_rend(d2_root).add_pointrend_config(cfg)
    if from_yaml:                                           # 推論：直接讀訓練時存下的完整 cfg
        cfg.merge_from_file(str(from_yaml))
        cfg.MODEL.DEVICE = device
        cfg.MODEL.WEIGHTS = ""
        return cfg, None
    if use_pointrend:
        cfg.merge_from_file(str(Path(d2_root) / D2_POINTREND_CFG))
        cfg.MODEL.POINT_HEAD.NUM_CLASSES = 1
        init = D2_POINTREND_URL
    else:
        cfg.merge_from_file(model_zoo.get_config_file(D2_MASKRCNN_CFG))
        init = model_zoo.get_checkpoint_url(D2_MASKRCNN_CFG)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.INPUT.MIN_SIZE_TRAIN = (img_size,)                  # 維持 640，不放大到 800（同 B1m）
    cfg.INPUT.MAX_SIZE_TRAIN = img_size
    cfg.INPUT.MIN_SIZE_TEST = img_size
    cfg.INPUT.MAX_SIZE_TEST = img_size
    cfg.TEST.DETECTIONS_PER_IMAGE = 300
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = init
    return cfg, init


def d2_input(img_rgb, device):
    import torch
    bgr = np.ascontiguousarray(img_rgb[:, :, ::-1].transpose(2, 0, 1))
    return torch.as_tensor(bgr.astype(np.float32), device=device)


class D2Seg:
    def __init__(self, cfg_json, work_dir, device=None):
        import torch
        from detectron2.modeling import build_model
        dev = pick_device(device)
        cfg, _ = d2_build_cfg(cfg_json["USE_POINTREND"], cfg_json.get("DETECTRON2_ROOT", ""),
                              cfg_json["IMG_SIZE"], dev,
                              from_yaml=Path(work_dir) / "d2_config.yaml")
        self.model = build_model(cfg)
        self.model.load_state_dict(torch.load(Path(work_dir) / "weights_ready.pt",
                                              map_location=dev))
        self.model.eval()
        self.device = dev

    def predict_instances(self, img_rgb, conf, key=None):
        return d2_predict(self.model, img_rgb, conf, self.device)


def d2_predict(model, img_rgb, conf, device):
    import torch
    H, W = img_rgb.shape[:2]
    model.roi_heads.box_predictor.test_score_thresh = float(conf)
    with torch.no_grad():
        ins = model([{"image": d2_input(img_rgb, device), "height": H, "width": W}])[0]["instances"]
    return list(ins.pred_masks.cpu().numpy().astype(bool)), ins.scores.cpu().numpy()


def build_backend(path, device=None):
    """detectron2 推論器：讀 WORK_DIR/config.json + weights_ready.pt。"""
    cfg = read_cfg(path)
    if not cfg or cfg.get("ARCH") != ARCH:
        raise ValueError(f"{path} 不是 {ARCH} 的權重（WORK_DIR 裡要有 config.json）")
    return D2Seg(cfg, run_dir_of(path), device)


# ######################################################################
# 訓練迴圈
# ######################################################################


def run(s, build_wrapper):
    work_dir = s["WORK_DIR"]
    train_items, val_items, poly_df = prepare_split()
    if not s["RUN_NOW"]:
        print(f"\nRUN_NOW=False：只檢查，未訓練。輸出會在 {work_dir}/")
        return
    if already_trained(work_dir):
        return

    import pandas as pd
    import torch

    copy_data_like_b1(work_dir, train_items, val_items)
    write_config(work_dir, s["ARCH"], s["CFG"], train_items, val_items)
    P = run_paths(work_dir)
    P["weights"].mkdir(parents=True, exist_ok=True)

    set_seed(TRAIN_SEED)
    rng = np.random.default_rng(TRAIN_SEED)
    w = build_wrapper()
    w.save_extra(work_dir)
    print(f"🖥️  device = {w.device}")
    if w.device == "cpu":
        print("   ⚠️ CPU 訓練會非常慢，確認 torch 有裝 CUDA 版")

    groups = w.param_groups()
    if s["OPTIM"] == "sgd":
        opt = torch.optim.SGD(groups, lr=s["LR0"], momentum=s["MOMENTUM"],
                              weight_decay=s["WEIGHT_DECAY"])
    else:
        opt = torch.optim.AdamW(groups, lr=s["LR0"], weight_decay=s["WEIGHT_DECAY"])
    steps_per_ep = max(1, len(train_items) // BATCH) if len(train_items) > BATCH else 1
    total = s["EPOCHS"] * steps_per_ep
    warm = min(s.get("WARMUP_ITERS", 0), max(1, total // 10)) if s.get("WARMUP_ITERS") else 0

    def lr_lambda(i):
        if i < warm:
            return 0.001 + 0.999 * i / warm
        x = (i - warm) / max(1, total - warm)
        if s["SCHED"] == "poly":
            return (1 - min(x, 1)) ** 0.9
        return 1 - (1 - s["LRF"]) * x                       # 線性（同 B1m / Ultralytics）

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    history, best = [], -1.0
    print(f"\n{'epoch':>5} {'lr':>9} {'loss':>8} {'Mask_mAP50-95':>14} {'fitness':>8}")
    for ep in range(s["EPOCHS"]):
        t0 = time.time()
        w.train(True)
        order = rng.permutation(len(train_items))
        sums, n = {}, 0
        for b in range(steps_per_ep):
            idx = order[b * BATCH:(b + 1) * BATCH]
            imgs, mlist = [], []
            for i in idx:
                it = train_items[i]
                img = imread_rgb(it["img_path"])
                H, W = img.shape[:2]
                masks = load_instances(it["label_path"], H, W)
                img, masks = augment(img, masks, rng, **AUG)
                masks = [m for m in drop_small(masks) if (mask_box(m)[2:] - mask_box(m)[:2]).min() >= 2]
                imgs.append(img)
                mlist.append(masks)
            loss, parts = w.loss(imgs, mlist)
            if loss is None:
                continue
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"epoch {ep + 1} loss = {loss.item()}，訓練發散；先把 LR0 減半")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if s.get("CLIP_GRAD"):
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], s["CLIP_GRAD"])
            opt.step()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1

        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"],
                   **{f"train/{k}": v / max(n, 1) for k, v in sums.items()})
        last = ep == s["EPOCHS"] - 1
        mark = ""
        tl = row.get("train/loss", float("nan"))
        if (ep + 1) % s["EVAL_EVERY"] == 0 or last:
            w.train(False)
            m = evaluate(w.predict_instances, val_items)
            row.update(m)
            if m["fitness"] > best:
                best = m["fitness"]
                torch.save(w.state_dict(), P["best"])
                mark = " ★"
            print(f"{ep + 1:>5} {row['lr']:>9.2e} {tl:>8.4f} {m['Mask_mAP50-95']:>14.4f}"
                  f" {m['fitness']:>8.4f}{mark}")
        else:
            print(f"{ep + 1:>5} {row['lr']:>9.2e} {tl:>8.4f}")
        row["sec"] = round(time.time() - t0, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(P["results"], index=False, encoding="utf-8-sig")

    torch.save(w.state_dict(), P["last"])
    finish_weights(work_dir)
    print("\n=== val 評估（best.pt，與 B1 同定義的 Box/Mask mAP）===")
    m = evaluate(build_backend(P["ready"], s.get("DEVICE")).predict_instances, val_items,
                   progress=True)
    write_summary(work_dir, train_items, val_items, poly_df, m, s["MODEL_NAME"])


def print_header(doc, s):
    print(doc)
    print(f"待執行設定：{json.dumps(s['CFG'], ensure_ascii=False, default=str)}\n")


if __name__ == "__main__":
    print_header(__doc__, SETTINGS)
    run(SETTINGS, D2Wrapper)
