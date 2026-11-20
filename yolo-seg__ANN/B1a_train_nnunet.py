"""
B1n_train_nnunet.py
===============================================================
分割模型對照 ①：nnU-Net v2（2D）。把 B1 的 YOLO11n-seg 換成 nnU-Net，其餘沿用 B1。

nnU-Net 是語意分割（每個像素一個類別），沒有「一顆一顆」的 instance。
訓練集每張標了好幾顆牙，相鄰牙會黏成一塊 → 加一個「邊界帶」類別：
    0 背景 / 1 牙齒內部 / 2 每顆牙輪廓往內 BORDER_PX 的邊界帶
推論：類別 1 的連通區 = 一顆牙，往外膨脹 BORDER_PX（限前景內）補回輪廓。

刻意改變的變因：模型（YOLO11n-seg → nnU-Net 2D）＋ nnU-Net 自動決定的配方
沿用 B1：資料夾、train/val 切分（seed 42、val 0.2，寫進 splits_final.json 當 fold 0）、不翻轉
無法對齊（解讀結果時要記得）：
  - 前處理、patch 大小、batch、優化器（SGD 0.01 poly）、增強（旋轉/縮放/彈性/亮度…）
    全由 nnU-Net 自動決定 —— 這本來就是 nnU-Net 方法的一部分
  - epoch：nnU-Net 1 epoch = 250 iteration，跟 YOLO 的 epoch 不等價；預設 1000，這裡用 EPOCHS
  - best checkpoint 由 nnU-Net 依 val 的 EMA pseudo-Dice 挑（不是 mAP fitness）
  - 鏡像增強關掉（自動產生 nnUNetTrainer_segx_e{EPOCHS}，繼承官方 nnUNetTrainerNoMirroring）

操作（PowerShell）：
1. 安裝：先依 CUDA 版本裝 torch（pytorch.org），再
       pip install nnunetv2 ultralytics pandas openpyxl pyyaml opencv-python
2. 本檔與 B1_train_yolo11-seg.py 放同一資料夾（用來核對切分；找不到只會略過）
3. 先直接執行（RUN_NOW=False）：只印設定、核對切分與 YOLO 相同
4. 確認後改 RUN_NOW=True
⚠️ 會在 nnunetv2 套件的 training/nnUNetTrainer/ 底下寫一個 trainer 檔（nnU-Net 只認那裡的 class）
產出：nnunet2d_run_<DATA_TAG>/ 結構同 B1（weights_ready.pt、訓練摘要.xlsx、tooth_seg/…）
      另有 nnUNet_raw/、nnUNet_preprocessed/、nnUNet_results/、病例對照.csv

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
SEG_DATA_DIR = "padded_train-308"            # 👈 與 B1_train_yolo11-seg.py 相同
DATA_TAG = "train-308"                       # WORK_DIR 名稱用：{模型}_run_{DATA_TAG}
YOLO_WORK_DIR = "yolo11n_seg_run_train-308"  # 👈 B1 的 WORK_DIR（核對切分、SAM 的框來源）
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


# ==================== 可手動修改 ====================
RUN_NOW = True                 # False = 只檢查設定、切分、標註品質，未訓練

EPOCHS = 150                 # nnU-Net 預設 1000（約數小時～一天）；250 是常見的縮短版
BORDER_PX = 3                # 邊界帶寬度（px）；太寬會吃掉細根尖，太窄黏牙分不開
DATASET_ID = 501
DATASET_NAME = "ToothSeg"
NUM_PROC = 4                 # 前處理用的行程數；Windows 記憶體不夠就調小
DEVICE = "cuda"              # "cuda" | "cpu"（CPU 只能做冒煙測試）
ITERS_PER_EPOCH = None       # None = nnU-Net 預設 250；冒煙測試才改小
MIN_INST_AREA = 50
# ===================================================

ARCH = "segx_nnunet2d"
MODEL_TAG = "nnunet2d"
WORK_DIR = work_dir_of(MODEL_TAG)
TRAINER = f"nnUNetTrainer_segx_e{EPOCHS}" + (f"_it{ITERS_PER_EPOCH}" if ITERS_PER_EPOCH else "")
DS_FOLDER = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
MODEL_DIR = f"nnUNet_results/{DS_FOLDER}/{TRAINER}__nnUNetPlans__2d"


def make_cfg():
    return dict(EPOCHS=EPOCHS, BORDER_PX=BORDER_PX, DATASET_ID=DATASET_ID, TRAINER=TRAINER,
                NNUNET_MODEL_DIR=MODEL_DIR, ITERS_PER_EPOCH=ITERS_PER_EPOCH,
                MIN_INST_AREA=MIN_INST_AREA, DEVICE=DEVICE)


# ============================================================
# 第1部分：轉成 nnU-Net 資料集（不需要 torch）
# ============================================================

def instance_label(masks, H, W):
    """多個 instance → 0/1/2 標籤圖。重疊處也算邊界。"""
    import cv2
    lab = np.zeros((H, W), np.uint8)
    cnt = np.zeros((H, W), np.uint8)
    border = np.zeros((H, W), bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BORDER_PX + 1,) * 2)
    for m in masks:
        cnt += m
        er = cv2.erode(m.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        border |= m & ~er.astype(bool)
    lab[cnt > 0] = 1
    lab[border | (cnt > 1)] = 2
    return lab


def build_nnunet_dataset(train_items, val_items):
    import cv2
    raw = Path(WORK_DIR) / "nnUNet_raw" / DS_FOLDER
    if raw.exists():
        shutil.rmtree(raw)
    rows, train_ids, val_ids = [], [], []
    for i, it in enumerate(train_items + val_items):
        case = f"tooth_{i:04d}"
        img = imread_rgb(it["img_path"])
        H, W = img.shape[:2]
        imwrite(raw / "imagesTr" / f"{case}_0000.png", cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        imwrite(raw / "labelsTr" / f"{case}.png",
                  instance_label(load_instances(it["label_path"], H, W), H, W))
        is_val = i >= len(train_items)
        (val_ids if is_val else train_ids).append(case)
        rows.append({"case": case, "img_id": it["img_id"], "split": "val" if is_val else "train"})
    (raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "xray"},
        "labels": {"background": 0, "tooth": 1, "border": 2},
        "numTraining": len(rows), "file_ending": ".png"}, indent=2), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(rows).to_csv(Path(WORK_DIR) / "病例對照.csv", index=False, encoding="utf-8-sig")
    print(f"🗂️  nnU-Net 資料集：{raw}（train {len(train_ids)} / val {len(val_ids)}）")
    return train_ids, val_ids


# ============================================================
# 第2部分：nnU-Net 指令
# ============================================================

def nn_env():
    wd = Path(WORK_DIR).resolve()
    env = os.environ.copy()
    env.update(nnUNet_raw=str(wd / "nnUNet_raw"),
               nnUNet_preprocessed=str(wd / "nnUNet_preprocessed"),
               nnUNet_results=str(wd / "nnUNet_results"))
    os.environ.update({k: env[k] for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")})
    return env


def run_entry(module, func, args):
    import subprocess
    cmd = [sys.executable, "-c", f"from {module} import {func} as f; f()", *map(str, args)]
    print(f"▶ {func} {' '.join(map(str, args))}")
    subprocess.run(cmd, env=nn_env(), check=True)


def write_trainer():
    """nnU-Net 會用 locals() 讀 __init__ 的具名參數 → 不能用 *args，照抄目前版本的簽名。"""
    import inspect

    import nnunetv2
    import torch
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

    params = list(inspect.signature(nnUNetTrainer.__init__).parameters.values())[1:]

    def fmt(p):
        if p.default is inspect.Parameter.empty:
            return p.name
        d = p.default
        return f"{p.name}=" + (f"torch.device('{d.type}')" if isinstance(d, torch.device) else repr(d))

    dst = Path(nnunetv2.__file__).parent / "training" / "nnUNetTrainer" / f"{TRAINER}.py"
    iters = (f"\n        self.num_iterations_per_epoch = {ITERS_PER_EPOCH}"
             f"\n        self.num_val_iterations_per_epoch = {ITERS_PER_EPOCH}") if ITERS_PER_EPOCH else ""
    dst.write_text(
        "import torch\n"
        "from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring "
        "import nnUNetTrainerNoMirroring\n\n\n"
        f"class {TRAINER}(nnUNetTrainerNoMirroring):\n"
        f'    """由 B1n_train_nnunet.py 自動產生：不翻轉、num_epochs={EPOCHS}"""\n\n'
        f"    def __init__(self, {', '.join(fmt(p) for p in params)}):\n"
        f"        super().__init__({', '.join(p.name for p in params)})\n"
        f"        self.num_epochs = {EPOCHS}{iters}\n", encoding="utf-8")
    print(f"🧩 trainer：{dst}")


def write_splits(train_ids, val_ids):
    p = Path(WORK_DIR) / "nnUNet_preprocessed" / DS_FOLDER / "splits_final.json"
    p.write_text(json.dumps([{"train": train_ids, "val": val_ids}], indent=2), encoding="utf-8")
    print(f"✂️  splits_final.json：fold 0 = B1 的 train/val")


def export_weights_and_log():
    import pandas as pd
    import torch
    md = Path(WORK_DIR) / MODEL_DIR / "fold_0"
    P = run_paths(WORK_DIR)
    P["weights"].mkdir(parents=True, exist_ok=True)
    shutil.copy2(md / "checkpoint_best.pth", P["best"])
    shutil.copy2(md / "checkpoint_final.pth", P["last"])
    log = torch.load(md / "checkpoint_final.pth", map_location="cpu", weights_only=False)["logging"]
    n = len(log["train_losses"])
    df = pd.DataFrame({"epoch": np.arange(1, n + 1),
                       **{k: v[:n] for k, v in log.items() if len(v) >= n and k != "epoch_start_timestamps"}})
    df.to_csv(P["results"], index=False, encoding="utf-8-sig")


# ============================================================
# 第3部分：主流程
# ============================================================

def main():
    train_items, val_items, poly_df = prepare_split()
    if not RUN_NOW:
        print(f"\nRUN_NOW=False：只檢查，未訓練。輸出會在 {WORK_DIR}/")
        return
    if already_trained(WORK_DIR):
        return
    set_seed(TRAIN_SEED)
    copy_data_like_b1(WORK_DIR, train_items, val_items)
    write_config(WORK_DIR, ARCH, make_cfg(), train_items, val_items)
    train_ids, val_ids = build_nnunet_dataset(train_items, val_items)

    run_entry("nnunetv2.experiment_planning.plan_and_preprocess_entrypoints",
              "plan_and_preprocess_entry",
              ["-d", DATASET_ID, "-c", "2d", "--verify_dataset_integrity", "-np", NUM_PROC])
    write_splits(train_ids, val_ids)
    write_trainer()
    run_entry("nnunetv2.run.run_training", "run_training_entry",
              [DATASET_ID, "2d", 0, "-tr", TRAINER, "-device", DEVICE])

    export_weights_and_log()
    finish_weights(WORK_DIR)
    print("\n=== val 評估（與 B1 同定義的 Box/Mask mAP）===")
    m = evaluate(build_backend(run_paths(WORK_DIR)["ready"], DEVICE).predict_instances,
                   val_items, progress=True)
    write_summary(WORK_DIR, train_items, val_items, poly_df, m, "nnU-Net 2D")


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
# nnU-Net v2（2D）
# ============================================================

class NNUNetSeg:
    """標籤：0 背景 / 1 牙齒內部 / 2 牙齒邊界帶（寬 BORDER_PX）。
    推論：類別 1 的連通區 = 一顆牙 → 往外膨脹 BORDER_PX（限制在前景內）補回邊界。
    分數 = 該區前景機率平均（給 mAP 排序用；跟 YOLO conf 不是同一種量）。"""

    def __init__(self, cfg, work_dir, device=None):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        self.border = int(cfg["BORDER_PX"])
        self.min_area = int(cfg.get("MIN_INST_AREA", 50))
        dev = torch.device(pick_device(device))
        self.p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                 perform_everything_on_device=dev.type == "cuda",
                                 device=dev, verbose=False, verbose_preprocessing=False,
                                 allow_tqdm=False)
        self.p.initialize_from_trained_model_folder(str(Path(work_dir) / cfg["NNUNET_MODEL_DIR"]),
                                                    use_folds=(0,),
                                                    checkpoint_name="checkpoint_best.pth")

    def predict_instances(self, img_rgb, conf, key=None):
        g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[None, None]
        seg, prob = self.p.predict_single_npy_array(g, {"spacing": [999.0, 1.0, 1.0]},
                                                    None, None, True)
        seg, prob = seg[0], prob[:, 0]
        fg = seg > 0
        pfg = prob[1:].sum(0)
        n, lab = cv2.connectedComponents((seg == 1).astype(np.uint8), connectivity=8)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * self.border + 1,) * 2)
        masks, scores = [], []
        for i in range(1, n):
            comp = lab == i
            if comp.sum() < self.min_area:
                continue
            inst = cv2.dilate(comp.astype(np.uint8), k).astype(bool) & fg
            s = float(pfg[inst].mean())
            if s >= conf:
                masks.append(inst)
                scores.append(s)
        return masks, np.asarray(scores)


def build_backend(path, device=None):
    """nnU-Net 推論器：讀 WORK_DIR/config.json + weights_ready.pt。"""
    cfg = read_cfg(path)
    if not cfg or cfg.get("ARCH") != ARCH:
        raise ValueError(f"{path} 不是 {ARCH} 的權重（WORK_DIR 裡要有 config.json）")
    return NNUNetSeg(cfg, run_dir_of(path), device)


if __name__ == "__main__":
    print(__doc__)
    print(f"待執行設定：{json.dumps(make_cfg(), ensure_ascii=False)}\n")
    main()
