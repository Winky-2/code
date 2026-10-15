"""
B1u_train_unet.py
===============================================================
UNet 對照組：把 B1 的 YOLO11n-seg 換成 UNet，其餘條件盡量不動。

刻意改變的變因：模型（YOLO11n-seg 實例分割 → UNet 語意分割）
沿用 B1：資料夾、train/val 切分(seed 42、val 0.2)、IMG_SIZE 640、AdamW、
        LR0 1e-3、BATCH 4、EPOCHS 150、AMP 關、無 early stopping、
        不做左右翻轉、旋轉 ±10°/平移 5%/縮放 10%
無法完全對齊（解讀結果時要記得）：
  - mosaic：UNet 不做（B1 是 0.3）
  - 學習率：線性衰減到 LR0×0.01（仿 Ultralytics 預設 lrf），無 warmup
  - 預訓練：encoder 是 ImageNet resnet34；YOLO11n-seg 是 COCO。都有預訓練但來源不同
  - UNet 是語意分割：一張圖一張前景機率圖，不分 instance。訓練集若標多顆牙，
    相鄰牙可能黏成一塊，要靠 B2u 的連通區挑目標牙

前處理：直接吃 Roboflow 匯出的圖（已是 A1 CLAHE 後的版本），這裡不再做 CLAHE。

操作：
1. pip install segmentation-models-pytorch opencv-python pandas
   torch 依 CUDA 版本到 pytorch.org 取指令安裝
2. 先直接執行（RUN_NOW=False）：只印設定 + 核對 val 切分是否與 YOLO 相同
3. 確認後改 RUN_NOW=True 再執行
產出：results/B1u/<run_tag>/  best.pt、last.pt、history.csv、config.json、split.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from _unet_common import (PAD_VALUE, check_split_against_yolo, collect_seg_dataset,
                          dataset_fingerprint, imread_rgb, label_to_mask, letterbox,
                          random_affine, split_train_val, to_input)

# ==================== 可手動修改 ====================
RUN_NOW = True

SEG_DATA_DIR = "train_single_seg-268"                  # 👈 與 B1 同一個資料夾
YOLO_VAL_DIR = "yolo11n_seg_run/data/val/images"       # 👈 B1 實際用的 val，用來核對切分
OUT_ROOT = "results/B1u"

ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# 以下對齊 B1，不要單獨調整
EPOCHS = 150
IMG_SIZE = 640
VAL_RATIO = 0.2
RANDOM_SEED = 42
LR0 = 1e-3
LRF = 0.01
BATCH = 4
AMP = False
AUG = dict(degrees=10, translate=0.05, scale=0.1)

NUM_WORKERS = 0          # 資料增強用單一 rng，>0 會讓各 worker 增強重複；Windows 也較穩
THRESH = 0.5             # val Dice 的前景閾值
# ===================================================

RUN_TAG = f"unet_{ENCODER}_img{IMG_SIZE}_e{EPOCHS}_s{RANDOM_SEED}"


def make_cfg():
    return dict(SEG_DATA_DIR=SEG_DATA_DIR, ENCODER=ENCODER, ENCODER_WEIGHTS=ENCODER_WEIGHTS,
                EPOCHS=EPOCHS, IMG_SIZE=IMG_SIZE, VAL_RATIO=VAL_RATIO,
                RANDOM_SEED=RANDOM_SEED, LR0=LR0, LRF=LRF, BATCH=BATCH, AMP=AMP,
                AUG=AUG, THRESH=THRESH, RUN_TAG=RUN_TAG)


def prepare_split():
    pool = collect_seg_dataset(SEG_DATA_DIR)
    if not pool:
        raise FileNotFoundError(f"在 {SEG_DATA_DIR} 找不到任何已標註資料")
    train_items, val_items = split_train_val(pool, VAL_RATIO, RANDOM_SEED)
    print(f"📊 pool {len(pool)} 張 → train {len(train_items)} / val {len(val_items)}")
    check_split_against_yolo(val_items, YOLO_VAL_DIR)
    return train_items, val_items


def train(train_items, val_items):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    from _unet_common import build_model

    out_dir = Path(OUT_ROOT) / RUN_TAG
    if (out_dir / "best.pt").exists():
        print(f"⚠️ {out_dir} 已有訓練結果，不覆寫。要重跑請改設定(run tag)或先移走該資料夾")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  device = {device}")
    if device == "cpu":
        print("   ⚠️ 用 CPU 訓練 150 epoch 會非常慢，確認 torch 有裝 CUDA 版")

    class SegDS(Dataset):
        def __init__(self, items, augment):
            self.items, self.augment = items, augment
            self.rng = np.random.default_rng(RANDOM_SEED)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            it = self.items[i]
            img = imread_rgb(it["img_path"])
            mask, _ = label_to_mask(it["label_path"], *img.shape[:2])
            img, _ = letterbox(img, IMG_SIZE, PAD_VALUE)
            mask, _ = letterbox(mask, IMG_SIZE, 0, interp=0)   # 0 = INTER_NEAREST
            if self.augment:
                img, mask = random_affine(img, mask, self.rng, **AUG)
            return (torch.from_numpy(to_input(img)),
                    torch.from_numpy(mask[None].astype(np.float32)))

    dl_tr = DataLoader(SegDS(train_items, True), batch_size=BATCH, shuffle=True,
                       num_workers=NUM_WORKERS, drop_last=len(train_items) > BATCH)
    dl_va = DataLoader(SegDS(val_items, False), batch_size=BATCH, shuffle=False,
                       num_workers=NUM_WORKERS)

    model = build_model(ENCODER, ENCODER_WEIGHTS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 1 - (1 - LRF) * e / max(1, EPOCHS - 1))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP and device == "cuda")

    def loss_fn(logits, y):
        bce = F.binary_cross_entropy_with_logits(logits, y)
        p = torch.sigmoid(logits)
        inter = (p * y).sum((1, 2, 3))
        den = p.sum((1, 2, 3)) + y.sum((1, 2, 3))
        return bce + (1 - (2 * inter + 1) / (den + 1)).mean()

    json.dump({**make_cfg(), "device": device, "n_train": len(train_items),
               "n_val": len(val_items),
               "train_fingerprint": dataset_fingerprint(train_items),
               "val_fingerprint": dataset_fingerprint(val_items)},
              open(out_dir / "config.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump({"train": [it["img_id"] for it in train_items],
               "val": [it["img_id"] for it in val_items]},
              open(out_dir / "split.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    history, best = [], -1.0
    print(f"\n{'epoch':>5} {'lr':>9} {'train_loss':>10} {'val_loss':>9} {'val_Dice':>9}")
    for ep in range(EPOCHS):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=AMP and device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item() * len(x)
        tr_loss /= len(dl_tr.dataset)

        model.eval()
        va_loss, dices = 0.0, []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                va_loss += loss_fn(logits, y).item() * len(x)
                pred = (torch.sigmoid(logits) > THRESH).float()
                inter = (pred * y).sum((1, 2, 3))
                den = pred.sum((1, 2, 3)) + y.sum((1, 2, 3))
                dices += torch.where(den > 0, 2 * inter / den.clamp(min=1),
                                     torch.ones_like(den)).tolist()
        va_loss /= len(dl_va.dataset)
        va_dice = float(np.mean(dices))

        lr = opt.param_groups[0]["lr"]
        sched.step()
        history.append(dict(epoch=ep + 1, lr=lr, train_loss=tr_loss,
                            val_loss=va_loss, val_dice=va_dice, sec=time.time() - t0))
        mark = ""
        if va_dice > best:
            best = va_dice
            torch.save(model.state_dict(), out_dir / "best.pt")
            mark = " ★"
        print(f"{ep + 1:>5} {lr:>9.2e} {tr_loss:>10.4f} {va_loss:>9.4f} {va_dice:>9.4f}{mark}")

    torch.save(model.state_dict(), out_dir / "last.pt")
    import pandas as pd
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")

    print(f"\n✅ 完成：best val Dice = {best:.4f}")
    print(f"📍 {out_dir}")
    print("   ⚠️ val Dice 只代表訓練有沒有學起來，也不能直接跟 YOLO 的 Mask mAP 比。")
    print("      兩者同指標的對照請跑 B2u_unet_inference.py。")
    return out_dir


if __name__ == "__main__":
    print(__doc__)
    print(f"待執行設定：{json.dumps(make_cfg(), ensure_ascii=False)}\n")
    tr, va = prepare_split()
    if RUN_NOW:
        train(tr, va)
    else:
        print("\nRUN_NOW=False：只檢查，未訓練。")
