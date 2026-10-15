"""
B2u_unet_inference.py
===============================================================
UNet 在 test 集推論，並與 YOLO11-seg 在「同一批圖、同一套指標」逐張對照。

為什麼不直接比 B2 的 mAP：mAP 是 instance 指標（要依信心度排序），UNet 是
語意分割、沒有 instance 也沒有信心度，兩者不可比。這裡改用兩邊都能算的：
  - IoU / Dice：挑出的目標牙 vs GT（test 只標目標牙）
  - 長度誤差(px，原圖空間)：mask 沿主軸的延伸長度，pred − GT。
    比 IoU 更接近真正在乎的根尖位置；但這是簡化定義，不保證等同 B3 方法3，
    定案數字仍以 B3 為準。

挑目標牙（兩模型都不看 GT，規則一致）：
  - UNet：前景機率 > THRESH → 最大連通區
  - YOLO：conf ≥ YOLO_CONF 的 mask 取面積最大者，剔除碎塊（同 B2 v3）
另報 UNet_oracle：與 GT 重疊最多的連通區——只用來分辨
「分割本身不好」vs「挑錯牙」，不能當成績。

產出：results/B2u/<run_tag>/  逐張對照.xlsx、視覺化/（綠=GT 紅=UNet 黃=YOLO）、
      UNet_mask/（原圖空間二值 mask，之後接 B3 用）
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from _unet_common import (IMAGE_EXTENSIONS, PAD_VALUE, best_overlap_cc, imread_rgb, imwrite,
                          iou_dice, label_to_mask, largest_cc, letterbox, pca_length,
                          to_input, unletterbox)

# ==================== 可手動修改 ====================
RUN_NOW = True

UNET_RUN_DIR = "results/B1u/unet_resnet34_img640_e150_s42"
UNET_WEIGHTS = "best.pt"
IMAGE_DIR = "test_single_seg-67/test/images"        # 👈 與 B2/B3 相同
LABEL_DIR = "test_single_seg-67/test/labels"
THRESH = 0.5

YOLO_WEIGHTS = "yolo11n_seg_run/weights_ready.pt"   # 👈 設 None 就不比 YOLO
YOLO_IMG_SIZE = 640
YOLO_CONF = 0.15                                    # 同 B2 的 SEG_CONF

OUT_ROOT = "results/B2u"
SAVE_VISUALIZATION = True
WRONG_TOOTH_GAP = 0.3     # oracle_IoU − IoU 超過這個 → 判定「挑錯牙/黏牙」
# ===================================================


def new_out_dir(tag):
    """只新增不覆寫：已存在就加 _2、_3…"""
    base = Path(OUT_ROOT) / tag
    d, k = base, 2
    while d.exists():
        d, k = Path(f"{base}_{k}"), k + 1
    d.mkdir(parents=True)
    return d


def yolo_target_mask(res, H, W):
    """conf ≥ YOLO_CONF 的 mask → 各自剔除碎塊（開運算+最大連通區）→ 取面積最大者。"""
    best = np.zeros((H, W), bool)
    if res.masks is None:
        return best
    confs = res.boxes.conf.tolist() if res.boxes is not None else []
    for i, poly in enumerate(res.masks.xy):
        if poly is None or len(poly) < 3 or (i < len(confs) and confs[i] < YOLO_CONF):
            continue
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m, _ = largest_cc(m)
        if m.sum() > best.sum():
            best = m
    return best


def draw(img_rgb, gt, unet, yolo, txt):
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    t = max(2, img.shape[1] // 400)
    for m, color in ((gt, (0, 255, 0)), (unet, (0, 0, 255)), (yolo, (0, 220, 255))):
        if m is not None and m.any():
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, color, t)
    s = max(0.6, img.shape[1] / 900)
    (tw, th), b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, s, t)
    cv2.rectangle(img, (0, 0), (tw + 16, th + b + 16), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 8 + th), cv2.FONT_HERSHEY_SIMPLEX, s, (255, 255, 255), t,
                cv2.LINE_AA)
    return img


def summarize(df, col):
    v = df[col].dropna()
    return f"{v.mean():.4f} ± {v.std():.4f}（中位數 {v.median():.4f}，n={len(v)}）"


def main():
    import pandas as pd
    import torch

    from _unet_common import build_model

    run_dir = Path(UNET_RUN_DIR)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    img_size = cfg["IMG_SIZE"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg["ENCODER"], None)
    model.load_state_dict(torch.load(run_dir / UNET_WEIGHTS, map_location=device))
    model.to(device).eval()

    yolo = None
    if YOLO_WEIGHTS:
        from ultralytics import YOLO
        yolo = YOLO(YOLO_WEIGHTS)

    images = sorted(p for p in Path(IMAGE_DIR).iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"{IMAGE_DIR} 裡沒有影像")
    out_dir = new_out_dir(run_dir.name)
    print(f"UNet：{run_dir / UNET_WEIGHTS}（{cfg['ENCODER']}, img {img_size}）")
    print(f"YOLO：{YOLO_WEIGHTS or '不比較'}")
    print(f"test：{len(images)} 張 → {out_dir}\n")

    rows = []
    for p in images:
        img = imread_rgb(p)
        H, W = img.shape[:2]
        gt, n_gt = label_to_mask(Path(LABEL_DIR) / f"{p.stem}.txt", H, W)
        has_gt = n_gt > 0

        lb, meta = letterbox(img, img_size, PAD_VALUE)
        with torch.no_grad():
            x = torch.from_numpy(to_input(lb))[None].to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        fg = unletterbox(prob, meta, H, W) > THRESH
        unet, n_cc = largest_cc(fg)
        imwrite(out_dir / "UNet_mask" / f"{p.stem}.png", unet.astype(np.uint8) * 255)

        ymask = None
        if yolo is not None:
            res = yolo.predict(source=str(p), imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF,
                               save=False, verbose=False)[0]
            ymask = yolo_target_mask(res, H, W)

        r = {"圖片檔名": p.name, "影像尺寸": f"{W}x{H}", "有GT": has_gt,
             "UNet前景連通區數": n_cc}
        if has_gt:
            gl = pca_length(gt)
            r["GT長度px"] = gl
            r["UNet_IoU"], r["UNet_Dice"] = iou_dice(unet, gt)
            r["UNet_長度誤差px"] = pca_length(unet) - gl
            r["UNet_oracle_IoU"], _ = iou_dice(best_overlap_cc(fg, gt), gt)
            if ymask is not None:
                r["YOLO_IoU"], r["YOLO_Dice"] = iou_dice(ymask, gt)
                r["YOLO_長度誤差px"] = pca_length(ymask) - gl
        rows.append(r)

        if SAVE_VISUALIZATION:
            txt = (f"UNet IoU {r.get('UNet_IoU', np.nan):.3f}"
                   + (f" | YOLO {r.get('YOLO_IoU', np.nan):.3f}" if ymask is not None else ""))
            imwrite(out_dir / "視覺化" / f"{p.stem}.jpg",
                    draw(img, gt if has_gt else None, unet, ymask, txt))

    df = pd.DataFrame(rows)
    for c in ("UNet_長度誤差px", "YOLO_長度誤差px"):
        if c in df:
            df[c.replace("誤差", "絕對誤差")] = df[c].abs()
    df["UNet挑錯牙"] = (df.get("UNet_oracle_IoU", np.nan) - df.get("UNet_IoU", np.nan)
                    ) > WRONG_TOOTH_GAP

    # ---- 摘要 ----
    lines = [f"test {len(df)} 張，有 GT {int(df['有GT'].sum())} 張"]
    for name in ("UNet", "YOLO"):
        if f"{name}_IoU" not in df:
            continue
        lines += [f"[{name}] IoU  {summarize(df, f'{name}_IoU')}",
                  f"[{name}] Dice {summarize(df, f'{name}_Dice')}",
                  f"[{name}] |長度誤差|px {summarize(df, f'{name}_長度絕對誤差px')}",
                  f"[{name}] 長度誤差(bias)px {summarize(df, f'{name}_長度誤差px')}"]
    if "UNet_oracle_IoU" in df:
        lines.append(f"[UNet oracle] IoU {summarize(df, 'UNet_oracle_IoU')}  ← 診斷用上限")
        lines.append(f"UNet 挑錯牙/黏牙：{int(df['UNet挑錯牙'].sum())} 張"
                     f"（oracle−IoU > {WRONG_TOOTH_GAP}）")
    if "YOLO_IoU" in df:
        both = df.dropna(subset=["UNet_IoU", "YOLO_IoU"])
        win = int((both["UNet_IoU"] > both["YOLO_IoU"]).sum())
        d_len = (both["UNet_長度絕對誤差px"] - both["YOLO_長度絕對誤差px"]).dropna()
        lines.append(f"逐張配對：UNet IoU 較高 {win}/{len(both)} 張；"
                     f"|長度誤差| 差(UNet−YOLO) 平均 {d_len.mean():+.1f}px，"
                     f"中位數 {d_len.median():+.1f}px（負 = UNet 較好）")
    print("\n".join(lines))

    with pd.ExcelWriter(out_dir / "逐張對照.xlsx", engine="openpyxl") as w:
        df.to_excel(w, sheet_name="逐張", index=False)
        pd.DataFrame({"摘要": lines}).to_excel(w, sheet_name="摘要", index=False)
    json.dump({"UNET_RUN_DIR": str(run_dir), "UNET_WEIGHTS": UNET_WEIGHTS,
               "IMAGE_DIR": IMAGE_DIR, "LABEL_DIR": LABEL_DIR, "THRESH": THRESH,
               "YOLO_WEIGHTS": YOLO_WEIGHTS, "YOLO_CONF": YOLO_CONF},
              open(out_dir / "config.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n📄 {out_dir / '逐張對照.xlsx'}")
    print("⚠️ n≈67 且單次訓練，差距小於幾個百分點不要下結論；看視覺化確認根尖那一端。")


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)
