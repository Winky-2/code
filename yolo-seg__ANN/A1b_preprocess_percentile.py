"""
A1b_preprocess_percentile.py
============================================
獨立版前處理腳本（跟 A1 同位置、同用法）：從「原始未增強」的圖出發，
輸出同檔名的處理後圖片。預設 MODE = "percentile"（百分位拉伸）。

*** 為什麼標註不用重做 ***
所有模式都只改灰階值，不改像素位置（不縮放、不裁切、不平移）。
所以 Roboflow 上已經標在 CLAHE 圖上的多邊形，套到這裡的輸出圖仍然對得上。
→ 你可以直接用同檔名的新圖取代舊圖，labels 不動。

⚠️ 輸入一定要是「原始未增強」的圖，不要疊在已 CLAHE 的圖上（雙重增強）。
⚠️ train / test / 未來推論必須用同一個 MODE 與同一組參數。
⚠️ 評估請看 B3 長度誤差與 C1 MAE（跨 5 seed 的 mean ± SD），不要只看 mask mAP。

*** 模式 ***
  "percentile" : 百分位線性拉伸（P_LO~P_HI → 0~255），只算有訊號的像素   ← 預設
  "gamma"      : 百分位拉伸後再做 gamma（<1 提亮暗部）
  "usm"        : 百分位拉伸後再做 unsharp masking
  "clahe"      : 與 A1 相同的 CLAHE（當對照組）
  "multi"      : 三通道組合 → 通道0(B)=USM、通道1(G)=CLAHE、通道2(R)=百分位拉伸
                 （YOLO 吃 3 通道，灰階複製 3 份是浪費；注意輸出是彩色圖，
                  一旦用了 multi，train/test/推論都必須是 multi）

黑邊（灰階 <= BLACK_BORDER_THRESHOLD）一律維持原值，不參與統計、不被改動。

用法：
    python A1b_preprocess_percentile.py                       # 用設定區
    python A1b_preprocess_percentile.py <輸入資料夾> <輸出資料夾>
    python A1b_preprocess_percentile.py --preview <輸入資料夾> <預覽輸出資料夾>
        # 預覽：每張圖輸出 original + 全部模式，並排比較

安裝需求：
    pip install opencv-python numpy pandas
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ============================================================
# 設定區
# ============================================================
INPUT_DIR = "test_single_seg-67_no_resize_noCLAHE/test/images"          # 👈 原始、未增強的圖
OUTPUT_DIR = "test_A1b_multi"      # 👈 輸出（同檔名）

MODE = "multi"              # "percentile" | "gamma" | "usm" | "clahe" | "multi"

BLACK_BORDER_THRESHOLD = 5       # 與 A1 / enhance_image 相同
P_LO, P_HI = 1.0, 99.0           # 百分位拉伸的下/上百分位

GAMMA = 0.7                      # <1 提亮暗部（gamma 模式）
USM_SIGMA = 2.0                  # usm 模式
USM_AMOUNT = 1.0
CLAHE_CLIP_LIMIT = 1.5           # 與 A1 一致
CLAHE_TILE_GRID_SIZE = (8, 8)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OVERWRITE_EXISTING = False
MANIFEST_CSV = "前處理清單_百分位.csv"
PREVIEW_MAX_IMAGES = 5


# ============================================================
# 影像處理（全部輸入/輸出灰階 uint8，黑邊不動）
# ============================================================

def _signal_mask(gray):
    return gray > BLACK_BORDER_THRESHOLD


def percentile_stretch(gray, p_lo=P_LO, p_hi=P_HI):
    """回傳 (stretched, lo, hi)。訊號像素太少或 hi<=lo 時原樣回傳。"""
    mask = _signal_mask(gray)
    if mask.sum() < 100:
        return gray.copy(), None, None
    lo, hi = np.percentile(gray[mask], [p_lo, p_hi])
    if hi <= lo:
        return gray.copy(), float(lo), float(hi)
    out = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0, 1) * 255.0
    res = gray.copy()
    res[mask] = np.round(out[mask]).astype(np.uint8)
    return res, float(lo), float(hi)


def apply_gamma(gray, gamma=GAMMA):
    mask = _signal_mask(gray)
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).round().astype(np.uint8)
    res = gray.copy()
    res[mask] = lut[gray[mask]]
    return res


def apply_usm(gray, sigma=USM_SIGMA, amount=USM_AMOUNT):
    mask = _signal_mask(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharp = cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)
    res = gray.copy()
    res[mask] = sharp[mask]
    return res


def apply_clahe(gray):
    """與 A1 的 enhance_image 同邏輯：只在非黑邊的 bounding box 內做。"""
    ys, xs = np.where(_signal_mask(gray))
    if len(ys) == 0:
        return gray.copy()
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    res = gray.copy()
    res[y0:y1, x0:x1] = clahe.apply(gray[y0:y1, x0:x1])
    return res


def process(img, mode=MODE):
    """回傳 (輸出影像, lo, hi)。輸入可為灰階或 BGR。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lo = hi = None

    if mode == "percentile":
        out, lo, hi = percentile_stretch(gray)
    elif mode == "gamma":
        out, lo, hi = percentile_stretch(gray)
        out = apply_gamma(out)
    elif mode == "usm":
        out, lo, hi = percentile_stretch(gray)
        out = apply_usm(out)
    elif mode == "clahe":
        out = apply_clahe(gray)
    elif mode == "multi":
        pct, lo, hi = percentile_stretch(gray)
        out = cv2.merge([apply_usm(pct), apply_clahe(gray), pct])   # B, G, R
        return out, lo, hi
    else:
        raise ValueError(f"未知 MODE：{mode!r}")

    # 非 multi 模式：維持與輸入相同 channel 數（YOLO 讀進來都會是 3 通道）
    if img.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out, lo, hi


def write_image(path, img):
    if path.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(str(path), img)


# ============================================================
# 量產 / 預覽
# ============================================================

def collect_images(d):
    return sorted(p for p in Path(d).glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def process_all(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"❌ 找不到輸入資料夾 {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_images(in_dir)
    if not paths:
        raise FileNotFoundError(f"❌ {in_dir} 底下沒有圖片")

    print(f"=== 前處理 MODE={MODE}：{len(paths)} 張 ===")
    if MODE in ("percentile", "gamma", "usm", "multi"):
        print(f"    百分位 {P_LO}~{P_HI}，黑邊門檻 {BLACK_BORDER_THRESHOLD}")

    records, n_ok = [], 0
    for p in paths:
        dst = out_dir / p.name
        if dst.exists() and not OVERWRITE_EXISTING:
            records.append({"檔名": p.name, "狀態": "⏭️ 已存在，跳過"})
            continue
        img = cv2.imread(str(p))
        if img is None:
            records.append({"檔名": p.name, "狀態": "❌ 讀取失敗"})
            print(f"   ❌ {p.name}：讀取失敗")
            continue
        out, lo, hi = process(img)
        write_image(dst, out)
        n_ok += 1
        rec = {"檔名": p.name, "狀態": "✅", "MODE": MODE,
               "lo": lo, "hi": hi, "寬": img.shape[1], "高": img.shape[0]}
        if lo is None or (hi is not None and hi <= lo):
            rec["狀態"] = "⚠️ 訊號不足或 hi<=lo，已原樣輸出"
        records.append(rec)

    mf = out_dir / MANIFEST_CSV
    pd.DataFrame(records).to_csv(mf, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(records)
    n_warn = int(df["狀態"].astype(str).str.startswith("⚠️").sum())

    print("\n=======================================================")
    print(f"✨ 完成：成功 {n_ok} 張，警告 {n_warn} 張")
    print(f"📍 輸出：{out_dir}（同檔名，可直接取代原圖，labels 不用動）")
    print(f"📍 清單：{mf}（含每張的 lo/hi，可檢查各張灰階範圍差多少）")
    print("=======================================================")
    print("⚠️ train / test / 推論要用同一個 MODE 與參數。")
    print("⚠️ 輸入必須是原始未增強圖。換前處理後，seg 要重訓，不能沿用舊權重。")


def preview(src_dir, dst_dir):
    src, dst = Path(src_dir), Path(dst_dir)
    if not src.exists():
        raise FileNotFoundError(f"❌ 找不到 {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for p in collect_images(src)[:PREVIEW_MAX_IMAGES]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(dst / f"{p.stem}_original.png"), img)
        for m in ("percentile", "gamma", "usm", "clahe", "multi"):
            out, _, _ = process(img, m)
            cv2.imwrite(str(dst / f"{p.stem}_{m}.png"), out)
        print(f"✅ {p.name}")
    print(f"\n📍 預覽輸出：{dst}")
    print("   肉眼看根尖附近有沒有出現假邊緣或雜訊顆粒，再事先決定要跑哪幾種。")


def main():
    a = sys.argv[1:]
    if a and a[0] == "--preview":
        if len(a) != 3:
            print(f"用法: python {Path(__file__).name} --preview <輸入資料夾> <預覽輸出資料夾>")
            sys.exit(1)
        preview(a[1], a[2])
    elif len(a) == 2:
        process_all(a[0], a[1])
    elif not a:
        process_all()
    else:
        print(f"用法: python {Path(__file__).name} [<輸入資料夾> <輸出資料夾> | --preview <輸入> <輸出>]")
        sys.exit(1)


if __name__ == "__main__":
    main()