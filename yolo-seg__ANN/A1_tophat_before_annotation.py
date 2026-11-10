"""
A1t_tophat_before_annotation.py
============================================
獨立版前處理腳本（跟 A1 / A1b 同位置、同用法）：從「原始未增強」的圖出發，
用形態學 top-hat / black-hat 做對比增強，輸出同檔名的處理後圖片。

*** 為什麼標註不用重做 ***
只改灰階值，不改像素位置（不縮放、不裁切、不平移）。
已經標在 CLAHE 圖上的多邊形，套到這裡的輸出圖仍然對得上 → labels 不動。

⚠️ 輸入一定要是「原始未增強」的圖，不要疊在已 CLAHE 的圖上（雙重增強）。
⚠️ train / test / 未來推論必須用同一個 MODE 與同一組參數。
⚠️ 換前處理後 seg 要重訓，不能沿用舊權重。
⚠️ 評估請看 B3 長度誤差與 C1 MAE（跨 5 seed 的 mean ± SD），不要只看 mask mAP。

*** top-hat 在做什麼（決定 KERNEL_SIZE 用） ***
  opening(g, SE)  = 把「比 SE 小的亮結構」磨掉
  top-hat    TH = g - opening(g)      → 只留下比 SE 小的亮細節
  black-hat  BH = closing(g) - g      → 只留下比 SE 小的暗細節
  輸出 out = g + STRENGTH*(TH - BH)   → 小於 SE 的結構被加強對比

所以 **KERNEL_SIZE 越大 → 被當成「細節」的結構越多 → 增強越兇、越容易飽和**。
K=61 在 640 圖上已經是很激進的一端。牙根尖端是逐漸變窄、對比又低的區域，
K 太大時那裡的灰階會被拉到 0 或 255 飽和，原本標示根尖位置的漸層被抹掉，
seg 看到的是一條人造的假邊緣 → mask mAP 可能照樣很高，但長度會歪掉。

→ 所以本檔案的 manifest 會記錄「飽和像素比例」，preview 會輸出差異圖，
  兩個都是用來判斷 K 有沒有過頭的直接證據。

*** 模式 ***
  "contrast"  : out = g + STRENGTH*(TH - BH)        ← 預設，與先前 tophat 實驗相同(STRENGTH=1)
  "tophat"    : out = TH（純 top-hat，背景全黑，長相差很多，通常不適合直接餵 seg）
  "th_clahe"  : contrast 之後再接 CLAHE（參數與 A1 一致）

黑邊（灰階 <= BLACK_BORDER_THRESHOLD）一律維持原值，不參與統計、不被改動
（與 A1 的 enhance_image 同邏輯：先抓非黑邊 bounding box，只在 ROI 內處理）。

用法：
    python A1t_tophat_before_annotation.py                      # 用設定區
    python A1t_tophat_before_annotation.py <輸入資料夾> <輸出資料夾>
    python A1t_tophat_before_annotation.py --preview <輸入資料夾> <預覽輸出資料夾>
        # 預覽：每張圖輸出 original + 各種 KERNEL_SIZE + 差異圖 + 一張併排 montage

安裝需求：
    pip install opencv-python numpy pandas
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ============================================================
# 第0部分：參數與增強函式
# ============================================================

# ---------------- 形態學參數設定區 ----------------
# ⚠️ 改這裡的值，B1/B2/未來推論那邊若也有一份，務必同步（A1 檔頭那個坑）。
KERNEL_SIZE = 31              # 結構元素邊長(px，奇數)。越大越兇。先用 --preview 決定
KERNEL_SHAPE = "ellipse"      # "ellipse" | "rect" | "cross"
STRENGTH = 1.0                # TH/BH 的加權。1.0 = 先前的 tophat 實驗；<1 較溫和
BLACK_BORDER_THRESHOLD = 5    # 與 A1 / A1b 一致

# th_clahe 模式才會用到，與 A1 一致
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)

MODE = "contrast"             # "contrast" | "tophat" | "th_clahe"

_SHAPES = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}


def _signal_bbox(gray):
    """回傳非黑邊的 bounding box (y0, y1, x0, x1)；整張近乎全黑時回傳 None。"""
    ys, xs = np.where(gray > BLACK_BORDER_THRESHOLD)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _kernel(k=None, shape=None):
    k = KERNEL_SIZE if k is None else k
    k = int(k) | 1                                   # 強制奇數
    return cv2.getStructuringElement(_SHAPES[shape or KERNEL_SHAPE], (k, k))


def _tophat_contrast(roi, se, strength):
    th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, se).astype(np.float32)
    bh = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, se).astype(np.float32)
    out = roi.astype(np.float32) + strength * (th - bh)
    return np.clip(out, 0, 255).astype(np.uint8)


def _clahe_roi(roi):
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    return clahe.apply(roi)


def enhance_image(img, mode=None, k=None, shape=None, strength=None):
    """
    對灰階或 BGR 影像做 top-hat 系列增強，回傳同 channel 數的 uint8 影像。
    輸入為 None 或整張近乎全黑時原樣回傳。
    """
    if img is None:
        return img
    mode = mode or MODE
    strength = STRENGTH if strength is None else strength

    is_color = (img.ndim == 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img.copy()

    bb = _signal_bbox(gray)
    if bb is None:
        return img
    y0, y1, x0, x1 = bb
    roi = gray[y0:y1, x0:x1]
    se = _kernel(k, shape)

    if mode == "contrast":
        roi_out = _tophat_contrast(roi, se, strength)
    elif mode == "tophat":
        roi_out = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, se)
    elif mode == "th_clahe":
        roi_out = _clahe_roi(_tophat_contrast(roi, se, strength))
    else:
        raise ValueError(f"未知 MODE：{mode!r}（可用：contrast / tophat / th_clahe）")

    out_gray = gray.copy()
    out_gray[y0:y1, x0:x1] = roi_out
    return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR) if is_color else out_gray


def roi_stats(gray_in, gray_out):
    """回傳 ROI 內的統計，重點是飽和比例（K 有沒有過頭的直接證據）。"""
    bb = _signal_bbox(gray_in)
    if bb is None:
        return {}
    y0, y1, x0, x1 = bb
    a, b = gray_in[y0:y1, x0:x1], gray_out[y0:y1, x0:x1]
    n = a.size
    return {
        "ROI_wh": f"{x1 - x0}x{y1 - y0}",
        "前_mean": round(float(a.mean()), 1), "後_mean": round(float(b.mean()), 1),
        "前_std": round(float(a.std()), 1), "後_std": round(float(b.std()), 1),
        "前_飽和pct": round(float(((a == 0) | (a == 255)).sum()) * 100 / n, 2),
        "後_飽和pct": round(float(((b == 0) | (b == 255)).sum()) * 100 / n, 2),
        "平均改動量": round(float(np.abs(b.astype(np.float32) - a.astype(np.float32)).mean()), 2),
    }


# ============================================================
# 第1部分：批次前處理設定區
# ============================================================

INPUT_DIR = "test-77/test/images"   # 👈 原始、未增強的圖
OUTPUT_DIR = "test-77_tophat/images"                                   # 👈 輸出（同檔名）

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OVERWRITE_EXISTING = False
MANIFEST_CSV = "前處理清單_tophat.csv"

# 預覽模式要比較的 KERNEL_SIZE 清單
PREVIEW_KERNEL_SIZES = [11, 21, 31, 45, 61]
PREVIEW_MAX_IMAGES = 5
SATURATION_WARN_PCT = 2.0     # 後_飽和pct 超過這個值就在 manifest 標警告


# ============================================================
# 第2部分：批次前處理（量產模式）
# ============================================================

def collect_images(d):
    return sorted(p for p in Path(d).glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def write_image(path, img):
    if path.suffix.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    else:
        cv2.imwrite(str(path), img)


def process_all(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    in_dir, out_dir = Path(input_dir), Path(output_dir)
    if not in_dir.exists():
        raise FileNotFoundError(f"❌ 找不到輸入資料夾 {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_images(in_dir)
    if not paths:
        raise FileNotFoundError(f"❌ {in_dir} 底下沒有找到任何圖片({IMAGE_EXTENSIONS})")

    print(f"=== top-hat 前處理 MODE={MODE}：{len(paths)} 張 ===")
    print(f"    KERNEL_SIZE={KERNEL_SIZE}（{KERNEL_SHAPE}）, STRENGTH={STRENGTH}, "
          f"黑邊門檻={BLACK_BORDER_THRESHOLD}")
    if MODE == "th_clahe":
        print(f"    後接 CLAHE clip={CLAHE_CLIP_LIMIT}, tile={CLAHE_TILE_GRID_SIZE}")

    records, n_ok, n_skip, n_fail, n_warn = [], 0, 0, 0, 0
    for p in paths:
        dst = out_dir / p.name
        if dst.exists() and not OVERWRITE_EXISTING:
            n_skip += 1
            records.append({"檔名": p.name, "狀態": "⏭️ 已存在，跳過"})
            continue
        img = cv2.imread(str(p))
        if img is None:
            n_fail += 1
            records.append({"檔名": p.name, "狀態": "❌ 讀取失敗"})
            print(f"   ❌ {p.name}：讀取失敗，跳過")
            continue

        out = enhance_image(img)
        write_image(dst, out)
        n_ok += 1

        g_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
        st = roi_stats(g_in, g_out)
        rec = {"檔名": p.name, "狀態": "✅", "MODE": MODE, "K": KERNEL_SIZE,
               "shape": KERNEL_SHAPE, "strength": STRENGTH, **st}
        if st.get("後_飽和pct", 0) > SATURATION_WARN_PCT:
            rec["狀態"] = f"⚠️ 飽和 {st['後_飽和pct']}%，K 可能過大"
            n_warn += 1
        records.append(rec)

    df = pd.DataFrame(records)
    mf = out_dir / MANIFEST_CSV
    df.to_csv(mf, index=False, encoding="utf-8-sig")

    print("\n=======================================================")
    print(f"✨ 完成：成功 {n_ok} 張，跳過 {n_skip} 張，失敗 {n_fail} 張")
    if n_warn:
        print(f"   ⚠️ 其中 {n_warn} 張飽和比例 > {SATURATION_WARN_PCT}%，見 manifest")
    if n_ok and "後_飽和pct" in df.columns:
        ok = df[df["狀態"].astype(str).str.startswith(("✅", "⚠️"))]
        print(f"   ROI 飽和比例：前 {ok['前_飽和pct'].median():.2f}% → "
              f"後 {ok['後_飽和pct'].median():.2f}%（中位數）")
        print(f"   ROI 標準差：  前 {ok['前_std'].median():.1f} → 後 {ok['後_std'].median():.1f}")
    print(f"📍 輸出：{out_dir}（同檔名，可直接取代原圖，labels 不用動）")
    print(f"📍 清單：{mf}")
    print("=======================================================")
    print("⚠️ train / test / 推論要用同一個 MODE 與參數。")
    print("⚠️ 換前處理後 seg 要重訓，不能沿用舊權重。")
    print("👉 評估看 B3 長度誤差 / C1 MAE 的跨 seed mean±SD，不要看 mask mAP。")


# ============================================================
# 第3部分：預覽模式（決定 KERNEL_SIZE 用）
# ============================================================

def _diff_map(g_in, g_out):
    """差異圖：顏色越亮代表這個像素被改越多。直接看根尖那塊有沒有被動到。"""
    d = np.abs(g_out.astype(np.float32) - g_in.astype(np.float32))
    d = np.clip(d * (255.0 / max(d.max(), 1.0)), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def _label(img, text):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def preview(src_dir, dst_dir):
    src, dst = Path(src_dir), Path(dst_dir)
    if not src.exists():
        raise FileNotFoundError(f"❌ 找不到 {src}")
    dst.mkdir(parents=True, exist_ok=True)

    paths = collect_images(src)[:PREVIEW_MAX_IMAGES]
    if not paths:
        print(f"⚠️ {src} 底下沒有找到圖片")
        return

    rows = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"⚠️ 讀不到 {p}，跳過")
            continue
        g_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        cv2.imwrite(str(dst / f"{p.stem}_original.png"), img)

        tiles = [_label(g_in, "original")]
        for k in PREVIEW_KERNEL_SIZES:
            out = enhance_image(img, k=k)
            g_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
            cv2.imwrite(str(dst / f"{p.stem}_K{k}.png"), out)
            cv2.imwrite(str(dst / f"{p.stem}_K{k}_diff.png"), _diff_map(g_in, g_out))
            st = roi_stats(g_in, g_out)
            tiles.append(_label(g_out, f"K={k}  sat {st.get('後_飽和pct', 0)}%"))
            rows.append({"檔名": p.name, "K": k, **st})

        cv2.imwrite(str(dst / f"{p.stem}_montage.png"), np.hstack(tiles))
        print(f"✅ {p.name} → original + {len(PREVIEW_KERNEL_SIZES)} 種 K + diff + montage")

    if rows:
        df = pd.DataFrame(rows)
        csv = dst / "preview_飽和統計.csv"
        df.to_csv(csv, index=False, encoding="utf-8-sig")
        print(f"\n=== 各 K 的 ROI 飽和比例（中位數）===")
        print(df.groupby("K")[["後_飽和pct", "後_std", "平均改動量"]].median().round(2).to_string())
        print(f"📍 統計：{csv}")

    print(f"\n📍 預覽輸出：{dst}")
    print("   看 *_montage.png 挑 K；再看 *_K<k>_diff.png 確認「被改最多的地方」")
    print("   是不是正好壓在牙根尖端 —— 如果是，那個 K 就太大了。")
    print("   決定後回來改本檔案最上面的 KERNEL_SIZE，並同步 B1/B2 那邊若有的同一份參數。")


# ============================================================
# 主流程
# ============================================================

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
        print("用法:")
        print(f"  python {Path(__file__).name}                              # 用設定區的資料夾")
        print(f"  python {Path(__file__).name} <輸入資料夾> <輸出資料夾>")
        print(f"  python {Path(__file__).name} --preview <輸入資料夾> <預覽輸出資料夾>")
        sys.exit(1)


if __name__ == "__main__":
    main()
