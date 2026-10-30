"""
A7_preprocess_variants.py
============================
前處理消融實驗用：一次把同一份「已標註資料集」產出多個前處理變體，
label 原封不動沿用，一個變體 = 一個變因。

*** 為什麼 label 可以直接沿用 ***
這裡所有變體都只改像素強度（LUT / 濾波 / 通道堆疊），完全不動幾何。
每個像素位置一一對應 → 多邊形、關鍵點座標不用改。
兩組之間唯一的差別 = 前處理（單一變因）。

*** 建議的輸入 ***
INPUT_DIR 請用「沒有 CLAHE 的乾淨底圖」，也就是 A6_build_noclahe_twin.py 的輸出：
    A6_noclahe_twin/<run_tag>/   (裡面有 images/ labels/)
若直接餵已經 CLAHE 過的資料集，等於在 CLAHE 上再疊一層 → 變因不乾淨，
腳本會偵測並警告（靠直方圖特徵判斷，不是 100% 可靠，只是提醒）。

⚠️ 與 A5 的順序：請對 A5（補邊+縮放）之前的資料跑本腳本，
   再對每個變體各自重跑 A5。先 A5 再換前處理不影響幾何，但黑邊統計會被污染。

*** 黑邊處理 ***
letterbox 640 的黑邊會污染百分位/CLAHE 統計。本腳本自動偵測內容區 bbox，
只在內容區做處理，黑邊原樣保留。

產出：OUTPUT_ROOT/<run_tag>/<variant>/images, labels, (data.yaml)
      OUTPUT_ROOT/<run_tag>/_qc/<variant>_<stem>.png     左=底圖 右=變體
      OUTPUT_ROOT/<run_tag>/A7_manifest.xlsx, config.json
      ※ 只新增不覆寫：同一個 run_tag 已存在就停止

安裝需求：
    pip install opencv-python pandas openpyxl numpy
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ==================== 可手動修改 ====================
RUN_NOW = True  # False = 只列印設定，True = 執行

# 輸入：含 images/ labels/ 的資料集目錄（建議用 A6 的無 CLAHE 雙胞胎）
INPUT_DIR = "test_single_seg-67_no_resize_noCLAHE/test"          # 👈 原始、未增強的圖

OUTPUT_ROOT = "A7_prep_variants"
RUN_TAG = ""                 # 留空 = 由 INPUT_DIR 自動產生

# 要產出的變體（註解掉就不跑）。raw 是基準，建議永遠留著
VARIANTS = [
    "raw",              # 不處理（基準線）
    "pct",              # 百分位拉伸 1-99%
    # --- CLAHE 掃參：先掃這個，通常比換方法有用 ---
    "clahe_c1_t8",
    "clahe_c2_t8",      # ≈ 現行 A1 的設定
    "clahe_c3_t8",
    "clahe_c2_t4",
    "clahe_c2_t16",
    # --- 去噪 → CLAHE ---
    "bil_clahe",        # bilateral
    "nlm_clahe",        # non-local means（慢）
    "aniso_clahe",      # anisotropic diffusion (Perona-Malik)
    # --- 照度/形態學 ---
    "illum_clahe",      # 背景照度校正 → CLAHE
    "tophat",           # top-hat + black-hat 對比增強
    # --- 銳化 ---
    "unsharp",
    "clahe_unsharp",    # ≈ 現行 A1 + A1b
    "gamma08",
    # --- 三通道堆疊（不是灰階複製三份）---
    "stack3",           # [pct, bil+clahe, unsharp]
]

BLACK_BORDER_THRESHOLD = 10   # <= 此灰階視為 letterbox 黑邊，不參與統計、不被處理
JPEG_QUALITY = 95
QC_N = 6                      # 每個變體存幾張 QC 疊圖
COPY_DATA_YAML = True         # 把輸入資料集同層/上層的 data.yaml 一起複製

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# 基本運算元（全部吃 uint8 單通道、回傳 uint8 單通道）
# ============================================================

def op_pct(g, lo=1.0, hi=99.0):
    a, b = np.percentile(g, [lo, hi])
    return np.clip((g.astype(np.float32) - a) * 255.0 / (b - a + 1e-6), 0, 255).astype(np.uint8)


def op_clahe(g, clip=2.0, tile=8):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(g)


def op_bilateral(g, d=5, sc=50, ss=50):
    return cv2.bilateralFilter(g, d, sc, ss)


def op_nlm(g, h=7):
    return cv2.fastNlMeansDenoising(g, None, h, 7, 21)


def op_aniso(g, n_iter=10, kappa=15.0, lam=0.15):
    """Perona-Malik 保邊擴散。"""
    f = g.astype(np.float32)
    for _ in range(n_iter):
        deltas = [np.roll(f, s, axis=a) - f for a, s in ((0, -1), (0, 1), (1, -1), (1, 1))]
        f = f + lam * sum(np.exp(-(d / kappa) ** 2) * d for d in deltas)
    return np.clip(f, 0, 255).astype(np.uint8)


def op_illum(g, sigma=60.0):
    """除以低頻背景，壓掉曝光不均。"""
    f = g.astype(np.float32)
    bg = cv2.GaussianBlur(f, (0, 0), sigma)
    return np.clip(f / (bg + 1e-6) * float(bg.mean()), 0, 255).astype(np.uint8)


def op_tophat(g, k=61):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se)
    return cv2.add(cv2.subtract(g, bh), th)


def op_unsharp(g, sigma=2.0, amount=1.5):
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)


def op_gamma(g, gm=0.8):
    lut = np.clip(((np.arange(256) / 255.0) ** gm) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(g, lut)


# ============================================================
# 變體配方：name -> (說明, 函式 g->單通道 or 3通道)
# ============================================================

def _chain(*fns):
    def f(g):
        for fn in fns:
            g = fn(g)
        return g
    return f


def _stack3(g):
    return cv2.merge([op_pct(g),
                      op_clahe(op_bilateral(g)),
                      op_unsharp(g)])


RECIPES = {
    "raw":           ("不處理（基準）", lambda g: g),
    "pct":           ("百分位拉伸 1-99%", op_pct),
    "clahe_c1_t8":   ("CLAHE clip=1.0 tile=8", lambda g: op_clahe(g, 1.0, 8)),
    "clahe_c2_t8":   ("CLAHE clip=2.0 tile=8（≈現行 A1）", lambda g: op_clahe(g, 2.0, 8)),
    "clahe_c3_t8":   ("CLAHE clip=3.0 tile=8", lambda g: op_clahe(g, 3.0, 8)),
    "clahe_c2_t4":   ("CLAHE clip=2.0 tile=4", lambda g: op_clahe(g, 2.0, 4)),
    "clahe_c2_t16":  ("CLAHE clip=2.0 tile=16", lambda g: op_clahe(g, 2.0, 16)),
    "bil_clahe":     ("bilateral → CLAHE(2,8)", _chain(op_bilateral, lambda g: op_clahe(g, 2.0, 8))),
    "nlm_clahe":     ("NLM 去噪 → CLAHE(2,8)", _chain(op_nlm, lambda g: op_clahe(g, 2.0, 8))),
    "aniso_clahe":   ("anisotropic diffusion → CLAHE(2,8)", _chain(op_aniso, lambda g: op_clahe(g, 2.0, 8))),
    "illum_clahe":   ("照度校正 → CLAHE(2,8)", _chain(op_illum, lambda g: op_clahe(g, 2.0, 8))),
    "tophat":        ("top-hat + black-hat 對比增強", op_tophat),
    "unsharp":       ("unsharp mask σ=2 amount=1.5", op_unsharp),
    "clahe_unsharp": ("CLAHE(2,8) → unsharp（≈A1+A1b）", _chain(lambda g: op_clahe(g, 2.0, 8), op_unsharp)),
    "gamma08":       ("gamma 0.8", op_gamma),
    "stack3":        ("3通道堆疊 [pct, bil+clahe, unsharp]", _stack3),
}


# ============================================================
# 工具
# ============================================================

def make_run_tag():
    return RUN_TAG or Path(INPUT_DIR).as_posix().strip("/").replace("/", "__")


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def content_bbox(g, thr):
    """回傳 letterbox 黑邊以內的內容區 (y0, y1, x0, x1)；找不到就整張。"""
    m = g > thr
    rows, cols = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return 0, g.shape[0], 0, g.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def apply_variant(img_bgr, fn, thr):
    """只在內容區做處理，黑邊原樣保留。輸出一律 3 通道（灰階變體 = 複製三份）。"""
    g = to_gray(img_bgr)
    y0, y1, x0, x1 = content_bbox(g, thr)
    sub = g[y0:y1, x0:x1]
    out_sub = fn(sub)
    if out_sub.ndim == 2:
        out_sub = cv2.cvtColor(out_sub, cv2.COLOR_GRAY2BGR)
    base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    base[y0:y1, x0:x1] = out_sub
    return base, (y0, y1, x0, x1)


def looks_clahe_already(g, thr):
    """粗略判斷底圖是否已經做過直方圖均化類處理（不可靠，只用來提醒）。"""
    v = g[g > thr]
    if v.size == 0:
        return False
    h = np.bincount(v, minlength=256)[thr + 1:].astype(np.float64)
    p = h / h.sum()
    flatness = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()) / 256.0)   # 正規化熵
    span = float((v.max() - v.min()) / 255.0)
    return flatness > 0.72 and span > 0.92


def draw_labels(vis, lbl_path, w, h, x_offsets, color=(0, 255, 0)):
    if not lbl_path.exists():
        return
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        t = line.split()
        if len(t) < 7:
            continue
        v = [float(x) for x in t[1:]]
        if len(v) % 2 == 0 and len(v) >= 6:
            pts = (np.array(v).reshape(-1, 2) * [w, h]).round().astype(np.int32)
            for dx in x_offsets:
                cv2.polylines(vis, [pts + [dx, 0]], True, color, 1)


# ============================================================
# 主流程
# ============================================================

def run():
    import pandas as pd

    in_dir = Path(INPUT_DIR)
    img_dir, lbl_dir = in_dir / "images", in_dir / "labels"
    for d in (img_dir, lbl_dir):
        if not d.exists():
            raise FileNotFoundError(f"❌ 找不到 {d}（INPUT_DIR 底下要有 images/ 與 labels/）")

    unknown = [v for v in VARIANTS if v not in RECIPES]
    if unknown:
        raise ValueError(f"❌ 未定義的變體：{unknown}；可用：{sorted(RECIPES)}")

    run_dir = Path(OUTPUT_ROOT) / make_run_tag()
    if run_dir.exists():
        print(f"⚠️ {run_dir} 已存在，為避免覆寫舊結果直接停止；要重跑請先刪除或改 RUN_TAG。")
        return
    qc_dir = run_dir / "_qc"
    qc_dir.mkdir(parents=True)

    tpl_paths = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not tpl_paths:
        raise FileNotFoundError(f"❌ {img_dir} 裡沒有影像")

    # 底圖健檢（抽 5 張）
    probe = [to_gray(cv2.imread(str(p))) for p in tpl_paths[:5]]
    n_flat = sum(looks_clahe_already(g, BLACK_BORDER_THRESHOLD) for g in probe if g is not None)
    print(f"=== A7：{len(tpl_paths)} 張底圖 × {len(VARIANTS)} 個變體 ===")
    print(f"    底圖：{in_dir}")
    if n_flat >= 3:
        print("    ⚠️ 底圖直方圖看起來已經被均化過（可能已做 CLAHE）。")
        print("       若是如此，以下變體等於疊第二層前處理，變因不乾淨。")
        print("       建議改用 A6_build_noclahe_twin.py 的輸出當底圖。（此判斷不可靠，僅提醒）")

    yaml_src = None
    if COPY_DATA_YAML:
        for cand in (in_dir / "data.yaml", in_dir.parent / "data.yaml"):
            if cand.exists():
                yaml_src = cand
                break

    records = []
    for vi, vname in enumerate(VARIANTS, 1):
        desc, fn = RECIPES[vname]
        v_img, v_lbl = run_dir / vname / "images", run_dir / vname / "labels"
        v_img.mkdir(parents=True)
        v_lbl.mkdir(parents=True)
        if yaml_src:
            shutil.copy2(yaml_src, run_dir / vname / "data.yaml")

        print(f"\n[{vi}/{len(VARIANTS)}] {vname} — {desc}")
        n_qc, n_ok = 0, 0
        for i, tp in enumerate(tpl_paths, 1):
            img = cv2.imread(str(tp))
            if img is None:
                records.append({"變體": vname, "圖片檔名": tp.name, "狀態": "❌ 讀不到"})
                continue
            out, bbox = apply_variant(img, fn, BLACK_BORDER_THRESHOLD)
            params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if tp.suffix.lower() in (".jpg", ".jpeg") else []
            cv2.imwrite(str(v_img / tp.name), out, params)

            lp = lbl_dir / (tp.stem + ".txt")
            if lp.exists():
                shutil.copy2(lp, v_lbl / lp.name)

            g0, g1 = to_gray(img), to_gray(out)
            y0, y1, x0, x1 = bbox
            c0, c1 = g0[y0:y1, x0:x1], g1[y0:y1, x0:x1]
            records.append({
                "變體": vname, "說明": desc, "圖片檔名": tp.name,
                "內容區_wh": f"{x1 - x0}x{y1 - y0}",
                "底圖_mean": round(float(c0.mean()), 1), "變體_mean": round(float(c1.mean()), 1),
                "底圖_std": round(float(c0.std()), 1), "變體_std": round(float(c1.std()), 1),
                # 梯度能量：apex 邊界清晰度的粗略代理指標（越高不一定越好，過高可能是放大雜訊）
                "底圖_gradmag": round(float(np.abs(cv2.Laplacian(c0, cv2.CV_32F)).mean()), 2),
                "變體_gradmag": round(float(np.abs(cv2.Laplacian(c1, cv2.CV_32F)).mean()), 2),
                "label": "有" if lp.exists() else "無（背景圖）",
                "狀態": "✅",
            })
            n_ok += 1

            if n_qc < QC_N:
                h, w = g0.shape
                vis = cv2.cvtColor(np.hstack([g0, g1]), cv2.COLOR_GRAY2BGR)
                draw_labels(vis, lp, w, h, (0, w))
                cv2.imwrite(str(qc_dir / f"{vname}__{tp.stem}_左底圖_右{vname}.png"), vis)
                n_qc += 1

            if i % 50 == 0:
                print(f"   ... {i}/{len(tpl_paths)}")
        print(f"   ✅ {n_ok}/{len(tpl_paths)} 張 → {v_img}")

    df = pd.DataFrame(records)
    df.to_excel(run_dir / "A7_manifest.xlsx", index=False)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "INPUT_DIR": INPUT_DIR, "OUTPUT_ROOT": OUTPUT_ROOT, "run_tag": make_run_tag(),
            "VARIANTS": VARIANTS,
            "recipes": {v: RECIPES[v][0] for v in VARIANTS},
            "BLACK_BORDER_THRESHOLD": BLACK_BORDER_THRESHOLD,
            "JPEG_QUALITY": JPEG_QUALITY,
            "n_images": len(tpl_paths),
            "base_looks_already_equalized": bool(n_flat >= 3),
            "time": datetime.now().isoformat(timespec="seconds"),
        }, f, ensure_ascii=False, indent=2)

    print("\n=======================================================")
    print(f"✨ A7 完成：{len(VARIANTS)} 個變體 × {len(tpl_paths)} 張")
    print(f"📍 輸出: {run_dir}/<variant>/images, labels（label 逐檔相同）")
    print(f"📍 QC 疊圖: {qc_dir}（左=底圖、右=變體、同一組多邊形）")
    print(f"📍 統計: {run_dir}/A7_manifest.xlsx")
    print("=======================================================")
    print("👉 下一步：先看 QC 疊圖，排掉明顯爆掉的變體，再對剩下的各自跑 B1 訓練")
    print("👉 n 小，單一 seed 的差異多半在雜訊內 → 用 B34_run_seeds_v2 跑多 seed 取平均±std")
    print("👉 判斷標準用 apex 點誤差 / 最終 mm MAE，不要用 seg mAP")
    if "stack3" in VARIANTS:
        print("👉 stack3 是真 3 通道：訓練時不要再做灰階化或 per-channel 的色彩擴增")


if __name__ == "__main__":
    if RUN_NOW:
        run()
    else:
        print(__doc__)
        print(f"待執行設定：INPUT_DIR={INPUT_DIR}")
        print(f"            變體({len(VARIANTS)})={VARIANTS}")
        print(f"            run_tag={make_run_tag()}")
        print("👉 確認後把 RUN_NOW 改成 True 再執行")