"""
A4d_build_noclahe_dataset.py
============================================
把 A4b 產生的「378 張無 CLAHE 裁切圖」，挑出實際被標註的那 308 張，
改成與 label 對得起來的檔名，照原資料集的結構輸出成一份可直接訓練的無 CLAHE 資料集。

輸入三份：
  SEG_DATASET_DIR    已標註的資料集（Roboflow 匯出，images/ + labels/，可含 train/valid/test）
  CLAHE_CROP_DIR     A4 的裁切圖（378 張，CLAHE 版）  ← 只用來「認親」與驗證
  NOCLAHE_CROP_DIR   A4b 的輸出（378 張，無 CLAHE 版，檔名與上面一致） ← 真正要輸出的內容

對每一張資料集影像 D：
  1. 認親：找出 D 是哪一張裁切圖
     a. 檔名反推：剝掉 Roboflow 的 `_jpg.rf.<hash>`
     b. 內容比對：縮圖 NCC（先裁掉 letterbox 黑邊），反推不出來時使用
  2. 求變換 T：D 可能被 Roboflow 翻轉/旋轉（擴增）、縮放或補邊過
     候選 = 8 種翻轉旋轉 × {同尺寸 / letterbox 到 D 的尺寸 / 拉伸到 D 的尺寸}
     用「T(CLAHE裁切) vs D」的 MAE 選最好的，不夠像就不輸出
  3. 輸出 T(無CLAHE裁切)，檔名與 D **完全相同**，放在與 D 相同的相對路徑
  4. 把 D 對應的 label 複製到鏡像位置；data.yaml 等非影像檔一併複製

結果：OUTPUT_DIR 是一份與原資料集「同結構、同檔名、同 label、只差沒做 CLAHE」的資料集，
      可以直接拿去訓練，唯一變因就是前處理。

⚠️ 驗證不過的不輸出，並列進「未還原清單.txt」。
   那幾張在 CLAHE 組也要一起剔除，否則兩組張數不一致，比較就不乾淨。

用法：
    python A4d_build_noclahe_dataset.py
    python A4d_build_noclahe_dataset.py <已標註資料集> <CLAHE裁切圖> <無CLAHE裁切圖> <輸出>

安裝需求：
    pip install opencv-python numpy pandas
"""

import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ============================================================
# 設定區
# ============================================================
SEG_DATASET_DIR = "test-77_CLAHE"          # 👈 已標註資料集根目錄
CLAHE_CROP_DIR = "data set/cropped_test_CLAHE"  # 👈 A4 的 378 張（CLAHE 版）
NOCLAHE_CROP_DIR = "data set/cropped_test"               # 👈 A4b 的輸出（378 張，無 CLAHE 版）
OUTPUT_DIR = "test-77"                 # 👈 輸出（不可已存在）

PASS_MAE = 8.0             # T(CLAHE裁切) vs 資料集影像 的灰階 MAE 門檻
PASS_NCC = 0.97
THUMB = 64                 # 內容比對縮圖邊長
NCC_PASS = 0.90            # 內容比對接受門檻
NCC_MARGIN = 0.04          # 與次佳差距小於此 → 標記區分度低
BLACK_THRESHOLD = 5        # letterbox 黑邊門檻
TOPK_CANDIDATES = 3        # 內容比對取前幾名進入變換搜尋
JPEG_QUALITY = 100
QC_N = 8

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
QC_DIR_NAME = "_qc"
MANIFEST_OUT = "A4d_還原QC.csv"
RF_PATTERN = re.compile(r"^(.*)_(jpg|jpeg|png|bmp|tif|tiff)\.rf\.[0-9a-fA-F]+$")


# ============================================================
# IO
# ============================================================
def imread_u(p, flags=cv2.IMREAD_UNCHANGED):
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)
    except (OSError, ValueError):
        return None


def imwrite_u(p, img, params=()):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".jpg", img, list(params))
    if not ok:
        raise IOError(f"寫不出 {p}")
    buf.tofile(str(p))


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None and img.ndim == 3 else img


def mae_of(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def ncc(a, b):
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def strip_rf(stem):
    m = RF_PATTERN.match(stem)
    return m.group(1) if m else stem


# ============================================================
# 內容比對
# ============================================================
def crop_content(g):
    ys, xs = np.where(g > BLACK_THRESHOLD)
    if len(ys) == 0:
        return g
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _vec(g):
    t = cv2.resize(g, (THUMB, THUMB), interpolation=cv2.INTER_AREA).astype(np.float64).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def thumb_vec(g):
    c = crop_content(g)
    return _vec(g), _vec(c), c.shape[1] / max(1, c.shape[0])


# ============================================================
# 變換候選：8 種翻轉旋轉 × 3 種尺寸對齊
# ============================================================
DIHEDRAL = {
    "id": lambda a: a,
    "flipH": lambda a: a[:, ::-1],
    "flipV": lambda a: a[::-1, :],
    "rot180": lambda a: a[::-1, ::-1],
    "rot90cw": lambda a: a.T[:, ::-1],
    "rot90ccw": lambda a: a.T[::-1, :],
    "transpose": lambda a: a.T,
    "antitranspose": lambda a: a.T[::-1, ::-1],
}


def fit_letterbox(g, W, H, interp):
    h, w = g.shape[:2]
    s = min(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(g, (nw, nh), interpolation=interp)
    o = np.zeros((H, W), np.uint8)
    o[(H - nh) // 2:(H - nh) // 2 + nh, (W - nw) // 2:(W - nw) // 2 + nw] = r
    return o


def size_modes(g, W, H):
    """回傳 [(名稱, 影像)]；尺寸已是 (W,H) 的話只回傳 same。"""
    out = []
    if g.shape[1] == W and g.shape[0] == H:
        out.append(("same", g))
        return out
    interp = cv2.INTER_AREA if (W * H) < (g.shape[1] * g.shape[0]) else cv2.INTER_LINEAR
    out.append(("letterbox", fit_letterbox(g, W, H, interp)))
    out.append(("stretch", cv2.resize(g, (W, H), interpolation=interp)))
    return out


def best_transform(crop_gray, target_gray):
    """回傳 (mae, ncc, dihedral名, 尺寸模式) 或 None。"""
    H, W = target_gray.shape[:2]
    best = None
    for dname, dfn in DIHEDRAL.items():
        d = np.ascontiguousarray(dfn(crop_gray))
        for sname, img in size_modes(d, W, H):
            m = mae_of(img, target_gray)
            if best is None or m < best[0]:
                best = (m, ncc(img, target_gray), dname, sname)
        if best is not None and best[0] <= PASS_MAE and best[2] == "id":
            break      # 原方向就很好，不必再試擴增方向
    return best


def apply_transform(g, dname, sname, W, H):
    d = np.ascontiguousarray(DIHEDRAL[dname](g))
    if sname == "same":
        return d
    interp = cv2.INTER_AREA if (W * H) < (d.shape[1] * d.shape[0]) else cv2.INTER_LINEAR
    return fit_letterbox(d, W, H, interp) if sname == "letterbox" else \
        cv2.resize(d, (W, H), interpolation=interp)


# ============================================================
# label 位置
# ============================================================
def label_path_of(img_path, ds_root):
    c1 = img_path.with_suffix(".txt")
    if c1.exists():
        return c1, img_path.relative_to(ds_root).with_suffix(".txt")
    rel = img_path.relative_to(ds_root)
    parts = ["labels" if x == "images" else x for x in rel.parts]
    c2 = (ds_root / Path(*parts)).with_suffix(".txt")
    return (c2, Path(*parts).with_suffix(".txt")) if c2.exists() else (None, None)


# ============================================================
# 主流程
# ============================================================
def process(ds_dir=SEG_DATASET_DIR, clahe_dir=CLAHE_CROP_DIR,
            noclahe_dir=NOCLAHE_CROP_DIR, out_dir=OUTPUT_DIR):
    ds_root, clahe_dir = Path(ds_dir), Path(clahe_dir)
    noclahe_dir, out_dir = Path(noclahe_dir), Path(out_dir)
    for d in (ds_root, clahe_dir, noclahe_dir):
        if not d.exists():
            raise FileNotFoundError(f"❌ 找不到 {d.resolve()}")
    if out_dir.exists():
        print(f"⚠️ {out_dir} 已存在，為避免新舊混雜直接停止；要重跑請先刪除。")
        return

    ds_imgs = list_images(ds_root)
    clahe_paths = list_images(clahe_dir)
    if not ds_imgs:
        raise FileNotFoundError(f"❌ {ds_root.resolve()} 底下沒有影像")
    if not clahe_paths:
        raise FileNotFoundError(f"❌ {clahe_dir.resolve()} 底下沒有影像")

    print(f"=== A4d：資料集 {len(ds_imgs)} 張，CLAHE 裁切圖 {len(clahe_paths)} 張 ===")
    print(f"    資料集分布：{dict(Counter(str(p.parent.relative_to(ds_root)) for p in ds_imgs))}")

    # 無 CLAHE 版索引（檔名要與 CLAHE 版一致）
    noclahe_by_name = {p.name: p for p in list_images(noclahe_dir)}
    noclahe_by_stem = {p.stem: p for p in list_images(noclahe_dir)}
    print(f"    無 CLAHE 裁切圖 {len(noclahe_by_name)} 張")

    # CLAHE 裁切圖索引
    print("    建立縮圖索引 ...", end="", flush=True)
    crop_by_stem, keys, v_raw, v_cc, ars, grays = {}, [], [], [], [], {}
    for p in clahe_paths:
        g = to_gray(imread_u(p))
        if g is None:
            continue
        crop_by_stem.setdefault(p.stem, p)
        a, b, ar = thumb_vec(g)
        keys.append(p)
        v_raw.append(a)
        v_cc.append(b)
        ars.append(ar)
        grays[p] = g
    M_raw = np.vstack(v_raw) if v_raw else np.zeros((0, THUMB * THUMB))
    M_cc = np.vstack(v_cc) if v_cc else np.zeros((0, THUMB * THUMB))
    ARS = np.array(ars) if ars else np.zeros(0)
    print(f" 完成（{len(keys)} 張）")

    rows, n_qc = [], 0
    for k, ip in enumerate(ds_imgs, 1):
        rel = ip.relative_to(ds_root)
        rec = {"資料集圖片": str(rel), "對應裁切檔": None, "配對方式": None,
               "翻轉": None, "尺寸模式": None, "MAE": None, "NCC": None, "label": "無"}
        color = imread_u(ip)
        g = to_gray(color)
        if g is None:
            rec["狀態"] = "❌ 讀不到資料集影像"
            rows.append(rec)
            continue
        H, W = g.shape[:2]
        rec["資料集尺寸"] = f"{W}x{H}"

        # 1) 認親
        cands = []
        c = crop_by_stem.get(strip_rf(ip.stem)) or crop_by_stem.get(ip.stem)
        if c is not None:
            cands, rec["配對方式"] = [c], "檔名"
        elif keys:
            qa, qb, qar = thumb_vec(g)
            s = np.maximum(M_raw @ qa, M_cc @ qb)
            if ARS.size:
                s = s * (0.85 + 0.15 * np.minimum(ARS, qar) / np.maximum(ARS, qar))
            o = np.argsort(-s)
            if float(s[o[0]]) >= NCC_PASS:
                cands = [keys[j] for j in o[:TOPK_CANDIDATES]]
                rec["配對方式"] = "內容比對"
                if len(o) > 1 and float(s[o[0]]) - float(s[o[1]]) < NCC_MARGIN:
                    rec["配對方式"] = "內容比對⚠️區分度低"
                rec["縮圖NCC"] = round(float(s[o[0]]), 4)
        if not cands:
            rec["狀態"] = "❌ 認不出是哪一張裁切圖"
            rows.append(rec)
            continue

        # 2) 求變換（在候選中選最好的）
        best, best_c = None, None
        for c in cands:
            r = best_transform(grays[c], g)
            if r is not None and (best is None or r[0] < best[0]):
                best, best_c = r, c
            if best is not None and best[0] <= PASS_MAE:
                break
        if best is None:
            rec["狀態"] = "❌ 找不到可用的變換"
            rows.append(rec)
            continue
        mae, sc, dname, sname = best
        rec.update({"對應裁切檔": best_c.name, "翻轉": dname, "尺寸模式": sname,
                    "MAE": round(mae, 2), "NCC": round(sc, 5) if sc == sc else None})
        if not (mae <= PASS_MAE and (sc != sc or sc >= PASS_NCC)):
            rec["狀態"] = f"❌ 驗證未過(MAE={mae:.2f}, NCC={sc:.5f})，未輸出"
            rows.append(rec)
            continue

        # 3) 套同一個變換到「無 CLAHE 版」
        np_path = noclahe_by_name.get(best_c.name) or noclahe_by_stem.get(best_c.stem)
        if np_path is None:
            rec["狀態"] = f"❌ 無 CLAHE 版找不到同名檔：{best_c.name}"
            rows.append(rec)
            continue
        ng = to_gray(imread_u(np_path))
        if ng is None:
            rec["狀態"] = f"❌ 無 CLAHE 版讀取失敗：{np_path.name}"
            rows.append(rec)
            continue
        if ng.shape != grays[best_c].shape:
            rec["狀態"] = (f"❌ 無 CLAHE 版尺寸與 CLAHE 版不同"
                           f"（{ng.shape[1]}x{ng.shape[0]} vs "
                           f"{grays[best_c].shape[1]}x{grays[best_c].shape[0]}）")
            rows.append(rec)
            continue
        out_img = apply_transform(ng, dname, sname, W, H)
        if color is not None and color.ndim == 3:
            out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
        imwrite_u(out_dir / rel, out_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        # 4) label
        lp, lrel = label_path_of(ip, ds_root)
        if lp is not None:
            (out_dir / lrel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lp, out_dir / lrel)
            rec["label"] = "已複製"
        rec["狀態"] = "✅"
        rows.append(rec)

        if n_qc < QC_N:
            ref_t = apply_transform(grays[best_c], dname, sname, W, H)
            tiles, titles = [g, ref_t, apply_transform(ng, dname, sname, W, H)], \
                            ["1 dataset (CLAHE)", "2 CLAHE crop -> T", "3 output (no CLAHE)"]
            vis = []
            for im, t in zip(tiles, titles):
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(im, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)
                vis.append(im)
            vis = np.hstack(vis)
            cv2.putText(vis, f"MAE={mae:.2f} {dname}/{sname} src={best_c.name}",
                        (5, vis.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
            imwrite_u(out_dir / QC_DIR_NAME / f"{Path(rel).stem}.png", vis)
            n_qc += 1

        if k % 50 == 0:
            print(f"   ... {k}/{len(ds_imgs)}")

    # 5) 非影像檔（data.yaml 等）一併複製
    for p in sorted(ds_root.rglob("*")):
        if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS and p.suffix.lower() != ".txt":
            dst = out_dir / p.relative_to(ds_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / MANIFEST_OUT, index=False, encoding="utf-8-sig")
    st = df["狀態"].astype(str)
    n_ok, n_bad = int((st == "✅").sum()), int(st.str.startswith("❌").sum())
    used = sorted(set(df.loc[st == "✅", "對應裁切檔"].dropna()))
    unused = sorted(set(p.name for p in clahe_paths) - set(used))
    (out_dir / "已使用裁切清單.txt").write_text("\n".join(used), encoding="utf-8")
    (out_dir / "未使用裁切清單.txt").write_text("\n".join(unused), encoding="utf-8")

    print("\n=======================================================")
    print(f"✨ A4d 完成：資料集 {len(df)} 張")
    print(f"   ✅ 已輸出: {n_ok}（檔名與原資料集完全相同，label 已複製）")
    print(f"   ❌ 未輸出: {n_bad}")
    print(f"   對應到 {len(used)} 個不重複裁切檔；裁切圖 {len(clahe_paths)} 張中未被用到 {len(unused)}")
    if n_ok:
        ok = df[st == "✅"]
        print(f"   驗證：MAE 中位數 {ok['MAE'].median():.2f}、最大 {ok['MAE'].max():.2f}")
        print(f"   配對方式：{dict(ok['配對方式'].astype(str).value_counts())}")
        print(f"   翻轉分布：{dict(ok['翻轉'].astype(str).value_counts())}")
        print(f"   尺寸模式：{dict(ok['尺寸模式'].astype(str).value_counts())}")
        nl = int((ok['label'] != '已複製').sum())
        if nl:
            print(f"   ⚠️ 有 {nl} 張找不到對應 label（背景圖？）")
    if n_bad:
        bad = df.loc[st.str.startswith("❌"), "資料集圖片"]
        (out_dir / "未還原清單.txt").write_text("\n".join(map(str, bad)), encoding="utf-8")
        print(f"   👉 未輸出清單：{out_dir / '未還原清單.txt'}")
        print(f"      ⚠️ CLAHE 組也要剔除同樣這些檔案，兩組張數一致才能比")
        for s_, c in df.loc[st.str.startswith("❌"), "狀態"].astype(str).str[:60]\
                .value_counts().head(3).items():
            print(f"        {c:>4} 筆  {s_}")
    print(f"📍 輸出資料集: {out_dir}（結構/檔名/label 與原資料集相同，只差沒做 CLAHE）")
    print(f"📍 QC 三聯圖: {out_dir / QC_DIR_NAME}   明細: {out_dir / MANIFEST_OUT}")
    print("=======================================================")
    print("👉 訓練前先抽看 QC 圖：中間那張（CLAHE 裁切經變換後）要跟左邊資料集影像一模一樣。")
    print("👉 兩組只差前處理，seg 要各自重訓；評估看 B3 長度誤差 / C1 MAE 的跨 seed mean±SD。")


def main():
    a = sys.argv[1:]
    if not a:
        process()
    elif len(a) == 4:
        process(*a)
    else:
        print("用法:")
        print(f"  python {Path(__file__).name}")
        print(f"  python {Path(__file__).name} <已標註資料集> <CLAHE裁切圖> <無CLAHE裁切圖> <輸出>")
        sys.exit(1)


if __name__ == "__main__":
    main()
