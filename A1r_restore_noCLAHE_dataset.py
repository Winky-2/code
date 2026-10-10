"""
A1r_restore_noCLAHE_dataset.py
==============================
消融實驗用：把「已標註、經過CLAHE」的資料集，換成「同一幾何位置的原圖(無CLAHE)」版本，
標註檔(labels)原封不動沿用。

*** 為什麼不是「反CLAHE」 ***
CLAHE 不可逆（clip 截斷 + 多對一的直方圖映射 + tile 雙線性插值），
從增強後的圖無法還原原圖。唯一正確做法是：拿回 A1 之前的原始X光片，
算出「原圖 → 資料集圖」的幾何轉換（Roboflow letterbox 縮放、A4 裁切、A5 補黑邊、
甚至 Roboflow 的翻轉/旋轉增強都會被吸收進這個轉換），再把原圖套同一個轉換貼過去。
因為像素位置完全一樣，標註座標可以直接沿用。

*** 做法 ***
1. 對每張原圖重跑 A1 的 enhance_image()（同參數，確定性），得到「重建的CLAHE全圖」
2. SIFT 特徵 + 投票，找出每張資料集圖對應哪張原圖（不依賴檔名，A2/Roboflow 改名也沒關係）
3. RANSAC 估相似轉換（縮放+旋轉+平移，必要時試翻轉），再用 ECC 做次像素精修
4. 用同一個轉換把「原圖」warp 到資料集圖的大小；轉換外的區域(letterbox黑邊)沿用資料集原像素
5. QC：重建CLAHE全圖 warp 後與資料集圖的 NCC，應接近 1；低於門檻的列出來人工檢查

用法：
    python A1r_restore_noCLAHE_dataset.py <原圖資料夾> <CLAHE資料集根目錄> <輸出資料集根目錄>
    # 資料集根目錄底下可有多個 images/ (例如 train/images, valid/images, test/images)，
    # 會自動遞迴找出來，並在輸出端鏡像同樣結構；labels/ 與 data.yaml 直接複製。

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

# ---------------- CLAHE 參數：必須與 A1 完全一致 ----------------
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)
BLACK_BORDER_THRESHOLD = 5

# ---------------- 配對/QC 參數 ----------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SIFT_MAX_SIDE = 1600        # 原圖抽特徵時的最大邊長（只影響速度，轉換會換算回原尺寸）
N_CANDIDATES = 3            # 投票前幾名的原圖拿來做 RANSAC 驗證
RATIO_TEST = 0.75
MIN_INLIERS = 15
MIN_INLIERS_HINT = 6        # 檔名對得上的原圖，門檻放寬（最後仍由 NCC 把關）
NCC_WARN = 0.99             # NCC 低於此值 → 不輸出，移到 _review（實測正確配對都 ≥0.996）
SIFT_CONTRAST = 0.02        # 預設0.04；調低讓小裁切區也有足夠特徵點
REVIEW_DIR = "_review_未還原"
QC_CSV = "A1r_配對QC.csv"


# ============================================================
# A1 的 enhance_image（原樣複製，確保重建的CLAHE圖與標註用圖一致）
# ============================================================
def enhance_gray(gray):
    mask = gray > BLACK_BORDER_THRESHOLD
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return gray
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    out = gray.copy()
    out[y0:y1, x0:x1] = clahe.apply(gray[y0:y1, x0:x1])
    return out


def imread_u(p, flags=cv2.IMREAD_UNCHANGED):
    """Windows 的 cv2.imread 讀不了中文路徑（如「論文」），改用 numpy 讀 bytes 再解碼"""
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), flags)
    except (OSError, ValueError):
        return None


def imwrite_u(p, img, params=()):
    ok, buf = cv2.imencode(Path(p).suffix or ".jpg", img, list(params))
    if not ok:
        raise IOError(f"寫不出 {p}")
    buf.tofile(str(p))


def read_gray(p):
    img = imread_u(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"讀不到 {p}")
    return img


def name_hints(stem):
    """001_d1 / 001_jpg.rf.xxx / 001 → {'001_d1','001'}"""
    s0 = re.split(r"_(?:jpg|jpeg|png|bmp|tif|tiff)\.rf\.", stem)[0]
    hs = {s0, re.sub(r"_d\d+$", "", s0)}
    m = re.match(r"^(\d+)", s0)
    if m:
        hs.add(m.group(1))
    return hs


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


# ============================================================
# 幾何工具
# ============================================================
def to3x3(M):
    return np.vstack([M, [0, 0, 1]])


def flip_matrix(kind, w, h):
    sx, tx = (-1, w - 1) if "h" in kind else (1, 0)
    sy, ty = (-1, h - 1) if "v" in kind else (1, 0)
    return np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=np.float64)


def apply_flip(img, kind):
    if kind == "h":
        return cv2.flip(img, 1)
    if kind == "v":
        return cv2.flip(img, 0)
    if kind == "hv":
        return cv2.flip(img, -1)
    return img


def ncc(a, b, mask):
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    if a.size < 100:
        return np.nan
    a -= a.mean()
    b -= b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


# ============================================================
# 原圖庫：重建CLAHE + SIFT
# ============================================================
class RawLibrary:
    def __init__(self, raw_dir):
        self.sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=SIFT_CONTRAST)
        self.items = []   # dict(path, raw, clahe, kp_xy(full-res), desc)
        all_desc, all_lab = [], []
        raw_dirs = [raw_dir] if isinstance(raw_dir, (str, Path)) else list(raw_dir)
        raw_dirs = [d for x in raw_dirs for d in str(x).split(";") if d.strip()]   # 命令列可用 ; 分隔多個
        paths = sorted({p.resolve() for d in raw_dirs for p in list_images(d.strip())})
        for d in raw_dirs:
            print(f"   原圖來源: {d}  ({len(list_images(d.strip()))} 張)")
        if not paths:
            raise FileNotFoundError(f"❌ {raw_dir} 沒有圖片")
        print(f"=== 建立原圖庫：{len(paths)} 張（重跑A1 CLAHE + SIFT）===")
        for i, p in enumerate(paths):
            raw = read_gray(p)
            cl = enhance_gray(raw)
            s = min(1.0, SIFT_MAX_SIDE / max(cl.shape))
            small = cv2.resize(cl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else cl
            kp, desc = self.sift.detectAndCompute(small, None)
            if desc is None:
                desc = np.zeros((0, 128), np.float32)
            xy = np.array([k.pt for k in kp], np.float32).reshape(-1, 2) / s
            self.items.append(dict(path=p, raw=raw, clahe=cl, xy=xy, desc=desc))
            all_desc.append(desc)
            all_lab.append(np.full(len(desc), i, np.int32))
        self.labels = np.concatenate(all_lab)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
        self.flann.add([np.concatenate(all_desc).astype(np.float32)])
        self.flann.train()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def hinted(self, name_hint):
        hs = name_hints(name_hint)
        return [i for i, it in enumerate(self.items) if it["path"].stem in hs]

    def candidates(self, desc, name_hint):
        """回傳 [(idx, 是否檔名對應)]，檔名對應的排最前面"""
        votes = Counter()
        for m in self.flann.knnMatch(desc, k=2):
            if len(m) == 2 and m[0].distance < RATIO_TEST * m[1].distance:
                votes[int(self.labels[m[0].trainIdx])] += 1
        hint = self.hinted(name_hint)
        return [(i, True) for i in hint] + \
               [(i, False) for i, _ in votes.most_common(N_CANDIDATES) if i not in hint]

    def estimate(self, idx, kp_q, desc_q, min_inl=MIN_INLIERS):
        """回傳 (M 2x3: 原圖full-res座標 → query座標, inliers)"""
        it = self.items[idx]
        if len(it["desc"]) < 2 or desc_q is None or len(desc_q) < 2:
            return None, 0
        good = [m[0] for m in self.bf.knnMatch(desc_q, it["desc"], k=2)
                if len(m) == 2 and m[0].distance < RATIO_TEST * m[1].distance]
        if len(good) < max(min_inl, 3):
            return None, len(good)
        src = np.float32([it["xy"][m.trainIdx] for m in good])
        dst = np.float32([kp_q[m.queryIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0, maxIters=5000)
        n = int(inl.sum()) if inl is not None else 0
        return (M if n >= min_inl else None), n


def refine_ecc(clahe_raw, ds_gray, M, valid):
    """ECC 次像素精修（在資料集圖座標系做殘差校正；失敗就回傳原M）
    ds 座標 x → W(x) 落在 warp(原圖,M) 上 → 最終 原圖→ds = W^-1 · M"""
    h, w = ds_gray.shape
    try:
        M = np.asarray(M, np.float64)
        for _ in range(2):                                  # 兩輪：第一輪拉近、第二輪精修
            warped = cv2.warpAffine(clahe_raw, M, (w, h), flags=cv2.INTER_LINEAR)
            ones = np.full(clahe_raw.shape, 255, np.uint8)
            foot = cv2.warpAffine(ones, M, (w, h), flags=cv2.INTER_NEAREST) == 255
            mask = (foot & valid).astype(np.uint8)
            if mask.sum() < 500:
                return M
            W = np.eye(2, 3, dtype=np.float32)
            crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
            _, W = cv2.findTransformECC(ds_gray.astype(np.float32), warped.astype(np.float32),
                                        W, cv2.MOTION_AFFINE, crit, mask, 5)
            M = (np.linalg.inv(to3x3(W.astype(np.float64))) @ to3x3(M))[:2]
        return M
    except cv2.error:
        return M


def score_M(it, ds, M, valid):
    h, w = ds.shape
    ones = np.full(it["raw"].shape, 255, np.uint8)
    foot = cv2.warpAffine(ones, M, (w, h), flags=cv2.INTER_NEAREST) == 255
    warped_cl = cv2.warpAffine(it["clahe"], M, (w, h), flags=cv2.INTER_LINEAR)
    s = ncc(warped_cl, ds, foot & valid)
    return (-1.0 if np.isnan(s) else s), foot


def template_fallback(it, ds, valid):
    """SIFT 失敗時的備援：多尺度 template matching（只做縮放+平移+翻轉），回傳 [(M, flip)]"""
    ys, xs = np.where(valid)
    if len(ys) < 100:
        return []
    by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = ds.shape
    out = []
    for flip in ["", "h", "v", "hv"]:
        q = apply_flip(ds, flip)
        qv = apply_flip(valid.astype(np.uint8), flip).astype(bool)
        ys, xs = np.where(qv)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        tpl = q[y0:y1, x0:x1]
        f = min(1.0, 160 / max(tpl.shape))
        tpl_s = cv2.resize(tpl, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        best = None
        for sc in np.geomspace(0.15, 6.0, 90):
            k = sc * f
            R = cv2.resize(it["clahe"], None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            if R.shape[0] < tpl_s.shape[0] or R.shape[1] < tpl_s.shape[1]:
                continue
            r = cv2.matchTemplate(R, tpl_s, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (mx, sc, loc[0] / f, loc[1] / f)
        if best is None:
            continue
        _, sc, X, Y = best
        Mq = np.array([[sc, 0, x0 - X], [0, sc, y0 - Y]], np.float64)   # 原圖 → q
        out.append(((flip_matrix(flip, w, h) @ to3x3(Mq))[:2], flip))
    return out


# ============================================================
# 主流程
# ============================================================
def process(raw_dir, ds_root, out_root):
    ds_root, out_root = Path(ds_root), Path(out_root)
    if out_root.exists():
        print(f"⚠️ {out_root} 已存在，請先刪除再跑（避免新舊混雜）")
        return
    all_imgs = list_images(ds_root)
    ds_imgs = [p for p in all_imgs if "images" in p.relative_to(ds_root).parts]
    if not ds_imgs:
        ds_imgs = all_imgs          # 沒有 images/ 子資料夾（扁平資料夾）→ 處理底下所有圖
    if not ds_imgs:
        print(f"❌ {ds_root.resolve()} 底下找不到任何圖片({sorted(IMAGE_EXTENSIONS)})")
        print("   該層的內容：", [p.name for p in sorted(ds_root.iterdir())][:30])
        return
    print("=== 資料集圖片分布：",
          dict(Counter(str(p.parent.relative_to(ds_root)) for p in ds_imgs)), "===")

    lib = RawLibrary(raw_dir)
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=SIFT_CONTRAST)

    # 先整包複製（labels / data.yaml / 其他檔），之後只覆寫圖片
    shutil.copytree(ds_root, out_root)
    print(f"=== 處理資料集圖片：{len(ds_imgs)} 張 ===")
    rows = []
    for k, p in enumerate(ds_imgs, 1):
        rel = p.relative_to(ds_root)
        rec = {"資料集圖片": str(rel)}
        color = imread_u(p, cv2.IMREAD_UNCHANGED)
        ds = read_gray(p)
        h, w = ds.shape

        valid_ds = ds > BLACK_BORDER_THRESHOLD
        hints = lib.hinted(p.stem)

        best = None  # (NCC, idx, M_ds, flip, inliers, 方法)
        def consider(idx, M, flip, n, how):
            nonlocal best
            sc_, _ = score_M(lib.items[idx], ds, M, valid_ds)
            if best is None or sc_ > best[0]:
                best = (sc_, idx, M, flip, n, how)

        for flip in ["", "h", "v", "hv"]:
            q = apply_flip(ds, flip)
            kp_q, desc_q = sift.detectAndCompute(q, None)
            if desc_q is None or len(desc_q) < 2:
                continue
            F = flip_matrix(flip, w, h)                     # ds → q（自身反矩陣）
            for idx, is_hint in lib.candidates(desc_q, p.stem):
                M, n = lib.estimate(idx, kp_q, desc_q, MIN_INLIERS_HINT if is_hint else MIN_INLIERS)
                if M is not None:
                    consider(idx, (F @ to3x3(M))[:2], flip, n, "SIFT")
            if best is not None and best[0] >= NCC_WARN:
                break                                       # 已經很好，不必再試翻轉

        if (best is None or best[0] < NCC_WARN) and hints:  # SIFT 不夠 → 對檔名對應的原圖做 template 備援
            for idx in hints:
                for M, flip in template_fallback(lib.items[idx], ds, valid_ds):
                    consider(idx, M, flip, 0, "template")

        if best is None:
            score, idx, M, flip, n, how = -1.0, None, None, "", 0, "-"
        else:
            score, idx, M, flip, n, how = best
            it = lib.items[idx]
            M2 = refine_ecc(it["clahe"], ds, M, valid_ds)
            s2, _ = score_M(it, ds, M2, valid_ds)
            if s2 >= score:
                M, score = M2, s2

        if idx is None or score < NCC_WARN:
            # 不輸出：把輸出端的 CLAHE 圖與 label 移到 _review，避免混進無CLAHE組
            rv = out_root / REVIEW_DIR
            rv.mkdir(exist_ok=True)
            dst = out_root / rel
            if dst.exists():
                shutil.move(str(dst), str(rv / dst.name))
            for lp in (dst.with_suffix(".txt"),
                       out_root / Path(*[("labels" if x == "images" else x) for x in rel.parts]).with_suffix(".txt")):
                if lp.exists():
                    shutil.move(str(lp), str(rv / lp.name))
            rec.update({"對應原圖": lib.items[idx]["path"].name if idx is not None else None,
                        "NCC(重建CLAHE vs 資料集)": round(score, 4) if idx is not None else None,
                        "方法": how,
                        "狀態": "❌ 未還原（已移到 _review，CLAHE組也要剔除這張）"})
            rows.append(rec)
            print(f"[{k}/{len(ds_imgs)}] {rel}: ❌ 未還原 "
                  f"(最佳 {rec['對應原圖']} NCC={score:.4f}, hint={[lib.items[i]['path'].name for i in hints]})")
            continue

        it = lib.items[idx]
        _, foot = score_M(it, ds, M, valid_ds)
        warped_raw = cv2.warpAffine(it["raw"], M, (w, h), flags=cv2.INTER_LINEAR)
        out = ds.copy()
        out[foot] = warped_raw[foot]                        # 足跡外（letterbox/補邊）沿用原像素
        if color is not None and color.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        imwrite_u(out_root / rel, out, [cv2.IMWRITE_JPEG_QUALITY, 100])

        sc = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
        rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        same = (not hints) or (idx in hints)
        rec.update({"對應原圖": it["path"].name, "編號一致": "是" if same else "⚠️否", "方法": how,
                    "inliers": n, "翻轉": flip or "-",
                    "縮放": round(sc, 4), "旋轉deg": round(rot, 2),
                    "NCC(重建CLAHE vs 資料集)": round(score, 4) if not np.isnan(score) else None,
                    "狀態": "✅" if same else "✅ 但原圖編號不同，請目視確認"})
        rows.append(rec)
        print(f"[{k}/{len(ds_imgs)}] {rel} ← {it['path'].name}  {how} inl={n} NCC={score:.4f} "
              f"{'✅' if same else '✅(編號不同⚠️)'}")

    df = pd.DataFrame(rows)
    df.to_csv(out_root / QC_CSV, index=False, encoding="utf-8-sig")
    st = df["狀態"].astype(str)
    print("\n=======================================================")
    n_bad = st.str.startswith('❌').sum()
    print(f"✅ 還原: {st.str.startswith('✅').sum()}（其中編號不同 {st.str.contains('編號不同').sum()}）  "
          f"❌ 未還原: {n_bad}")
    if n_bad:
        bad = df.loc[st.str.startswith('❌'), "資料集圖片"]
        (out_root / "未還原清單.txt").write_text("\n".join(bad), encoding="utf-8")
        print(f"👉 未還原的 {n_bad} 張已移到 {REVIEW_DIR}/；CLAHE 組也要剔除同樣這些（清單：未還原清單.txt）")
    dup = df.loc[st.str.startswith('✅'), "對應原圖"].value_counts() if "對應原圖" in df else pd.Series(dtype=int)
    dup = dup[dup > 1]
    if len(dup):
        print(f"ℹ️ 有 {len(dup)} 張原圖被多張資料集圖配到（Roboflow增強副本或多顆牙裁切屬正常，否則請檢查）")
    print(f"📍 輸出: {out_root}   QC: {out_root / QC_CSV}")
    print("=======================================================")


# ---------------- 直接按「執行」時用這三個（命令列有給參數就以命令列為準）----------------
RAW_DIR = [r"C:\Users\user\Desktop\論文\code\data set\Data_first",
           r"C:\Users\user\Desktop\論文\code\data set\Data_second",
           r"C:\Users\user\Desktop\論文\code\data set\Data_RCTKKK",
           r"C:\Users\user\Desktop\論文\code\data set\Data_train"]                 # 👈 A1 之前的原圖；多個資料夾就列多個，例如 [r"C:\...\train原圖1", r"C:\...\train原圖2"]
CLAHE_SET_DIR = "data set/cropped_train_A1"   # 👈 底下有 train/valid/test 或 images/labels 的那層
OUTPUT_DIR = "A1r_train_noCLAHE"               # 👈 輸出（不可已存在）

if __name__ == "__main__":
    if len(sys.argv) == 4:
        process(*sys.argv[1:])
    elif len(sys.argv) == 1:
        raw_list = [RAW_DIR] if isinstance(RAW_DIR, str) else RAW_DIR
        for d in (*raw_list, CLAHE_SET_DIR):
            if not Path(d).exists():
                print(f"❌ 找不到 {Path(d).resolve()}，請改檔案底部的 RAW_DIR / CLAHE_SET_DIR")
                sys.exit(1)
        process(RAW_DIR, CLAHE_SET_DIR, OUTPUT_DIR)
    else:
        print(f"用法: python {Path(__file__).name} <原圖資料夾> <CLAHE資料集根目錄> <輸出資料集根目錄>")
        sys.exit(1)