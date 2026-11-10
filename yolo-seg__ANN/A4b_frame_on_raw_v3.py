"""
A4b_frame_on_raw.py  (v4：自動偵測 resize + 自動配對原圖)
============================================
用 A4 的裁切座標，把「原始未 CLAHE 全圖」切出一份與現有 CLAHE 裁切圖
逐像素對齊的無 CLAHE 雙胞胎。labels 完全不用動。

*** 原理 ***
CLAHE 只改灰階、不改幾何 → 同一個像素格上的絕對座標可以直接共用。
把「原圖 → A4 輸入圖」那層幾何重現出來，裁切就是零誤差。

    原圖 ──resize(?)──> 640 圖 ──A1 CLAHE──> A4 crop(px) ──> CLAHE裁切圖（你標註的）
      └──────────────── 同一條路，但不做 CLAHE ──────────> 無CLAHE裁切圖（本腳本輸出）

*** 兩個自動化 ***
1. 幾何自動偵測（GEOMETRY="auto"）
   窮舉 順序(resize→CLAHE / CLAHE→resize) × resize(stretch/letterbox/fill) × 內插(AREA/LINEAR)
   ＋「中間沒 resize」共 13 種，用 manifest 座標裁切後跟現有 CLAHE 裁切圖比對。
   取各候選在樣本中的「最小 MAE」而非中位數 —— 樣本可能有一半配對是錯的，
   只要有一張配對正確，正確的幾何就會跳出來。

2. 原圖自動配對（PAIRING="name_then_search"）
   先用檔名對；對不上或驗證不過，就拿該張的 CLAHE 裁切圖去跟「全部原圖」
   逐一比對，MAE 最小且夠小的那張就是真正的來源。
   （檔名偏移不固定、A2 改名表遺失時唯一可靠的辦法。實測正確匹配 MAE 2~4、
     次佳 24~35，差一個數量級，判定可靠。）
   配對結果輸出成 A4b_配對表.csv，可人工稽核、也可以重複使用。

*** 自我驗證 ***
每一張都比對「重建的 CLAHE 裁切」vs「現有的 CLAHE 裁切」，
門檻依幾何偵測的實測值自動校準，不過的不輸出。

*** OUTPUT_SCALE ***
  "as_cropped"（預設）：在 640 空間裁切，與現有 CLAHE 裁切圖逐像素對齊 → CLAHE 消融用
  "native"           ：換算回原圖座標、從原始解析度裁切，解析度較高
                       ⚠️ 同時改變「輸入解析度」第二個變因，不可直接與現有 CLAHE 組比。
                          框壓到 letterbox 黑邊的那幾張會跳過。

用法：
    python A4b_frame_on_raw.py
    python A4b_frame_on_raw.py <原圖資料夾> <A4manifest.xlsx> <CLAHE裁切圖資料夾> <輸出資料夾>

安裝需求：
    pip install opencv-python numpy pandas openpyxl
"""

import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ============================================================
# 設定區
# ============================================================
RAW_DIR = "data set/rename_train/images"
CROP_MANIFEST_XLSX = "裁切結果_train_A1.xlsx"
CLAHE_CROP_DIR = "data set/cropped_train_CLAHE"
OUTPUT_DIR = "cropped_train"

# 配對方式
#   "name"               只用檔名（舊行為）
#   "search"             全部用全圖搜尋
#   "name_then_search"   先檔名、不行再搜尋 ← 建議
PAIRING = "name_then_search"
RENAME_MAP_CSV = ""        # 有 A2 的 merge_rename_對照表.csv 就填，會在檔名階段先試

# 幾何："auto" 或 ("順序","resize","內插")，例如 ("CLAHE→resize","letterbox","AREA")
GEOMETRY = "auto"
TARGET = 640               # A4 輸入圖的邊長
PAD_VALUE = 0              # Fit(black edges) 的黑邊值
PROBE_N = 8                # 幾何偵測取幾張樣本

OUTPUT_SCALE = "as_cropped"   # "as_cropped"（消融用） | "native"（高解析度，另一個實驗）

# 驗證門檻：依幾何偵測的實測值自動校準
# 幾何偵測取的是「最佳樣本」，典型樣本會比它差一些，所以倍率要留寬。
# 判別力來自差距而非絕對值：正確配對 MAE 2~4，配錯原圖 MAE 30+、NCC < 0.6，
# 門檻放到 MAE<=12 / NCC>=0.97 仍然把錯的擋得乾乾淨淨。
AUTO_THRESHOLD = True
PASS_MAE = 4.0             # 門檻下限
PASS_NCC = 0.999           # 門檻上限
MARGIN_MAE = 5.0           # 容許到最佳樣本 MAE 的幾倍
MARGIN_NCC = 0.020         # 容許最佳樣本 NCC 往下幾點
MAX_PASS_MAE = 12.0        # 再寬也不接受超過這個 MAE
MIN_PASS_NCC = 0.970       # 再寬也不接受低於這個 NCC

JPEG_QUALITY = 100
QC_N = 8
RAW_CACHE_SIZE = 4

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
QC_DIR_NAME = "_qc"
MANIFEST_OUT = "A4b_還原QC.csv"
PAIRMAP_OUT = "A4b_配對表.csv"

COL_SRC = "圖片檔名"
COL_DST_NAME = "裁切檔名"
COL_DST_PATH = "裁切檔案路徑"
COL_PX_JOINED = "裁切像素座標_x1y1x2y2"
COLS_PX_SPLIT = ("裁切像素座標_x1", "裁切像素座標_y1", "裁切像素座標_x2", "裁切像素座標_y2")
COL_STATUS_IN = "狀態"


# ============================================================
# IO（中文路徑安全）
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


def ncc(a, b):
    a = a.astype(np.float64).ravel() - a.mean()
    b = b.astype(np.float64).ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def mae_of(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def basename_any(s):
    return str(s).replace("\\", "/").rstrip("/").split("/")[-1]


# ============================================================
# resize 候選。回傳 (640圖, map_back, 內容區box)
# ============================================================
def _rs_none(g, interp):
    return g, (lambda x, y: (x, y)), None


def _rs_stretch(g, interp):
    h, w = g.shape[:2]
    return (cv2.resize(g, (TARGET, TARGET), interpolation=interp),
            (lambda x, y: (x * w / TARGET, y * h / TARGET)), (0, 0, TARGET, TARGET))


def _rs_letterbox(g, interp):
    """Roboflow「Fit (black edges)」：長邊縮到 TARGET、置中補黑。"""
    h, w = g.shape[:2]
    s = TARGET / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(g, (nw, nh), interpolation=interp)
    o = np.full((TARGET, TARGET), PAD_VALUE, np.uint8)
    y0, x0 = (TARGET - nh) // 2, (TARGET - nw) // 2
    o[y0:y0 + nh, x0:x0 + nw] = r
    return o, (lambda x, y: ((x - x0) * w / nw, (y - y0) * h / nh)), (x0, y0, x0 + nw, y0 + nh)


def _rs_fill(g, interp):
    h, w = g.shape[:2]
    s = TARGET / min(h, w)
    nw, nh = max(TARGET, round(w * s)), max(TARGET, round(h * s))
    r = cv2.resize(g, (nw, nh), interpolation=interp)
    cy, cx = (nh - TARGET) // 2, (nw - TARGET) // 2
    return (r[cy:cy + TARGET, cx:cx + TARGET],
            (lambda x, y: ((x + cx) * w / nw, (y + cy) * h / nh)), (0, 0, TARGET, TARGET))


RESIZERS = {"none": _rs_none, "stretch": _rs_stretch,
            "letterbox": _rs_letterbox, "fill": _rs_fill}
INTERPS = {"none": None, "AREA": cv2.INTER_AREA, "LINEAR": cv2.INTER_LINEAR}


def candidate_list():
    out = [("none", "none", "none")]
    for order in ("resize→CLAHE", "CLAHE→resize"):
        for rname in ("letterbox", "stretch", "fill"):
            for iname in ("AREA", "LINEAR"):
                out.append((order, rname, iname))
    return out


def build_640(raw_gray, geom, enhance_image):
    """回傳 (clahe_640, plain_640, map_back, 內容區box)。"""
    order, rname, iname = geom
    rfn, interp = RESIZERS[rname], INTERPS[iname]
    if order == "CLAHE→resize":
        cl, mb, box = rfn(to_gray(enhance_image(raw_gray)), interp)
        plain, _, _ = rfn(raw_gray, interp)
        return cl, plain, mb, box
    plain, mb, box = rfn(raw_gray, interp)
    return to_gray(enhance_image(plain)), plain, mb, box


def build_clahe_640(raw_gray, geom, enhance_image):
    """只要驗證用的 CLAHE 640（搜尋索引用，省一半記憶體）。"""
    order, rname, iname = geom
    rfn, interp = RESIZERS[rname], INTERPS[iname]
    if order == "CLAHE→resize":
        return rfn(to_gray(enhance_image(raw_gray)), interp)[0]
    return to_gray(enhance_image(rfn(raw_gray, interp)[0]))


# ============================================================
# manifest 欄位
# ============================================================
def make_px_getter(df):
    if all(c in df.columns for c in COLS_PX_SPLIT):
        def get(r):
            try:
                v = [int(float(r[c])) for c in COLS_PX_SPLIT]
            except (ValueError, TypeError):
                return None
            return tuple(v) if v[2] > v[0] and v[3] > v[1] else None
        return get, "四欄分開（裁切像素座標_x1/_y1/_x2/_y2）"
    if COL_PX_JOINED in df.columns:
        def get(r):
            try:
                v = [int(float(t)) for t in str(r[COL_PX_JOINED]).split(",")]
            except (ValueError, TypeError):
                return None
            return tuple(v) if len(v) == 4 and v[2] > v[0] and v[3] > v[1] else None
        return get, f"單欄逗號（{COL_PX_JOINED}）"
    raise KeyError("❌ manifest 找不到裁切座標欄位。\n"
                   f"   支援 A) {' / '.join(COLS_PX_SPLIT)}\n"
                   f"        B) {COL_PX_JOINED}\n"
                   f"   實際欄位：{list(df.columns)}")


def make_name_getter(df):
    if COL_DST_NAME in df.columns:
        return (lambda r: basename_any(r[COL_DST_NAME])), COL_DST_NAME
    if COL_DST_PATH in df.columns:
        return (lambda r: basename_any(r[COL_DST_PATH])), COL_DST_PATH
    return (lambda r: basename_any(r[COL_SRC])), f"{COL_SRC}（無裁切檔名欄）"


def list_raw(raw_dirs):
    dirs = [str(d).strip() for d in (raw_dirs if isinstance(raw_dirs, (list, tuple))
                                     else str(raw_dirs).split(";")) if str(d).strip()]
    paths = []
    for d in dirs:
        if not Path(d).exists():
            print(f"   ⚠️ 原圖來源不存在: {d}")
            continue
        got = [p for p in sorted(Path(d).rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
        print(f"   原圖來源: {d}  ({len(got)} 張)")
        paths += got
    if not paths:
        raise FileNotFoundError(f"❌ {raw_dirs} 底下找不到任何圖片")
    return sorted(set(paths))


def load_rename_map():
    if not RENAME_MAP_CSV:
        return {}
    df = pd.read_csv(RENAME_MAP_CSV, encoding="utf-8-sig")
    cols = {str(c).strip(): c for c in df.columns}
    new_c, old_c = cols.get("新檔名"), cols.get("原始圖片檔名")
    if not (new_c and old_c):
        print(f"   ⚠️ {RENAME_MAP_CSV} 找不到「新檔名」/「原始圖片檔名」欄，忽略")
        return {}
    return {Path(str(r[new_c])).stem: Path(str(r[old_c])).stem for _, r in df.iterrows()}


# ============================================================
# 幾何偵測（取最小 MAE，對「樣本本身配對是錯的」有抵抗力）
# ============================================================
def probe_geometry(samples, enhance_image):
    rows = []
    for geom in candidate_list():
        best = None
        for raw, px, ref in samples:
            x1, y1, x2, y2 = px
            try:
                cl = build_clahe_640(raw, geom, enhance_image)
                if not (0 <= x1 < x2 <= cl.shape[1] and 0 <= y1 < y2 <= cl.shape[0]):
                    continue
                crop = cl[y1:y2, x1:x2]
                if crop.shape != ref.shape:
                    continue
                m, s = mae_of(crop, ref), ncc(crop, ref)
                if best is None or m < best[0]:
                    best = (m, s)
            except (cv2.error, ValueError, IndexError):
                continue
        rows.append({"順序": geom[0], "resize": geom[1], "內插": geom[2],
                     "最佳MAE": best[0] if best else np.nan,
                     "該張NCC": best[1] if best else np.nan, "_geom": geom})
    tab = pd.DataFrame(rows).sort_values("最佳MAE", na_position="last").reset_index(drop=True)
    print("\n=== 幾何候選（取樣本中最小 MAE；樣本可能有配對錯的，故不用中位數）===")
    print(tab.drop(columns="_geom").round(4).to_string(index=False))
    best = tab.iloc[0]
    if pd.isna(best["最佳MAE"]) or best["最佳MAE"] > MAX_PASS_MAE:
        print("\n❌ 13 種組合都對不上，連最好的樣本都 MAE > "
              f"{MAX_PASS_MAE}。中間還有本腳本沒涵蓋的處理。")
        print("   → 改用 A1r_restore_noCLAHE_dataset.py（SIFT 自己解幾何）。")
        return None, None
    print(f"\n✅ 採用：順序={best['順序']}  resize={best['resize']}  內插={best['內插']}"
          f"  (最佳樣本 MAE={best['最佳MAE']:.2f}, NCC={best['該張NCC']:.5f})")
    if len(tab) > 1 and pd.notna(tab.iloc[1]["最佳MAE"]):
        gap = tab.iloc[1]["最佳MAE"] - best["最佳MAE"]
        if gap < 1.0 and tab.iloc[1]["resize"] == best["resize"]:
            print(f"   ℹ️ 第二名只差 {gap:.2f}，兩種順序分不太出來；")
            print(f"      但它們的『無 CLAHE 版』都是 {best['resize']}(原圖)，輸出相同，不影響結果。")
    return best["_geom"], (float(best["最佳MAE"]), float(best["該張NCC"]))


# ============================================================
# 全圖搜尋索引
# ============================================================
class SearchIndex:
    def __init__(self, raw_paths, geom, enhance_image):
        self.paths, self.imgs = [], []
        print(f"   建立搜尋索引：{len(raw_paths)} 張原圖 ...", end="", flush=True)
        for k, p in enumerate(raw_paths, 1):
            g = to_gray(imread_u(p))
            if g is None:
                continue
            try:
                self.paths.append(p)
                self.imgs.append(build_clahe_640(g, geom, enhance_image))
            except (cv2.error, ValueError):
                self.paths.pop()
            if k % 100 == 0:
                print(f" {k}", end="", flush=True)
        print(f" 完成（{len(self.paths)} 張可用）")

    def best(self, ref, px, topk=2):
        x1, y1, x2, y2 = px
        out = []
        for p, cl in zip(self.paths, self.imgs):
            if not (0 <= x1 < x2 <= cl.shape[1] and 0 <= y1 < y2 <= cl.shape[0]):
                continue
            crop = cl[y1:y2, x1:x2]
            if crop.shape != ref.shape:
                continue
            out.append((mae_of(crop, ref), p))
        out.sort(key=lambda t: t[0])
        return out[:topk]


# ============================================================
# 主流程
# ============================================================
def process(raw_dir=RAW_DIR, manifest=CROP_MANIFEST_XLSX,
            clahe_dir=CLAHE_CROP_DIR, out_dir=OUTPUT_DIR):
    try:
        from A1_CLAHE_before_annotation import (enhance_image, CLAHE_CLIP_LIMIT,
                                                CLAHE_TILE_GRID_SIZE, BLACK_BORDER_THRESHOLD)
    except ImportError:
        raise ImportError("❌ 找不到 A1_CLAHE_before_annotation.py，請放在同一個資料夾"
                          "（CLAHE 參數必須跟標註時用的是同一份，不可以另外抄一份）")
    if OUTPUT_SCALE not in ("as_cropped", "native"):
        raise ValueError(f"❌ OUTPUT_SCALE 只能是 'as_cropped' 或 'native'，收到 {OUTPUT_SCALE!r}")
    if PAIRING not in ("name", "search", "name_then_search"):
        raise ValueError(f"❌ PAIRING 只能是 name / search / name_then_search，收到 {PAIRING!r}")

    mf_path, clahe_dir, out_dir = Path(manifest), Path(clahe_dir), Path(out_dir)
    if not mf_path.exists():
        raise FileNotFoundError(f"❌ 找不到 manifest {mf_path.resolve()}")
    if not clahe_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 CLAHE 裁切圖資料夾 {clahe_dir.resolve()}")
    if out_dir.exists():
        print(f"⚠️ {out_dir} 已存在，為避免新舊混雜直接停止；要重跑請先刪除。")
        return

    df = pd.read_excel(mf_path)
    if COL_SRC not in df.columns:
        raise KeyError(f"❌ manifest 缺少欄位「{COL_SRC}」，實際欄位：{list(df.columns)}")
    get_px, px_desc = make_px_getter(df)
    get_name, name_desc = make_name_getter(df)

    print(f"=== A4b v4：manifest {len(df)} 列，{df[COL_SRC].nunique()} 張來源全圖 ===")
    print(f"    座標欄格式：{px_desc}")
    print(f"    輸出檔名來源：{name_desc}")
    print(f"    配對方式：{PAIRING}")
    print(f"    輸出解析度：{OUTPUT_SCALE}"
          + ("（與現有 CLAHE 裁切圖逐像素對齊，消融用）" if OUTPUT_SCALE == "as_cropped"
             else "  ⚠️ 多改了一個變因，不可直接與現有 CLAHE 組比"))
    print(f"    CLAHE 參數（來自 A1）：clip={CLAHE_CLIP_LIMIT}, "
          f"tile={CLAHE_TILE_GRID_SIZE}, 黑邊門檻={BLACK_BORDER_THRESHOLD}")

    raw_paths = list_raw(raw_dir)
    raw_by_stem = {}
    for p in raw_paths:
        raw_by_stem.setdefault(p.stem, p)
        raw_by_stem.setdefault(p.stem.replace(" ", "-"), p)
    rename_map = load_rename_map()

    # ---------- 每列的基本資料 ----------
    info = {}
    for i in df.index:
        r = df.loc[i]
        src = str(r[COL_SRC])
        dst = get_name(r)
        px = get_px(r)
        ref_p = clahe_dir / dst
        ref = to_gray(imread_u(ref_p)) if ref_p.exists() else None
        info[i] = {"src": src, "dst": dst, "px": px, "ref_p": ref_p, "ref": ref}

    # ---------- 幾何偵測 ----------
    if GEOMETRY == "auto":
        samples = []
        for i in df.sort_values(COL_SRC, kind="stable").index:
            i0 = info[i]
            if len(samples) >= PROBE_N:
                break
            if i0["px"] is None or i0["ref"] is None:
                continue
            p = raw_by_stem.get(Path(i0["src"]).stem) or \
                raw_by_stem.get(rename_map.get(Path(i0["src"]).stem, ""))
            if p is None:
                continue
            g = to_gray(imread_u(p))
            if g is not None:
                samples.append((g, i0["px"], i0["ref"]))
        if not samples:
            print("❌ 連一組檔名配對都找不到，無法偵測幾何。請確認 RAW_DIR。")
            return
        print(f"\n--- 幾何自動偵測（{len(samples)} 張樣本）---")
        geom, observed = probe_geometry(samples, enhance_image)
        if geom is None:
            return
    else:
        geom, observed = tuple(GEOMETRY), None
        print(f"    幾何（手動指定）：{geom}")

    if AUTO_THRESHOLD and observed is not None:
        o_mae, o_ncc = observed
        pass_mae = min(MAX_PASS_MAE, max(PASS_MAE, o_mae * MARGIN_MAE))
        pass_ncc = max(MIN_PASS_NCC, min(PASS_NCC, o_ncc - MARGIN_NCC))
        print(f"    驗證門檻（依實測自動校準）：MAE <= {pass_mae:.2f}，NCC >= {pass_ncc:.5f}")
    else:
        pass_mae, pass_ncc = PASS_MAE, PASS_NCC
        print(f"    驗證門檻（固定）：MAE <= {pass_mae:.2f}，NCC >= {pass_ncc:.5f}")

    # ---------- 逐「來源圖」配對（一張來源圖只解析一次，多顆牙共用）----------
    print(f"\n--- 原圖配對（{df[COL_SRC].nunique()} 張來源圖）---")
    index = None
    if PAIRING in ("search", "name_then_search"):
        need_index = PAIRING == "search"
    pair_rows, pairing = [], {}
    groups = list(df.groupby(COL_SRC, sort=True).groups.items())

    def probe_pair(raw_path, i):
        """用第 i 列驗證 raw_path 對不對，回傳 (mae, ncc) 或 None。"""
        i0 = info[i]
        g = to_gray(imread_u(raw_path))
        if g is None:
            return None
        try:
            cl = build_clahe_640(g, geom, enhance_image)
        except (cv2.error, ValueError):
            return None
        x1, y1, x2, y2 = i0["px"]
        if not (0 <= x1 < x2 <= cl.shape[1] and 0 <= y1 < y2 <= cl.shape[0]):
            return None
        crop = cl[y1:y2, x1:x2]
        if crop.shape != i0["ref"].shape:
            return None
        return mae_of(crop, i0["ref"]), ncc(crop, i0["ref"])

    for n, (src, idxs) in enumerate(groups, 1):
        testable = [i for i in idxs if info[i]["px"] is not None and info[i]["ref"] is not None]
        if not testable:
            pairing[src] = (None, "無可驗證的列（缺座標或缺對照圖）", np.nan)
            pair_rows.append({"manifest來源": src, "實際原圖": None,
                              "方式": "—", "MAE": None})
            continue
        t = testable[0]
        chosen, how, mae = None, "", np.nan

        if PAIRING in ("name", "name_then_search"):
            cand = raw_by_stem.get(Path(src).stem) or \
                   raw_by_stem.get(rename_map.get(Path(src).stem, ""))
            if cand is not None:
                res = probe_pair(cand, t)
                if res and res[0] <= pass_mae and res[1] >= pass_ncc:
                    chosen, how, mae = cand, "檔名", res[0]

        if chosen is None and PAIRING in ("search", "name_then_search"):
            if index is None:
                index = SearchIndex(raw_paths, geom, enhance_image)
            hits = index.best(info[t]["ref"], info[t]["px"], topk=2)
            if hits and hits[0][0] <= pass_mae:
                chosen, how, mae = hits[0][1], "全圖搜尋", hits[0][0]
                if len(hits) > 1 and hits[1][0] < hits[0][0] * 3:
                    how += f"（次佳 {hits[1][0]:.1f}，區分度低⚠️）"

        pairing[src] = (chosen, how or "找不到匹配的原圖", mae)
        pair_rows.append({"manifest來源": src,
                          "實際原圖": chosen.name if chosen else None,
                          "方式": how or "找不到", "MAE": round(mae, 2) if mae == mae else None,
                          "與manifest同名": (chosen is not None
                                            and chosen.stem == Path(src).stem)})
        if n % 25 == 0 or n == len(groups):
            ok = sum(1 for v in pairing.values() if v[0] is not None)
            print(f"   ... {n}/{len(groups)}  已配對 {ok}")

    out_dir.mkdir(parents=True)
    pm = pd.DataFrame(pair_rows)
    pm.to_csv(out_dir / PAIRMAP_OUT, index=False, encoding="utf-8-sig")
    n_named = int(pm["方式"].astype(str).str.startswith("檔名").sum())
    n_searched = int(pm["方式"].astype(str).str.startswith("全圖搜尋").sum())
    n_nopair = int(pm["實際原圖"].isna().sum())
    print(f"   結果：檔名 {n_named}、全圖搜尋 {n_searched}、找不到 {n_nopair}")
    if "與manifest同名" in pm.columns:
        diff = pm[(pm["實際原圖"].notna()) & (~pm["與manifest同名"].fillna(False))]
        if len(diff):
            print(f"   ⚠️ 有 {len(diff)} 張的真實來源與 manifest 檔名不同（已記錄在 {PAIRMAP_OUT}）")
    dup = pm.loc[pm["實際原圖"].notna(), "實際原圖"].value_counts()
    dup = dup[dup > 1]
    if len(dup):
        print(f"   ⚠️ 有 {len(dup)} 張原圖被多個來源配到，請在 {PAIRMAP_OUT} 查核：")
        for k, v in dup.head(5).items():
            print(f"      {k} ← {v} 個來源")

    # ---------- 逐列輸出 ----------
    cache = OrderedDict()

    def get_built(p):
        if p not in cache:
            g = to_gray(imread_u(p))
            cache[p] = None if g is None else (g,) + build_640(g, geom, enhance_image)
            while len(cache) > RAW_CACHE_SIZE:
                cache.popitem(last=False)
        return cache[p]

    rows, n_qc, done = {}, 0, 0
    for i in df.sort_values(COL_SRC, kind="stable").index:
        i0 = info[i]
        r = df.loc[i]
        rec = {"來源全圖": i0["src"], "輸出檔名": i0["dst"],
               "原manifest狀態": r.get(COL_STATUS_IN)}
        done += 1

        if i0["px"] is None:
            rec["狀態"] = "⏭️ manifest 無有效裁切座標（該列當初就失敗/跳過）"
            rows[i] = rec
            continue
        if i0["ref"] is None:
            rec["狀態"] = f"❌ 找不到/讀不到對照的 CLAHE 裁切圖：{i0['ref_p']}"
            rows[i] = rec
            continue
        raw_p, how, _ = pairing.get(i0["src"], (None, "未配對", np.nan))
        if raw_p is None:
            rec["狀態"] = f"❌ 配對失敗：{how}"
            rows[i] = rec
            continue
        rec["原圖"] = raw_p.name
        rec["配對方式"] = how

        built = get_built(raw_p)
        if built is None:
            rec["狀態"] = "❌ 原圖讀取失敗"
            rows[i] = rec
            continue
        raw_gray, clahe_640, plain_640, map_back, content = built

        x1, y1, x2, y2 = i0["px"]
        H640, W640 = clahe_640.shape[:2]
        rec["原圖_wh"] = f"{raw_gray.shape[1]}x{raw_gray.shape[0]}"
        rec["裁切座標"] = f"{x1},{y1},{x2},{y2}"
        if not (0 <= x1 < x2 <= W640 and 0 <= y1 < y2 <= H640):
            rec["狀態"] = f"❌ 座標超出重建圖範圍（{W640}x{H640}）"
            rows[i] = rec
            continue
        crop_rebuilt = clahe_640[y1:y2, x1:x2]
        if crop_rebuilt.shape != i0["ref"].shape:
            rec["狀態"] = (f"❌ 尺寸對不上：現有 {i0['ref'].shape[1]}x{i0['ref'].shape[0]} "
                           f"vs 重建 {crop_rebuilt.shape[1]}x{crop_rebuilt.shape[0]}")
            rows[i] = rec
            continue

        mae = mae_of(crop_rebuilt, i0["ref"])
        sc = ncc(crop_rebuilt, i0["ref"])
        rec.update({"裁切_wh": f"{x2 - x1}x{y2 - y1}",
                    "MAE_重建CLAHE_vs_現有": round(mae, 2),
                    "NCC": round(sc, 5) if not np.isnan(sc) else None})
        if not (mae <= pass_mae and (np.isnan(sc) or sc >= pass_ncc)):
            rec["狀態"] = f"❌ 驗證未過(MAE={mae:.2f}, NCC={sc:.5f})，未輸出"
            rows[i] = rec
            continue

        if OUTPUT_SCALE == "as_cropped":
            crop_out = plain_640[y1:y2, x1:x2]
        else:
            if content is None:
                rec["狀態"] = "❌ native 模式：此幾何無法換算回原圖座標"
                rows[i] = rec
                continue
            cx0, cy0, cx1_, cy1_ = content
            if not (cx0 <= x1 and x2 <= cx1_ and cy0 <= y1 and y2 <= cy1_):
                rec["狀態"] = "❌ native 模式：裁切框壓到 letterbox 黑邊，跳過"
                rows[i] = rec
                continue
            nx1, ny1 = map_back(x1, y1)
            nx2, ny2 = map_back(x2, y2)
            Hn, Wn = raw_gray.shape[:2]
            nx1, ny1 = max(0, int(round(nx1))), max(0, int(round(ny1)))
            nx2, ny2 = min(Wn, int(round(nx2))), min(Hn, int(round(ny2)))
            if nx2 <= nx1 or ny2 <= ny1:
                rec["狀態"] = "❌ native 模式：換算後的框無效"
                rows[i] = rec
                continue
            crop_out = raw_gray[ny1:ny2, nx1:nx2]
            rec["native座標"] = f"{nx1},{ny1},{nx2},{ny2}"
            rec["native_wh"] = f"{nx2 - nx1}x{ny2 - ny1}"

        imwrite_u(out_dir / i0["dst"], crop_out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        rec["狀態"] = "✅"
        rows[i] = rec

        if n_qc < QC_N:
            tiles = [i0["ref"], crop_rebuilt, plain_640[y1:y2, x1:x2]]
            titles = ["1 CLAHE (existing)", "2 CLAHE (rebuilt)", "3 no-CLAHE (640)"]
            vis = []
            for im, t in zip(tiles, titles):
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(im, (0, 0), (im.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(im, t, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255), 1, cv2.LINE_AA)
                vis.append(im)
            vis = np.hstack(vis)
            cv2.putText(vis, f"MAE={mae:.2f} NCC={sc:.5f} src={raw_p.name} [{how}]",
                        (5, vis.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 255, 255), 1, cv2.LINE_AA)
            imwrite_u(out_dir / QC_DIR_NAME / f"{Path(i0['dst']).stem}.png", vis)
            n_qc += 1

        if done % 100 == 0:
            print(f"   ... 輸出 {done}/{len(df)}")

    out = pd.DataFrame([rows[i] for i in df.index if i in rows])
    out.to_csv(out_dir / MANIFEST_OUT, index=False, encoding="utf-8-sig")
    st = out["狀態"].astype(str)
    n_ok, n_bad = int(st.str.startswith("✅").sum()), int(st.str.startswith("❌").sum())

    print("\n=======================================================")
    print(f"✨ A4b v4 完成：manifest {len(out)} 列   幾何={geom}   輸出={OUTPUT_SCALE}")
    print(f"   ✅ 已輸出: {n_ok}")
    print(f"   ❌ 未輸出: {n_bad}")
    print(f"   ⏭️ 本來就沒座標: {int(st.str.startswith('⏭️').sum())}")
    if n_ok:
        ok = out[st.str.startswith("✅")]
        print(f"   驗證：MAE 中位數 {ok['MAE_重建CLAHE_vs_現有'].median():.2f}，"
              f"最大 {ok['MAE_重建CLAHE_vs_現有'].max():.2f}，NCC 中位數 {ok['NCC'].median():.5f}")
        if "配對方式" in ok.columns:
            print(f"   配對方式分布：{dict(ok['配對方式'].astype(str).str[:4].value_counts())}")
    if n_ok == 0:
        print("   ⚠️ 一張都沒輸出 —— 先看 QC 圖與配對表確認幾何/配對對不對。")
    if n_bad:
        bad = out.loc[st.str.startswith("❌")]
        (out_dir / "未還原清單.txt").write_text(
            "\n".join(map(str, bad["輸出檔名"])), encoding="utf-8")
        print(f"   👉 未輸出清單：{out_dir / '未還原清單.txt'}")
        print(f"      ⚠️ CLAHE 組也要剔除同樣這些檔案，兩組張數一致才能比")
        if "MAE_重建CLAHE_vs_現有" in bad.columns and bad["MAE_重建CLAHE_vs_現有"].notna().any():
            bm = bad["MAE_重建CLAHE_vs_現有"].dropna()
            print(f"      未過者的 MAE：中位數 {bm.median():.1f}、最小 {bm.min():.1f}、"
                  f"最大 {bm.max():.1f}（門檻 {pass_mae:.1f}）")
            near = int((bm <= pass_mae * 1.5).sum())
            if near:
                print(f"      其中 {near} 筆只是略高於門檻（<= {pass_mae * 1.5:.1f}）→ "
                      f"像是門檻偏嚴而非配錯，可調大 MARGIN_MAE / MARGIN_NCC 後重跑")
        for s_, c in bad["狀態"].astype(str).str[:62].value_counts().head(3).items():
            print(f"        {c:>4} 筆  {s_}")
    print(f"📍 輸出: {out_dir}")
    print(f"📍 配對表: {out_dir / PAIRMAP_OUT}  👈 務必抽查「與manifest同名=False」那幾列")
    print(f"📍 QC: {out_dir / QC_DIR_NAME}   明細: {out_dir / MANIFEST_OUT}")
    print("=======================================================")
    if OUTPUT_SCALE == "native":
        print("⚠️ native 模式：解析度與現有 CLAHE 裁切圖不同，不可直接做 CLAHE 消融。")
    print("👉 A4 之後若還有 A5 補邊 / Roboflow 匯出，請對本輸出重跑同樣步驟。")
    print("👉 換前處理後 seg 要重訓；評估看 B3 長度誤差 / C1 MAE 的跨 seed mean±SD。")


def main():
    a = sys.argv[1:]
    if not a:
        process()
    elif len(a) == 4:
        process(*a)
    else:
        print("用法:")
        print(f"  python {Path(__file__).name}")
        print(f"  python {Path(__file__).name} <原圖資料夾> <A4manifest.xlsx> "
              f"<CLAHE裁切圖資料夾> <輸出資料夾>")
        sys.exit(1)


if __name__ == "__main__":
    main()
