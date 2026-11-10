"""
A4e_rename_to_label.py
============================================
把無 CLAHE 裁切圖「改名成與已標註資料集相同的檔名」，照原資料集結構輸出，
label 與 data.yaml 一併複製。純檔名對應，不做任何影像處理。

原理（兩段式，先嚴後寬）：
    Roboflow 匯出檔名 = <原始裁切檔名>_<副檔名>.rf.<32位雜湊>.jpg
    1. 剝掉 `_jpg.rf.<hash>` → 用完整 stem 去無 CLAHE 資料夾找同名檔
    2. 找不到就改用「開頭的數字編號」比對（001、002…），忽略後面的 _d0 / _t1 之類的尾巴
       編號會正規化掉前導零，所以 001 和 1 視為同一個
    → 存成資料集那個檔名 → label 自然就對得起來

⚠️ 用編號比對時，若同一個編號在來源或資料集裡出現多次（同一張片子切了多顆牙），
   對應就不唯一。本腳本會照 ID_TIE_BREAK 規則挑一個並標記 ⚠️，
   請務必看對照表的「配對方式」欄與「編號重複」欄。

輸出：OUTPUT_DIR/<與資料集相同的相對路徑>
      只挑資料集裡真的有的那些（例如 378 張裡的 308 張），其餘不理

用法：
    python A4e_rename_to_label.py
    python A4e_rename_to_label.py <已標註資料集> <無CLAHE裁切圖> <輸出>

安裝需求：
    pip install opencv-python pandas
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
NOCLAHE_CROP_DIR = "data set/cropped_test"               # 👈 A4b 的輸出（378 張，無 CLAHE 版）
OUTPUT_DIR = "test-77"          # 👈 輸出（不可已存在）

COPY_LABELS = True        # 一併複製 label 與 data.yaml
WARN_SIZE_DIFF = True     # 尺寸與資料集影像不同時提醒（不會擋，只是告訴你）

# 編號比對
ID_REGEX = r"^(\d+)"      # 從 stem 開頭抓數字編號；預設抓最前面那串數字
ID_STRIP_LEADING_ZERO = True   # 001 == 1
ID_TIE_BREAK = "suffix"   # 同一編號有多個來源時怎麼挑：
                          #   "suffix" = 尾巴（_d0/_t1…）最相近的（建議）
                          #   "first"  = 排序後第一個
                          #   "skip"   = 不猜，直接標失敗

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MANIFEST_OUT = "A4e_改名對照.csv"
RF_PATTERN = re.compile(r"^(.*)_(jpg|jpeg|png|bmp|tif|tiff)\.rf\.[0-9a-fA-F]+$")


def strip_rf(stem):
    m = RF_PATTERN.match(stem)
    return m.group(1) if m else stem


def id_of(stem):
    """從 stem 取出開頭的數字編號（已剝掉 Roboflow 尾巴）。取不到回傳 None。"""
    m = re.match(ID_REGEX, strip_rf(stem))
    if not m:
        return None
    d = m.group(1)
    return str(int(d)) if ID_STRIP_LEADING_ZERO and d.isdigit() else d


def suffix_of(stem):
    """編號之後的尾巴，例如 001_d0 -> _d0。"""
    s = strip_rf(stem)
    m = re.match(ID_REGEX, s)
    return s[m.end():] if m else s


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def label_of(img_path, ds_root):
    """回傳 (label 原始路徑, 相對路徑) 或 (None, None)。"""
    c1 = img_path.with_suffix(".txt")
    if c1.exists():
        return c1, img_path.relative_to(ds_root).with_suffix(".txt")
    rel = img_path.relative_to(ds_root)
    parts = ["labels" if x == "images" else x for x in rel.parts]
    c2 = (ds_root / Path(*parts)).with_suffix(".txt")
    return (c2, Path(*parts).with_suffix(".txt")) if c2.exists() else (None, None)


def imsize(p):
    try:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return (img.shape[1], img.shape[0]) if img is not None else None
    except (OSError, ValueError):
        return None


def process(ds_dir=SEG_DATASET_DIR, src_dir=NOCLAHE_CROP_DIR, out_dir=OUTPUT_DIR):
    ds_root, src_root, out_dir = Path(ds_dir), Path(src_dir), Path(out_dir)
    for d in (ds_root, src_root):
        if not d.exists():
            raise FileNotFoundError(f"❌ 找不到 {d.resolve()}")
    if out_dir.exists():
        print(f"⚠️ {out_dir} 已存在，請先刪除再跑。")
        return

    ds_imgs = list_images(ds_root)
    if not ds_imgs:
        raise FileNotFoundError(f"❌ {ds_root.resolve()} 底下沒有影像")

    # 無 CLAHE 裁切圖：stem 與「數字編號」兩套索引
    src_paths = list_images(src_root)
    src_by_stem, src_by_id = {}, {}
    for p in src_paths:
        src_by_stem.setdefault(p.stem, p)
        i = id_of(p.stem)
        if i is not None:
            src_by_id.setdefault(i, []).append(p)

    print(f"=== A4e：資料集 {len(ds_imgs)} 張，無 CLAHE 裁切圖 {len(src_paths)} 張 ===")
    print(f"    資料集分布：{dict(Counter(str(p.parent.relative_to(ds_root)) for p in ds_imgs))}")
    print(f"    來源編號：{len(src_by_id)} 個不重複"
          + (f"，其中 {sum(1 for v in src_by_id.values() if len(v) > 1)} 個編號對到多個檔案"
             if any(len(v) > 1 for v in src_by_id.values()) else ""))
    ds_id_count = Counter(id_of(p.stem) for p in ds_imgs)

    rows, n_ok, n_lbl, n_size = [], 0, 0, 0
    for ip in ds_imgs:
        rel = ip.relative_to(ds_root)
        stem = strip_rf(ip.stem)
        key = id_of(ip.stem)
        rec = {"資料集檔名": ip.name, "相對路徑": str(rel),
               "還原出的裁切檔名": stem, "編號": key}

        # 1) 完整 stem
        sp = src_by_stem.get(stem) or src_by_stem.get(ip.stem)
        how = "完整檔名" if sp is not None else ""

        # 2) 數字編號
        if sp is None and key is not None:
            cand = src_by_id.get(key, [])
            if len(cand) == 1:
                sp, how = cand[0], "編號"
            elif len(cand) > 1:
                rec["編號重複"] = f"來源有 {len(cand)} 個：" + "/".join(c.name for c in cand[:4])
                if ID_TIE_BREAK == "skip":
                    rec["狀態"] = f"❌ 編號 {key} 對到 {len(cand)} 個來源，未指定挑選規則"
                    rows.append(rec)
                    continue
                if ID_TIE_BREAK == "suffix":
                    want = suffix_of(ip.stem)
                    cand = sorted(cand, key=lambda c: (suffix_of(c.stem) != want,
                                                       suffix_of(c.stem), c.name))
                sp, how = cand[0], "編號⚠️多個來源"
        if sp is None:
            rec["狀態"] = ("❌ 無 CLAHE 資料夾找不到（完整檔名與編號都對不到）"
                           if key is not None else "❌ 檔名開頭抓不到數字編號")
            rows.append(rec)
            continue
        if key is not None and ds_id_count.get(key, 0) > 1 and how.startswith("編號"):
            rec["編號重複"] = (rec.get("編號重複", "") +
                               f" | 資料集也有 {ds_id_count[key]} 張同編號").strip(" |")
            how += "⚠️資料集同編號多張"
        rec["來源檔"] = sp.name
        rec["配對方式"] = how

        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sp, dst)          # 內容原封不動，只換檔名與位置
        n_ok += 1
        rec["狀態"] = "✅"

        if WARN_SIZE_DIFF:
            a, b = imsize(ip), imsize(sp)
            if a and b:
                rec["資料集尺寸"] = f"{a[0]}x{a[1]}"
                rec["來源尺寸"] = f"{b[0]}x{b[1]}"
                if a != b:
                    rec["尺寸"] = "⚠️ 不同"
                    n_size += 1

        if COPY_LABELS:
            lp, lrel = label_of(ip, ds_root)
            if lp is not None:
                (out_dir / lrel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(lp, out_dir / lrel)
                rec["label"] = "已複製"
                n_lbl += 1
            else:
                rec["label"] = "無"
        rows.append(rec)

    if COPY_LABELS:
        for p in sorted(ds_root.rglob("*")):
            if p.is_file() and p.suffix.lower() not in IMAGE_EXTENSIONS \
                    and p.suffix.lower() != ".txt":
                d = out_dir / p.relative_to(ds_root)
                d.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, d)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / MANIFEST_OUT, index=False, encoding="utf-8-sig")
    miss = df[df["狀態"].astype(str).str.startswith("❌")]

    print("\n=======================================================")
    print(f"✨ A4e 完成：資料集 {len(df)} 張")
    print(f"   ✅ 已改名輸出: {n_ok}")
    print(f"   ❌ 找不到同名檔: {len(miss)}")
    if COPY_LABELS:
        print(f"   label 已複製: {n_lbl}")
    if "配對方式" in df.columns:
        print(f"   配對方式：{dict(df['配對方式'].dropna().astype(str).value_counts())}")
        warn = df[df["配對方式"].astype(str).str.contains("⚠️", na=False)]
        if len(warn):
            print(f"   ⚠️ {len(warn)} 張是靠編號猜的、而且編號不唯一 👈 一定要看對照表")
    dup = df.loc[df["來源檔"].notna(), "來源檔"].value_counts() if "來源檔" in df else pd.Series(dtype=int)
    dup = dup[dup > 1]
    if len(dup):
        print(f"   ⚠️ {len(dup)} 個來源檔被多張資料集影像用到：")
        for k, v in dup.head(5).items():
            print(f"      {k} ← {v} 張")
    if len(miss):
        (out_dir / "找不到清單.txt").write_text(
            "\n".join(map(str, miss["還原出的裁切檔名"])), encoding="utf-8")
        print(f"   👉 清單：{out_dir / '找不到清單.txt'}（A4b 可能沒還原出這幾張）")
    if n_size:
        print(f"   ⚠️ {n_size} 張的尺寸與資料集影像不同（見 {MANIFEST_OUT} 的「尺寸」欄）")
        print(f"      YOLO 的 label 是正規化座標，只要兩張是**同一塊區域**、只差解析度，")
        print(f"      label 仍然對得上；但若裁切框本身不一樣，label 就會偏掉。")
        print(f"      不放心的話用 A4d（會逐張驗證並對齊）。")
    print(f"📍 輸出: {out_dir}（檔名與原資料集相同）")
    print(f"📍 對照表: {out_dir / MANIFEST_OUT}")
    print("=======================================================")


def main():
    a = sys.argv[1:]
    if not a:
        process()
    elif len(a) == 3:
        process(*a)
    else:
        print("用法:")
        print(f"  python {Path(__file__).name}")
        print(f"  python {Path(__file__).name} <已標註資料集> <無CLAHE裁切圖> <輸出>")
        sys.exit(1)


if __name__ == "__main__":
    main()
