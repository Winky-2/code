"""
B8_tta_ensemble_test.py
===============================================================
不重訓的推論改良 ablation（test 集）：水平翻轉 TTA、三模型融合。
設定全部寫死在檔頭 → 跑一次、全部實驗都報告、跑完不回頭改設定（避免拿 test 調參）。

實驗（一次一個變因）：
  E0_raw  各模型原推論，不截斷（= 原本的結果，用來核對）
  E0      各模型原推論，只留分數前 TOPK 個（baseline；應與 E0_raw 幾乎相同）
  E1      各模型 + 水平翻轉 TTA
  E2      三模型融合（不加 TTA）
  E3      三模型融合 + TTA

融合方法（與模型無關，只用各 B1 的 predict_instances 回傳的二值 mask）：
  1. 所有來源的 instance 依分數排序，貪婪分群（mask IoU ≥ MATCH_IOU，同一來源一群最多一個）
  2. 群內每個 mask 轉成 signed distance（內正外負）取平均，> 0 的像素 = 融合 mask
     → 等於取「平均邊界」；兩個來源時，邊界落在兩者中間
  3. 分數 = 群內分數總和 / 來源數（某來源沒偵測到就扣分，同 WBF 的做法）
⚠️ 這是二值 mask 的邊界平均，不是機率圖平均（各 B1 只回二值 mask），效果可能比機率平均小。
⚠️ 訓練時沒有翻轉增強（fliplr=0、nnU-Net NoMirroring）→ 翻轉影像模型沒看過，E1 可能反而變差；這正是要驗的。
⚠️ 各模型的分數定義不同（例如 nnU-Net = 前景機率平均），融合時直接平均，mAP 的排序可能不完美。

指標：
  Box/Mask P、R、mAP50、mAP50-95：B1 同一個評估器（C.evaluate，conf = VAL_CONF）
  IoU/Dice：每張圖取「分數最高的預測」對 GT（多個 GT 取 IoU 最大者）
            → 逐張平均（Mask）與全體像素合併（Pooled）
  p 值：逐張 IoU 的配對 Wilcoxon。E1 對同模型 E0；E2/E3 對 REF_MODEL 的 E0
  ⚠️ IoU/Dice 的定義不一定與簡報第 8 頁相同 → 只比本檔內的相對差異；
     E0 的 mAP 若與簡報不同，代表評估設定不同，一樣只看本檔內的差。

前提：本檔與 B234x_run_segx.py（或 B2347_run_segx.py）、各模型的 B1 放同一資料夾；
      權重路徑與 test 影像沿用 B234x 檔頭的 DATA_TAG、TEST_IMAGE_DIR、TEST_LABEL_DIR。
操作（PowerShell）：  python .\\B8_tta_ensemble_test.py
產出：B8_TTA融合_test.xlsx（總表、逐張IoU、設定）
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

R = None
for _name in ("B2347_run_segx"):
    try:
        R = importlib.import_module(_name)
        break
    except ModuleNotFoundError:
        continue
if R is None:
    raise SystemExit("❌ 找不到 B234x_run_segx.py（或 B2347_run_segx.py），要放在同一資料夾")

# ==================== 可手動修改（跑之前定好，跑完不再改） ====================
RUN_NOW = True

MODELS = ["mask2former_swint", "medsam", "nnunet2d", "pointrend_r50fpn", "sam2"]  # E0/E1
ENSEMBLE = ["nnunet2d", "pointrend_r50fpn", "mask2former_swint"]                   # E2/E3 融合這三個
REF_MODEL = "nnunet2d"        # E2/E3 的 p 值對照
VIEWS = ["orig", "hflip"]     # TTA 只做水平翻轉；不上下翻轉（牙冠/牙根方向不能反）、不縮放
MATCH_IOU = 0.5               # 分群門檻
TOPK = 20                     # 每個來源只留分數前 K 個（分群計算量才不會爆）
OUT_XLSX = "B8_TTA融合_test.xlsx"
# =================================================================================

TEST_IMAGE_DIR = R.TEST_IMAGE_DIR
TEST_LABEL_DIR = R.TEST_LABEL_DIR
DATA_TAG = R.DATA_TAG
TAGS = R.TAGS


def weights_of(method):
    return HERE / f"{method}_run_{DATA_TAG}" / "weights_ready.pt"


# ============================================================
# 資料
# ============================================================

def test_items():
    img_dir, lbl_dir = HERE / TEST_IMAGE_DIR, HERE / TEST_LABEL_DIR
    for d in (img_dir, lbl_dir):
        if not d.exists():
            raise FileNotFoundError(f"找不到 {d}：改 B234x 檔頭的 TEST_IMAGE_DIR / TEST_LABEL_DIR")
    items = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        lb = lbl_dir / f"{p.stem}.txt"
        if lb.exists():
            items.append({"img_id": p.stem, "img_path": p, "label_path": lb})
    print(f"🖼️  test：{len(items)} 張（{TEST_IMAGE_DIR}）")
    return items


def img_hash(img):
    return hashlib.md5(img.tobytes()).hexdigest()


# ============================================================
# instance：存成 (box[x0,y0,x1,y1], 裁切後的 bool mask, 分數)，省記憶體
# ============================================================

def to_inst(mask, score):
    ys, xs = np.nonzero(mask)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return np.array([x0, y0, x1, y1], int), mask[y0:y1, x0:x1].copy(), float(score)


def to_full(inst, H, W):
    b, m, _ = inst
    full = np.zeros((H, W), bool)
    full[b[1]:b[3], b[0]:b[2]] = m
    return full


def topk(insts):
    return sorted(insts, key=lambda t: -t[2])[:TOPK]


def iou_inst(a, b):
    (ba, ma, _), (bb, mb, _) = a, b
    x0, y0 = max(ba[0], bb[0]), max(ba[1], bb[1])
    x1, y1 = min(ba[2], bb[2]), min(ba[3], bb[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ia = ma[y0 - ba[1]:y1 - ba[1], x0 - ba[0]:x1 - ba[0]]
    ib = mb[y0 - bb[1]:y1 - bb[1], x0 - bb[0]:x1 - bb[0]]
    inter = np.count_nonzero(ia & ib)
    return inter / (ma.sum() + mb.sum() - inter + 1e-9)


def sdf_fuse(members):
    """群內 mask 的 signed distance 平均 → > 0 的像素。回傳 (box, crop) 或 None。"""
    if len(members) == 1:
        return members[0][0], members[0][1]
    pad = 2
    X0 = min(t[0][0] for t in members) - pad
    Y0 = min(t[0][1] for t in members) - pad
    X1 = max(t[0][2] for t in members) + pad
    Y1 = max(t[0][3] for t in members) + pad
    acc = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    for b, m, _ in members:
        full = np.zeros_like(acc, np.uint8)
        full[b[1] - Y0:b[3] - Y0, b[0] - X0:b[2] - X0] = m
        din = cv2.distanceTransform(full, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(1 - full, cv2.DIST_L2, 5)
        acc += din - dout
    f = acc > 0
    if not f.any():
        return None
    ys, xs = np.nonzero(f)
    box = np.array([X0 + xs.min(), Y0 + ys.min(), X0 + xs.max() + 1, Y0 + ys.max() + 1], int)
    return box, f[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()


def fuse(sets):
    """sets：多個來源（翻轉視角或不同模型）的 instance 清單 → 融合後的清單。"""
    n = len(sets)
    if n == 1:
        return topk(sets[0])
    pool = sorted([(s, t) for s, insts in enumerate(sets) for t in insts], key=lambda p: -p[1][2])
    used = [False] * len(pool)
    out = []
    for i, (si, ti) in enumerate(pool):
        if used[i]:
            continue
        used[i] = True
        grp, srcs = [ti], {si}
        for j in range(i + 1, len(pool)):
            sj, tj = pool[j]
            if used[j] or sj in srcs:
                continue
            if iou_inst(ti, tj) >= MATCH_IOU:
                used[j] = True
                grp.append(tj)
                srcs.add(sj)
        f = sdf_fuse(grp)
        if f is not None:
            out.append((f[0], f[1], sum(t[2] for t in grp) / n))
    return topk(out)


# ============================================================
# 推論（每個模型跑一次，所有視角的結果先存起來）
# ============================================================

def run_model(method, items, C):
    w = weights_of(method)
    if not w.exists() or not R.is_segx_run(w):
        print(f"— 略過 {method}：找不到 {w.relative_to(HERE)} 或旁邊的 config.json")
        return None
    print(f"\n▶ 推論 {method}（視角 {VIEWS}）")
    backend = R.module_of_run(w).build_backend(w)
    out = {}
    for k, it in enumerate(items):
        img = C.imread_rgb(it["img_path"])
        d = {}
        for v in VIEWS:
            x = img if v == "orig" else np.ascontiguousarray(img[:, ::-1])
            masks, scores = backend.predict_instances(x, C.VAL_CONF)
            insts = []
            for m, s in zip(masks, np.asarray(scores, np.float64).ravel()):
                m = np.asarray(m, bool)
                if v == "hflip":
                    m = np.ascontiguousarray(m[:, ::-1])
                if m.any():
                    insts.append(to_inst(m, s))
            d[v] = insts
        out[it["img_id"]] = d
        if (k + 1) % 20 == 0:
            print(f"   {k + 1}/{len(items)}")
    del backend
    try:
        import torch
        torch.cuda.empty_cache()
    except ImportError:
        pass
    return out


# ============================================================
# 評估
# ============================================================

def evaluate_preds(preds, items, C, hash2id, shapes):
    def predict_fn(img, conf):
        iid = hash2id[img_hash(img)]
        H, W = shapes[iid]
        insts = preds[iid]
        return [to_full(t, H, W) for t in insts], np.array([t[2] for t in insts])
    return C.evaluate(predict_fn, items)


def top1_iou(preds, items, C, shapes):
    """每張取分數最高的預測 vs GT（多個 GT 取 IoU 最大）。"""
    rows = []
    for it in items:
        iid = it["img_id"]
        H, W = shapes[iid]
        gts = C.load_instances(it["label_path"], H, W)
        if not gts:
            continue
        insts = preds[iid]
        if insts:
            p = to_full(max(insts, key=lambda t: t[2]), H, W)
        else:
            p = np.zeros((H, W), bool)
        best = None
        for g in gts:
            inter = np.count_nonzero(p & g)
            union = np.count_nonzero(p | g)
            iou = inter / union if union else 0.0
            if best is None or iou > best[0]:
                best = (iou, inter, union, int(p.sum()), int(g.sum()))
        iou, inter, union, sp, sg = best
        rows.append({"img_id": iid, "IoU": iou, "Dice": 2 * inter / (sp + sg) if sp + sg else 0.0,
                     "inter": inter, "union": union, "sp": sp, "sg": sg})
    return pd.DataFrame(rows).set_index("img_id")


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return np.nan
    d = (a - b).dropna()
    if len(d) < 5 or np.allclose(d, 0):
        return np.nan
    try:
        return float(wilcoxon(d).pvalue)
    except ValueError:
        return np.nan


# ============================================================
# 主流程
# ============================================================

def main():
    C = R.any_b1_module()
    items = test_items()
    shapes, hash2id = {}, {}
    for it in items:
        img = C.imread_rgb(it["img_path"])
        shapes[it["img_id"]] = img.shape[:2]
        hash2id[img_hash(img)] = it["img_id"]
    if len(hash2id) != len(items):
        print("⚠️ 有內容完全相同的 test 影像，逐張對應可能錯置")

    raw = {}
    for m in dict.fromkeys(MODELS + ENSEMBLE):
        r = run_model(m, items, C)
        if r is not None:
            raw[m] = r

    exps = []      # (實驗, 模型, 說明, preds, 對照實驗名)
    E0, E1 = {}, {}
    for m in MODELS:
        if m not in raw:
            continue
        tag = TAGS.get(m, m)
        p_raw = {i: d["orig"] for i, d in raw[m].items()}
        E0[m] = {i: topk(v) for i, v in p_raw.items()}
        E1[m] = {i: fuse([topk(d[v]) for v in VIEWS]) for i, d in raw[m].items()}
        exps += [("E0_raw", tag, "原推論（不截斷）", p_raw, None),
                 ("E0", tag, f"原推論（前 {TOPK}）", E0[m], None),
                 ("E1", tag, "+ 水平翻轉 TTA", E1[m], f"E0|{tag}")]
    ens = [m for m in ENSEMBLE if m in raw]
    for m in ens:                                     # ENSEMBLE 有、MODELS 沒列的也補算
        if m not in E0:
            E0[m] = {i: topk(d["orig"]) for i, d in raw[m].items()}
            E1[m] = {i: fuse([topk(d[v]) for v in VIEWS]) for i, d in raw[m].items()}
    ref = f"E0|{TAGS.get(REF_MODEL, REF_MODEL)}"
    if len(ens) >= 2:
        name = "+".join(TAGS.get(m, m) for m in ens)
        ids = [it["img_id"] for it in items]
        exps += [("E2", name, "多模型融合", {i: fuse([E0[m][i] for m in ens]) for i in ids}, ref),
                 ("E3", name, "多模型融合 + TTA", {i: fuse([E1[m][i] for m in ens]) for i in ids}, ref)]
    else:
        print(f"⚠️ 融合需要至少 2 個模型，目前只有 {ens} → 略過 E2/E3")

    rows, per_img = [], {}
    for exp, tag, desc, preds, _ in exps:
        print(f"\n▶ 評估 {exp} {tag}")
        met = evaluate_preds(preds, items, C, hash2id, shapes)
        df = top1_iou(preds, items, C, shapes)
        per_img[f"{exp}|{tag}"] = df["IoU"]
        rows.append({"實驗": exp, "模型": tag, "說明": desc, **{k: met[k] for k in C.METRIC_KEYS},
                     "IoU_mean": df["IoU"].mean(), "Dice_mean": df["Dice"].mean(),
                     "IoU_pooled": df["inter"].sum() / df["union"].sum(),
                     "Dice_pooled": 2 * df["inter"].sum() / (df["sp"].sum() + df["sg"].sum()),
                     "n": len(df)})
    for row, (exp, tag, _, _, cmp) in zip(rows, exps):
        if cmp and cmp in per_img:
            a, b = per_img[f"{exp}|{tag}"], per_img[cmp]
            row["對照"] = cmp
            row["ΔIoU_mean"] = (a - b).mean()
            row["p_Wilcoxon"] = wilcoxon_p(a, b)

    res = pd.DataFrame(rows)
    for k in res.columns:
        if res[k].dtype.kind == "f":
            res[k] = res[k].round(4)
    cfg = dict(MODELS=MODELS, ENSEMBLE=ENSEMBLE, REF_MODEL=REF_MODEL, VIEWS=VIEWS, MATCH_IOU=MATCH_IOU,
               TOPK=TOPK, DATA_TAG=DATA_TAG, TEST_IMAGE_DIR=TEST_IMAGE_DIR, TEST_LABEL_DIR=TEST_LABEL_DIR,
               VAL_CONF=C.VAL_CONF, B234x=R.__name__, weights={m: str(weights_of(m)) for m in raw})
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="總表", index=False)
        pd.DataFrame(per_img).to_excel(w, sheet_name="逐張IoU")
        pd.DataFrame({"設定": list(cfg), "值": [json.dumps(v, ensure_ascii=False) for v in cfg.values()]}
                     ).to_excel(w, sheet_name="設定", index=False)

    show = ["實驗", "模型", "Mask_mAP50-95", "Box_mAP50-95", "IoU_mean", "Dice_mean",
            "IoU_pooled", "Dice_pooled"] + [c for c in ("ΔIoU_mean", "p_Wilcoxon") if c in res]
    print(f"\n===== test {len(items)} 張 =====")
    print(res[show].to_string(index=False))
    print(f"\n📄 {OUT_XLSX}")
    print("核對：E0 應與 E0_raw 幾乎相同（不同 → TOPK 截斷影響了 baseline，回報給我）")
    print("⚠️ test 只有幾十張；mAP 差 0.01 以內、p ≥ 0.05 都不要當成真的變好。設定不要回頭改。")


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)