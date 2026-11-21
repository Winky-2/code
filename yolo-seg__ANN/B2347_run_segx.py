"""
B234x_run_segx.py
===============================================================
新分割模型（nnU-Net / MedSAM / SAM2 / PointRend / Mask2Former）→ B2 → B3 → B4 → B7。
B2/B3/B4/B34/B7 原檔一行不改；本檔只換三樣：
  1) 權重路徑 → 選定模型的 weights_ready.pt
  2) 輸出檔名後綴 TAG（不會蓋到 YOLO 的結果），檔名不帶 seed
  3) ultralytics.YOLO 換成本檔的分派器：遇到新模型的權重，就載入「訓練它的那支 B1」
     裡的推論器（config.json 記著是哪支）；其他權重照舊
→ 剔除碎塊、挑牙、PCA 幾何、B4 篩選、B7 正規化都與 YOLO 時相同，唯一變因是分割模型。

前提：已跑完對應的 B1；本檔與該 B1、B2_seg_inference_seed_v5、B3_mask_assist_and_compare_v3、
      B4_build_C1_input、B34_run_seeds_v2、B7_size_normalize_seed 放同一資料夾。
      B7 的 FULL_IMAGE_DIR、REF 沿用 B7 檔裡的設定（本檔不覆寫）。
產出（以 nnU-Net 為例，TAG = nnunet）：
  B2_seg推論統計_nnunet.xlsx、B2_seg視覺化_nnunet\、B2_seed彙整_nnunet.xlsx
  B3_mask輔助對照_nnunet.xlsx、B3_mask視覺化_nnunet\、B4_配對診斷_nnunet.xlsx、B34_seed彙整_nnunet.xlsx
  根管填充物像素長度_已配對_mask幾何_nnunet.xlsx              ← C1 METHOD = 'mask幾何_nnunet'
  根管填充物像素長度_已配對_冠寬比例尺_nnunet.xlsx            ← C1 METHOD = '冠寬比例尺_nnunet'
  根管填充物像素長度_已配對_mask幾何_長邊正規化_nnunet.xlsx    ← C1 METHOD = 'mask幾何_長邊正規化_nnunet'
  B7_正規化摘要_nnunet.xlsx、B7_config_nnunet.json
  C1 的 filename 要改成 ['根管填充物像素長度_已配對_' METHOD '.xlsx']（新檔名沒有 11n_tophat_）
牙位基準 不看影像、跟分割無關，B4 照常產生後直接刪掉（同 B234m 的做法）。
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

# ==================== 可手動修改 ====================
RUN_NOW = True

METHOD = "mask2former_swint"          # 👈 nnunet2d | medsam | sam2 | pointrend_r50fpn  | mask2former_swint
DATA_TAG = "train-308_tophat"  # 👈 與各 B1 共用設定區的 DATA_TAG 相同
RUN_B2, RUN_B34, RUN_B7 = True, True, True

# test 影像：B2 與 B3 一定要同一組，而且要跟 YOLO 跑 B2/B3 時一樣（None = 沿用 B2/B3 檔裡的設定）
TEST_IMAGE_DIR = "padded_test-77_tophat/images"      # 👈 改成你 YOLO 那次 B3 的 IMAGE_DIR
TEST_LABEL_DIR = "padded_test-77_tophat/labels"      # 👈 改成你 YOLO 那次 B3 的 GT_LABEL_DIR
# ===================================================

# ⚠️ 根目錄的 B2_seg_inference_v4.py 是「單一權重版」（SEG_WEIGHTS，沒有 WEIGHTS_TMPL）：
#    本檔對它設的路徑全部無效 → 會默默跑 YOLO、還蓋掉 B2_seg推論統計_11n.xlsx，所以這裡排除它
B2_CANDIDATES = ["B2_seg_inference_seed_v5", "B2_seg_inference_v5"]
B7_CANDIDATES = ["B7_size_normalize_seed", "B7_size_normalize_v2"]
B2_REQUIRED = ["TRAIN_SEEDS", "WEIGHTS_TMPL", "OUTPUT_TMPL", "VIS_TMPL", "SUMMARY_XLSX"]
B7_REQUIRED = ["TRAIN_SEEDS", "METHODS", "INPUT_TMPL", "NORM"]
B3_CANDIDATES = ["B3_mask_assist_and_compare_v3"]
B34_CANDIDATES = ["B34_run_seeds_v2"]
B34_REQUIRED = ["TRAIN_SEEDS", "B3_MODULE", "B4_MODULE", "SEG_WEIGHTS_TMPL", "B2_XLSX_TMPL",
                "B3_OUT_TMPL", "B3_VIS_TMPL", "B4_RAW_DIR_TMPL", "B4_DIAG_TMPL", "FINAL_TMPL", "SUMMARY_XLSX"]
B4_MODULE = "B4_build_C1_input"
PREFIX_NAME = "根管填充物像素長度_已配對"           # 與 B4 / B34 / B7 / C1 一致
DROP_LABELS = ["牙位基準"]                         # 不看影像，跟分割無關 → 不輸出
B7_SIDE_FILES = ["B7_正規化摘要.xlsx", "B7_config.json"]   # B7 寫死的檔名，跑完改名加 TAG
SEED = 0                                            # B2/B34/B7 都是多 seed 版，這裡只跑一個

TAGS = {"nnunet2d": "nnunet", "medsam": "medsam", "sam2": "sam2", "pointrend_r50fpn": "pointrend",
        "d2maskrcnn_r50fpn": "d2mrcnn", "mask2former_swint": "m2f"}
TAG = TAGS[METHOD]
WEIGHTS = f"{METHOD}_run_{DATA_TAG}/weights_ready.pt"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# config.json 沒寫 SCRIPT 時（舊版訓練出來的）依 ARCH 找
ARCH_SCRIPT = {"segx_nnunet2d": "B1n_train_nnunet.py", "segx_medsam": "B1sm_train_medsam.py",
               "segx_sam2": "B1s2_train_sam2.py", "segx_d2": "B1p_train_pointrend.py",
               "segx_mask2former": "B1f_train_mask2former.py"}
B1_SCRIPTS = list(dict.fromkeys(ARCH_SCRIPT.values()))


# ============================================================
# 分派器：讓 B2/B3 的 YOLO(權重) 也能載入新模型（B1x_compare_val 也用這裡）
# ============================================================

def _cfg_of(path):
    p = Path(path)
    c = (p if p.is_dir() else p.parent) / "config.json"
    try:
        return json.loads(c.read_text(encoding="utf-8")) if c.exists() else None
    except (ValueError, OSError):
        return None


def is_segx_run(path):
    cfg = _cfg_of(path)
    return bool(cfg) and str(cfg.get("ARCH", "")).startswith("segx_")


def load_b1_module(script):
    """把某支 B1 當模組載入（它的 main 在 __main__ 判斷裡，不會被執行）。"""
    p = HERE / script
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p}（推論器在訓練用的那支 B1 裡，要放同一資料夾）")
    name = p.stem
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _find_b1(arch=None):
    """依檔案內容找新版 B1（檔名可以自己改，例如 B1a_train_nnunet.py）。
    arch=None → 任一支；否則找 ARCH = "<arch>" 的那支。"""
    for p in sorted(HERE.glob("B1*.py")):
        try:
            src = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "def check_same_as_b1" not in src or "def build_backend" not in src:
            continue
        if arch is None or f'ARCH = "{arch}"' in src:
            return p.name
    return None


def any_b1_module():
    """拿任一支 B1 當工具庫（切分、評估器在五支裡都一樣）。"""
    name = _find_b1()
    if name is None:
        raise FileNotFoundError(f"{HERE} 裡找不到任何一支新的 B1（nnU-Net/MedSAM/SAM2/PointRend/Mask2Former）")
    return load_b1_module(name)


def module_of_run(path):
    """先用 config.json 記的檔名；改過名找不到時，依 ARCH 從檔案內容找。"""
    cfg = _cfg_of(path)
    name = cfg.get("SCRIPT")
    if not name or not (HERE / name).exists():
        name = _find_b1(cfg["ARCH"]) or ARCH_SCRIPT[cfg["ARCH"]]
    return load_b1_module(name)


def is_any_custom_run(path):
    """segx 權重；有 _maskrcnn_common（B1m）的話也認得它的權重。"""
    if is_segx_run(path):
        return True
    try:
        from _maskrcnn_common import is_maskrcnn_run
        return is_maskrcnn_run(path)
    except ImportError:
        return False


def load_seg_model(path):
    """新模型 → 該 B1 的 SegxModel；B1m → MaskRCNNSeg；其他 → ultralytics.YOLO。"""
    if is_segx_run(path):
        return module_of_run(path).SegxModel(path)
    try:
        from _maskrcnn_common import MaskRCNNSeg, is_maskrcnn_run
        if is_maskrcnn_run(path):
            return MaskRCNNSeg(path)
    except ImportError:
        pass
    import ultralytics
    return getattr(ultralytics.YOLO, "_segx_orig", ultralytics.YOLO)(path)


def patch_ultralytics_yolo():
    """把 ultralytics.YOLO 換成分派器（只影響目前這個 Python 行程）。
    必須在 import B2/B3 之前呼叫：B2 在模組層 from ultralytics import YOLO。"""
    import ultralytics
    orig = ultralytics.YOLO
    if getattr(orig, "_segx_dispatch", False):
        return
    orig = getattr(orig, "_segx_orig", orig)

    def YOLO(model="yolo11n.pt", *args, **kwargs):          # noqa: N802
        if isinstance(model, (str, Path)) and is_any_custom_run(model):
            return load_seg_model(model)
        return orig(model, *args, **kwargs)

    YOLO._segx_dispatch = True
    YOLO._segx_orig = orig
    ultralytics.YOLO = YOLO


# ============================================================
# 主流程
# ============================================================

def pick_checked(cands, required, what):
    """找第一個存在、而且設定名稱齊全的檔。用讀文字檢查，不 import
    （B2 在模組層 from ultralytics import YOLO，太早 import 會綁到沒換過的 YOLO）。"""
    for m in cands:
        p = HERE / f"{m}.py"
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        miss = [k for k in required if f"\n{k} =" not in src]
        if miss:
            raise RuntimeError(f"{p.name} 缺少設定 {miss}，本檔無法改它的路徑 → 會默默跑錯，已停止")
        return m
    raise FileNotFoundError(f"{HERE} 裡找不到 {what}：{', '.join(c + '.py' for c in cands)}")


def run_b2(mod):
    b2 = importlib.import_module(mod)
    if TEST_IMAGE_DIR:
        b2.IMAGE_DIR = TEST_IMAGE_DIR
    if TEST_LABEL_DIR:
        b2.LABEL_DIR = TEST_LABEL_DIR
    print(f"🖼️  B2 test 影像：{b2.IMAGE_DIR}（標註 {b2.LABEL_DIR}）")
    b2.TRAIN_SEEDS = [SEED]
    b2.WEIGHTS_TMPL = WEIGHTS
    b2.OUTPUT_TMPL = f"B2_seg推論統計_{TAG}.xlsx"
    b2.VIS_TMPL = f"B2_seg視覺化_{TAG}"
    b2.SUMMARY_XLSX = f"B2_seed彙整_{TAG}.xlsx"
    print(f"\n==================== B2（{TAG}）====================")
    b2.main()


def drop_b4_labels():
    """包住 B4.main：寫完後刪掉 DROP_LABELS 的檔。B4 的篩選照舊 → 列數、fold 跟單跑 B4 相同。"""
    b4 = importlib.import_module(B4_MODULE)
    if getattr(b4.main, "_dropped", False) or not DROP_LABELS:
        return
    orig = b4.main

    def main():
        orig()
        for lb in DROP_LABELS:
            p = Path(f"{b4.OUTPUT_PREFIX}_{lb}.xlsx")
            if p.exists():
                p.unlink()
                print(f"🗑️  不輸出「{lb}」：{p.name}")

    main._dropped = True
    b4.main = main


def run_b34(b3_mod, b34_mod):
    b3 = importlib.import_module(b3_mod)            # B34 之後 import 拿到的是同一個物件
    if TEST_IMAGE_DIR:
        b3.IMAGE_DIR = TEST_IMAGE_DIR
    if TEST_LABEL_DIR:
        b3.GT_LABEL_DIR = TEST_LABEL_DIR
    print(f"🖼️  B3 test 影像：{b3.IMAGE_DIR}（GT {b3.GT_LABEL_DIR}）；"
          f"scale 原圖 {b3.ORIGINAL_IMAGE_DIR}（沿用 B3 檔裡的設定）")
    for d in (b3.IMAGE_DIR, b3.GT_LABEL_DIR):
        if d and not (HERE / d).exists():
            raise FileNotFoundError(f"找不到 {d}：改本檔的 TEST_IMAGE_DIR / TEST_LABEL_DIR")
    b34 = importlib.import_module(b34_mod)
    b34.B3_MODULE = b3_mod
    b34.B4_MODULE = B4_MODULE
    b34.TRAIN_SEEDS = [SEED]
    b34.SEG_WEIGHTS_TMPL = WEIGHTS
    b34.B2_XLSX_TMPL = f"B2_seg推論統計_{TAG}.xlsx"
    b34.B3_OUT_TMPL = f"B3_mask輔助對照_{TAG}.xlsx"
    b34.B3_VIS_TMPL = f"B3_mask視覺化_{TAG}"
    b34.B4_RAW_DIR_TMPL = f"B4_raw_{TAG}"
    b34.B4_DIAG_TMPL = f"B4_配對診斷_{TAG}.xlsx"
    b34.FINAL_TMPL = PREFIX_NAME + f"_{{label}}_{TAG}.xlsx"
    b34.SUMMARY_XLSX = f"B34_seed彙整_{TAG}.xlsx"
    b34.RUN_NOW = True
    drop_b4_labels()
    print(f"\n==================== B3 → B4（{TAG}）====================")
    b34.main()
    print(f"⚠️ 上面 B34 印的「C1 的 METHOD」是它寫死的格式，以本檔最後印的為準")


def run_b7(mod):
    """呼叫多 seed 版 B7，只換輸入方法；輸出改成不帶 seed 的檔名，並保護 YOLO 那份 B7 摘要。"""
    b7 = importlib.import_module(mod)
    label = f"mask幾何_{TAG}"
    src = HERE / f"{PREFIX_NAME}_{label}.xlsx"
    if not src.exists():
        raise FileNotFoundError(f"找不到 {src.name}（先跑 B34：RUN_B34 = True）")
    b7.TRAIN_SEEDS = [SEED]
    b7.METHODS = [label]
    b7.INPUT_TMPL = PREFIX_NAME + "_{label}.xlsx"
    b7.NORM = "長邊"
    b7.RUN_NOW = True

    backup = {}
    for n in B7_SIDE_FILES:                       # 先把 YOLO 那份移開
        p = HERE / n
        if p.exists():
            q = HERE / f"_暫存_{n}"
            os.replace(p, q)
            backup[n] = q
    try:
        b7.main()
    finally:
        for n in B7_SIDE_FILES:
            p = HERE / n
            if p.exists():                        # 這次的 → 加 TAG
                os.replace(p, HERE / f"{Path(n).stem}_{TAG}{Path(n).suffix}")
            if n in backup:                       # YOLO 那份放回原位
                os.replace(backup[n], p)
    # B7 一定會加 _seed{s} → 改掉
    out_seed = HERE / f"{PREFIX_NAME}_{label}_長邊正規化_seed{SEED}.xlsx"
    out = HERE / f"{PREFIX_NAME}_{label}_長邊正規化.xlsx"
    if out_seed.exists():
        os.replace(out_seed, out)
        print(f"📄 {out.name}")


def main():
    os.chdir(HERE)                                # 相對路徑一律以本資料夾為準
    if not (HERE / WEIGHTS).exists() or not is_segx_run(HERE / WEIGHTS):
        raise FileNotFoundError(f"找不到 {WEIGHTS} 或旁邊的 config.json（先跑對應的 B1 訓練腳本）")
    module_of_run(HERE / WEIGHTS)                 # 先確認對應的 B1 在資料夾裡
    b2_mod = pick_checked(B2_CANDIDATES, B2_REQUIRED, "多 seed 版 B2") if RUN_B2 else "-"
    b3_mod = b34_mod = "-"
    if RUN_B34:
        b3_mod = pick_checked(B3_CANDIDATES, [], "B3")
        b34_mod = pick_checked(B34_CANDIDATES, B34_REQUIRED, "B34")
        pick_checked([B4_MODULE], [], "B4")
    b7_mod = pick_checked(B7_CANDIDATES, B7_REQUIRED, "多 seed 版 B7") if RUN_B7 else "-"
    print(f"🔁 {METHOD} → TAG = {TAG}，權重 {WEIGHTS}")
    print(f"使用版本：B2={b2_mod}  B3={b3_mod}  B34={b34_mod}  B7={b7_mod}")
    for d in (TEST_IMAGE_DIR, TEST_LABEL_DIR):
        if d and not (HERE / d).exists():
            raise FileNotFoundError(f"找不到 {d}：改本檔開頭的 TEST_IMAGE_DIR / TEST_LABEL_DIR")

    if RUN_B2 or RUN_B34:
        patch_ultralytics_yolo()                  # 必須在 import B2/B3 之前
    if RUN_B2:
        run_b2(b2_mod)
    if RUN_B34:
        run_b34(b3_mod, b34_mod)
    if RUN_B7:
        print(f"\n==================== B7 長邊正規化（{TAG}）====================")
        run_b7(b7_mod)

    print(f"\n👉 C1：filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];")
    print(f"   METHOD = 'mask幾何_{TAG}'  |  'mask幾何_長邊正規化_{TAG}'  |  '冠寬比例尺_{TAG}'")
    print("   ⚠️ 跟 YOLO 比之前，先確認兩邊 C1 輸入檔的 圖片檔名 / 折數fold 完全相同")


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)
