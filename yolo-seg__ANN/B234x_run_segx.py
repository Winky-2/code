"""
B234x_run_segx.py
===============================================================
新分割模型（nnU-Net / MedSAM / SAM2 / PointRend / Mask2Former）→ B2 → B3 → B4 → B7。
直接重用 B234m_run_maskrcnn.py 的流程（B2/B3/B4/B34/B7 原檔一行不改），只換三樣：
  1) 權重路徑 → 選定模型的 weights_ready.pt
  2) 輸出檔名後綴 TAG（不會蓋到 YOLO 或 Mask R-CNN 的結果）
  3) ultralytics.YOLO 換成本檔的分派器：遇到新模型的權重，就載入「訓練它的那支 B1」
     裡的推論器（config.json 記著是哪支）；B1m 權重交給 _maskrcnn_common；其他照舊
→ 剔除碎塊、挑牙、PCA 幾何、B4 對齊、B7 正規化都與 YOLO 時相同，唯一變因是分割模型。

前提：已跑完對應的 B1；本檔與該 B1、B234m_run_maskrcnn.py、_maskrcnn_common.py、
      _unet_common.py、B2/B3/B4/B34/B7 放同一資料夾。
產出：同 B234m，檔名後綴換成 TAG（例：B3_mask輔助對照_nnunet_seed0.xlsx）
      C1 輸入：根管填充物像素長度_已配對_mask幾何_{TAG}_seed0.xlsx（及 _長邊正規化_ 版）
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

# ==================== 可手動修改 ====================
RUN_NOW = True

METHOD = "nnunet2d"          # 👈 nnunet2d | medsam | sam2 | pointrend_r50fpn | d2maskrcnn_r50fpn | mask2former_swint
DATA_TAG = "train-308_tophat"  # 👈 與各 B1 共用設定區的 DATA_TAG 相同
RUN_B2, RUN_B34, RUN_B7 = True, True, True
# ===================================================

# B234m 預設只認 B2_seg_inference_v5 / v4、B7_size_normalize_v2；你資料夾裡的檔名是 *_seed_v5 / *_seed
# ⚠️ 根目錄的 B2_seg_inference_v4.py 是「單一權重版」（SEG_WEIGHTS，沒有 WEIGHTS_TMPL）：
#    B234m 對它設的路徑全部無效 → 會默默跑 YOLO、還蓋掉 B2_seg推論統計_11n.xlsx，所以這裡排除它
B2_CANDIDATES = ["B2_seg_inference_seed_v5", "B2_seg_inference_v5"]
B7_CANDIDATES = ["B7_size_normalize_seed", "B7_size_normalize_v2"]
B2_REQUIRED = ["TRAIN_SEEDS", "WEIGHTS_TMPL", "OUTPUT_TMPL", "VIS_TMPL", "SUMMARY_XLSX"]
B7_REQUIRED = ["TRAIN_SEEDS", "METHODS", "INPUT_TMPL", "NORM"]

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


def any_b1_module():
    """拿任一支 B1 當工具庫（切分、評估器在五支裡都一樣）。"""
    for s in B1_SCRIPTS:
        if (HERE / s).exists():
            return load_b1_module(s)
    raise FileNotFoundError(f"{HERE} 裡找不到任何一支新的 B1：{B1_SCRIPTS}")


def module_of_run(path):
    cfg = _cfg_of(path)
    return load_b1_module(cfg.get("SCRIPT") or ARCH_SCRIPT[cfg["ARCH"]])


def is_any_custom_run(path):
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
            raise RuntimeError(f"{p.name} 缺少設定 {miss}，B234m 無法改它的路徑 → 會默默跑錯權重，已停止")
        return m
    raise FileNotFoundError(f"{HERE} 裡找不到 {what}：{', '.join(c + '.py' for c in cands)}")


def main():
    import _maskrcnn_common as mc
    import B234m_run_maskrcnn as base

    if not (HERE / WEIGHTS).exists() or not is_segx_run(HERE / WEIGHTS):
        raise FileNotFoundError(f"找不到 {WEIGHTS} 或旁邊的 config.json（先跑對應的 B1 訓練腳本）")
    module_of_run(HERE / WEIGHTS)                # 先確認對應的 B1 在資料夾裡
    if RUN_B2:
        base.B2_CANDIDATES = [pick_checked(B2_CANDIDATES, B2_REQUIRED, "多 seed 版 B2")]
    if RUN_B7:
        base.B7_MODULE = pick_checked(B7_CANDIDATES, B7_REQUIRED, "多 seed 版 B7")
    # B234m.main() 在函式內才 from _maskrcnn_common import ...，所以換掉模組屬性就生效
    mc.is_maskrcnn_run = is_any_custom_run
    mc.patch_ultralytics_yolo = patch_ultralytics_yolo

    base.TAG = TAG
    base.TRAIN_SEEDS = [0]
    base.WEIGHTS_TMPL = WEIGHTS                  # 沒有 {s}，seed0 = 這一份
    base.B2_XLSX_TMPL = f"B2_seg推論統計_{TAG}_seed{{s}}.xlsx"
    base.RUN_B2, base.RUN_B34, base.RUN_B7 = RUN_B2, RUN_B34, RUN_B7
    print(f"🔁 {METHOD} → TAG = {TAG}，權重 {WEIGHTS}")
    base.main()
    print("⚠️ B234m 印出的文字裡寫「Mask R-CNN」的地方，這次指的是", METHOD)
    print(f"👉 C1：filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx']，"
          f"METHOD = 'mask幾何_{TAG}_seed0'（或 'mask幾何_{TAG}_長邊正規化_seed0'）")


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)
