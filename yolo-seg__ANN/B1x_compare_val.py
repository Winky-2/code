"""
B1x_compare_val.py
===============================================================
所有分割模型在「同一組 val、同一個評估器」下的 Box/Mask mAP 對照（只讀，不訓練）。

為什麼要重算：各模型 訓練摘要.xlsx 的數字定義相同，但 YOLO 那份是 Ultralytics 內部算的
（mask IoU 解析度、Box 來源可能不同）。這裡把 YOLO 也用新 B1 裡的同一個評估器重算，
六個模型才是真正同一把尺。

⚠️ 這仍然只是 val（約 60 張）的輪廓重疊度。Mask mAP 高 ≠ 根尖準；
   最後的判準是 B234x 跑出來的 B3 長度誤差與 C1 的 MAE。

操作：直接執行（RUN_NOW=True，只讀）。找不到的權重會略過。
前提：本檔與 B234x_run_segx.py、至少一支新的 B1 放同一資料夾。
產出：B1x_val同指標對照.xlsx
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from B234x_run_segx import any_b1_module, is_segx_run, module_of_run

C = any_b1_module()          # 切分、評估器在五支 B1 裡都一樣，拿任一支來用

# ==================== 可手動修改 ====================
RUN_NOW = True

RUNS = {                                         # 顯示名稱 → 權重
    "YOLO11n-seg (B1)": f"{C.YOLO_WORK_DIR}/weights_ready.pt",
    "Mask R-CNN torchvision (B1m)": "results/B1m/maskrcnn_r50fpn_b1_img640_e50_seed0/best.pt",
    "nnU-Net 2D": f"{C.work_dir_of('nnunet2d')}/weights_ready.pt",
    "MedSAM + YOLO框": f"{C.work_dir_of('medsam')}/weights_ready.pt",
    "SAM2 + YOLO框": f"{C.work_dir_of('sam2')}/weights_ready.pt",
    "Mask R-CNN d2": f"{C.work_dir_of('d2maskrcnn_r50fpn')}/weights_ready.pt",
    "Mask R-CNN + PointRend d2": f"{C.work_dir_of('pointrend_r50fpn')}/weights_ready.pt",
    "Mask2Former Swin-T": f"{C.work_dir_of('mask2former_swint')}/weights_ready.pt",
}
OUT_XLSX = "B1x_val同指標對照.xlsx"
# ===================================================


def yolo_fn(model):
    def f(img, conf):
        r = model.predict(source=cv2.cvtColor(img, cv2.COLOR_RGB2BGR), imgsz=C.IMG_SIZE, conf=conf,
                          max_det=C.VAL_MAX_DET, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return [], np.zeros(0)
        m = r.masks.data.cpu().numpy() > 0.5
        if m.shape[1:] != img.shape[:2]:
            m = np.stack([cv2.resize(x.astype(np.uint8), img.shape[1::-1],
                                     interpolation=cv2.INTER_NEAREST) > 0 for x in m])
        return list(m), r.boxes.conf.cpu().numpy()
    return f


def mrcnn_fn(obj):
    """B1m 的 MaskRCNNSeg 只回多邊形；這裡直接拿它的模型出 mask。"""
    import torch
    from _maskrcnn_common import to_tensor

    def f(img, conf):
        obj.model.roi_heads.score_thresh = float(conf)
        with torch.no_grad():
            out = obj.model([to_tensor(img).to(obj.device)])[0]
        return list((out["masks"][:, 0] > obj.mask_thresh).cpu().numpy()), out["scores"].cpu().numpy()
    return f


def predict_fn_of(path):
    if is_segx_run(path):
        return module_of_run(path).build_backend(path).predict_instances
    try:
        from _maskrcnn_common import MaskRCNNSeg, is_maskrcnn_run
        if is_maskrcnn_run(path):
            return mrcnn_fn(MaskRCNNSeg(path))
    except ImportError:
        pass
    from ultralytics import YOLO
    return yolo_fn(YOLO(str(path)))


def data_dir_of(path):
    c = Path(path).parent / "config.json"
    if not c.exists():
        return None
    return json.loads(c.read_text(encoding="utf-8")).get("SEG_DATA_DIR")


def summary_value(path, key="Mask_mAP50-95"):
    p = Path(path).parent / "訓練摘要.xlsx"
    if not p.exists():
        return np.nan
    df = pd.read_excel(p, sheet_name="訓練摘要")
    return float(df[key].iloc[0]) if key in df else np.nan


def main():
    _, val_items, _ = C.prepare_split()
    rows = []
    for name, w in RUNS.items():
        if not Path(w).exists():
            print(f"— 略過 {name}：找不到 {w}")
            continue
        dd = data_dir_of(w)
        if dd and Path(dd) != Path(C.SEG_DATA_DIR):
            print(f"⚠️ {name} 是用 {dd} 訓練的，不是 {C.SEG_DATA_DIR} → 不是同一份資料，數字僅供參考")
        print(f"\n▶ {name}")
        m = C.evaluate(predict_fn_of(w), val_items, progress=True)
        rows.append({"模型": name, **m, "訓練摘要Mask_mAP50-95": summary_value(w),
                     "訓練資料": dd or "(無 config.json)", "權重": w})
        print(f"   Mask mAP50-95 = {m['Mask_mAP50-95']:.4f}   fitness = {m['fitness']:.4f}")
    if not rows:
        print("沒有任何權重可評估")
        return
    df = pd.DataFrame(rows).sort_values("Mask_mAP50-95", ascending=False)
    df.to_excel(OUT_XLSX, index=False)
    print(f"\n===== val {len(val_items)} 張，同一評估器 =====")
    print(df[["模型"] + C.METRIC_KEYS].to_string(index=False))
    print(f"📄 {OUT_XLSX}")
    print("⚠️ val 只有幾十張，差距在 0.01～0.02 以內不要當成真的有差；決定還是看 B3 長度誤差。")


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)
