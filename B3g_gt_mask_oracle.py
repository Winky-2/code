"""
B3g_gt_mask_oracle.py  (純 GT 版，沿用 B3 _v2 的幾何)
===============================================================
Oracle 測試：如果分割「完美」(直接用人工 GT mask)，C1 的 mm 預測能好到哪？

判讀(看 C1 的 OOF MAE，與本腳本印出的「常數平均基準」比)：
  - GT mask 也贏不了常數平均 → 瓶頸不在分割(標籤定義 / pixel pitch)，改分割模型沒用
  - GT mask 明顯贏 → 分割值得改，這個數字就是分割改到完美時的上限

*** 怎麼把 GT 塞進 B3 _v2(B3 一行都不改) ***
  _v2 在 main() 裡 YOLO(SEG_WEIGHTS).predict() 取 res.masks.xy。
  本腳本把 _v2 模組裡的 YOLO 換成 GtAsYOLO：predict() 回傳 GT 多邊形(原圖 px)，
  格式與 Ultralytics 的 masks.xy 相同。之後的碎塊剔除、挑牙、PCA 幾何、方向、
  scale、冠寬、QC 全部走 _v2 原本的程式 → 與你之前 YOLO 結果同一版幾何。
  跑完會檢查 predict 確實被呼叫(防止「以為是 GT，其實跑了 YOLO」)。

*** 純 GT：不讀任何 YOLO 結果、不產生牙位相關方法 ***
  B4：主檔 = GT 版 B3 輸出；EXTRA_B3 = {}；USE_POSITION_METHODS = False
      → 只產出一份 根管填充物像素長度_已配對_GT_mask幾何.xlsx

*** 與既有結果完全隔離 ***
所有產出都進 results/B3g/<run_tag>/，既有檔案一個都不碰：
  B3_mask輔助對照_gt.xlsx、B3_mask視覺化/(圖例的 "seg" = GT)
  根管填充物像素長度_已配對_GT_mask幾何.xlsx、B4_配對診斷.xlsx
  常數平均基準.json、config.json

前提：本檔與 B3 _v2、B4 放在同一個資料夾；B3 的 IMAGE_DIR / GT_LABEL_DIR 指向 test 集。
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

# ==================== 可手動修改 ====================
RUN_NOW = True  # False 只印說明，不執行

B3_FILE = "B3_mask_assist_and_compare_v2.py" # 👈 B3 的檔名(與本檔同資料夾)
B4_FILE = "B4_build_C1_input.py"             # 👈 B4 的檔名(與本檔同資料夾)
OUT_ROOT = "results/B3g"
METHOD_LABEL = "GT_mask幾何"                 # B4 輸出後綴 = C1 的 METHOD
SOURCE_LABEL = "gt_oracle"                   # 寫進 B3 Excel 的 mask來源 欄

# scale(letterbox px → 原圖 px)：直接沿用之前 YOLO 版 B3 輸出的「縮放比scale」欄，
# 保證 GT 版與舊結果用同一組 scale(只讀)。設成 "" 就照 B3 自己的 SCALE_* 設定。
SCALE_FROM_XLSX = "B3_mask輔助對照_11n.xlsx"
# ===================================================

HERE = Path(__file__).resolve().parent
LENGTH_COL = "AB長度_mask幾何_原圖px"        # B3 輸出的長度欄 → B4 的 X


# ============================================================
# 載入與檢查
# ============================================================

def load_script(filename, prefix):
    """依檔名載入同資料夾的腳本；找不到就列出候選檔。"""
    path = HERE / filename
    if not path.exists():
        cands = sorted(p.name for p in HERE.glob(f"{prefix}*.py"))
        raise FileNotFoundError(
            f"找不到 {path}\n"
            f"   這個資料夾裡 {prefix} 開頭的 .py：{cands or '(沒有)'}\n"
            f"   把正確檔名填進本檔頂部的 {prefix}_FILE")
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_interface(b3, b4):
    """缺這些名稱時覆寫會「靜默失效」，先擋。B4 只要求本腳本一定會覆寫的名稱；
    EXTRA_B3 / USE_POSITION_METHODS 舊版沒有也沒關係(有才關掉)。"""
    need = {b3: ("YOLO", "main", "IMAGE_DIR", "GT_LABEL_DIR", "REMOVE_FRAGMENTS",
                 "OUTPUT_XLSX", "VIS_DIR"),
            b4: ("main", "B3_XLSX", "METHODS", "OUTPUT_PREFIX", "DIAGNOSTIC_XLSX")}
    for mod, names in need.items():
        miss = [n for n in names if not hasattr(mod, n)]
        if miss:
            raise RuntimeError(f"{Path(mod.__file__).name} 缺少 {miss}，與本腳本假設的版本不同")


def new_out_dir(tag):
    """只新增不覆寫：已存在就加 _2、_3…"""
    base = Path(OUT_ROOT) / tag
    d, k = base, 2
    while d.exists():
        d, k = Path(f"{base}_{k}"), k + 1
    d.mkdir(parents=True)
    return d


# ============================================================
# 假 YOLO：predict() 回傳 GT 多邊形
# ============================================================

def is_pose_line(parts):
    """pose 格式：class cx cy w h ax ay av bx by bv，av/bv ∈ {0,1,2}(同 B3 判法)。"""
    if len(parts) != 11:
        return False
    try:
        return float(parts[7]) in (0, 1, 2) and float(parts[10]) in (0, 1, 2)
    except ValueError:
        return False


def gt_polygons(img_path, label_dir):
    """GT 標註 → 原圖 px 的多邊形清單(與 Ultralytics masks.xy 同座標系)。"""
    img = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []
    H, W = img.shape[:2]
    lp = Path(label_dir) / f"{Path(img_path).stem}.txt"
    if not lp.exists():
        return []
    polys = []
    for line in lp.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        v = parts[1:]
        if len(v) < 6 or len(v) % 2 or is_pose_line(parts):
            continue
        polys.append((np.asarray(v, np.float64).reshape(-1, 2) * [W, H]).astype(np.float32))
    return polys


class _Masks:
    def __init__(self, xy):
        self.xy = xy


class _Result:
    def __init__(self, xy):
        self.masks = _Masks(xy) if xy else None    # 沒有 GT → 同 YOLO 沒偵測到


def make_gt_yolo(label_dir, log):
    class GtAsYOLO:
        """取代 B3 裡的 ultralytics.YOLO；只實作 B3 用到的 predict(source=...)[0].masks.xy。"""
        def __init__(self, *_, **__):
            log["init"] += 1

        def predict(self, source, **_):
            xy = gt_polygons(source, label_dir)
            log["calls"] += 1
            log["no_gt" if not xy else ("multi" if len(xy) > 1 else "single")].append(
                Path(source).name)
            return [_Result(xy)]
    return GtAsYOLO


# ============================================================
# 後處理
# ============================================================

def tag_source(xlsx):
    """在逐顆牙對照加 mask來源 欄，免得日後跟 YOLO 版的 B3 輸出搞混。"""
    import pandas as pd
    sheets = pd.read_excel(xlsx, sheet_name=None)
    df = sheets.get("逐顆牙對照")
    if df is not None:
        if "mask來源" in df.columns:
            df["mask來源"] = SOURCE_LABEL
        else:
            df.insert(1, "mask來源", SOURCE_LABEL)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for name, d in sheets.items():
            d.to_excel(w, sheet_name=name, index=False)


def constant_mean_baseline(c1_input, out_json):
    """常數平均基準：每摺預測 = train 摺的平均 mm。與 C1 同一組 折數fold、同一批牙。"""
    import pandas as pd
    df = pd.read_excel(c1_input)
    y = df["填充物長度(mm)"].to_numpy(float)
    fold = df["折數fold"].to_numpy(int)
    pred = np.empty_like(y)
    for k in np.unique(fold):
        pred[fold == k] = y[fold != k].mean()
    err = pred - y
    res = {"n": int(len(y)), "folds": int(len(np.unique(fold))),
           "MAE_mm": round(float(np.abs(err).mean()), 3),
           "RMSE_mm": round(float(np.sqrt((err ** 2).mean())), 3),
           "Y_mean_mm": round(float(y.mean()), 3), "Y_std_mm": round(float(y.std(ddof=1)), 3)}
    json.dump(res, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return res


def banner(txt):
    print("\n" + "=" * 60 + f"\n{txt}\n" + "=" * 60)


# ============================================================
# 主流程
# ============================================================

def main():
    os.chdir(HERE)          # B3/B4 的相對路徑都以這個資料夾為準
    b3 = load_script(B3_FILE, "B3")
    b4 = load_script(B4_FILE, "B4")
    print(f"📜 B3 = {B3_FILE}，B4 = {B4_FILE}")
    check_interface(b3, b4)
    if not b3.GT_LABEL_DIR:
        raise ValueError("B3 的 GT_LABEL_DIR 是空的，沒有 GT 可用")

    out = new_out_dir(f"gt_oracle_frag{int(bool(b3.REMOVE_FRAGMENTS))}")
    try:
        run(b3, b4, out)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)     # 失敗就不留半成品資料夾
        print(f"\n🗑️  本次失敗，已刪除 {out}")
        raise


def apply_scale_override(b3):
    """B3 改用 SCALE_FROM_XLSX 的「縮放比scale」欄(只讀)，與舊 YOLO 結果同一組 scale。"""
    if not SCALE_FROM_XLSX:
        print(f"   scale：照 B3 自己的設定(SCALE_SOURCE={getattr(b3, 'SCALE_SOURCE', '?')})")
        return
    import pandas as pd
    p = Path(SCALE_FROM_XLSX)
    if not p.exists():
        raise FileNotFoundError(f"找不到 {p.resolve()}(SCALE_FROM_XLSX)。"
                                f"填之前 YOLO 版 B3 的輸出檔名，或設成 \"\" 改用 B3 自己的 scale 設定")
    sheet = getattr(b3, "SCALE_XLSX_SHEET", "逐顆牙對照")
    df = pd.read_excel(p, sheet_name=sheet)
    if "縮放比scale" not in df.columns or "圖片檔名" not in df.columns:
        raise KeyError(f"{p} 的「{sheet}」沒有 圖片檔名/縮放比scale 欄")
    n_ok = int(df["縮放比scale"].notna().sum())
    b3.SCALE_XLSX = str(p)
    b3.SCALE_SOURCE = "xlsx"
    print(f"   scale：沿用 {p.name} 的 縮放比scale(有值 {n_ok}/{len(df)} 張)")


def run(b3, b4, out):
    b3_xlsx = out / "B3_mask輔助對照_gt.xlsx"
    print(f"📁 本次所有產出 → {out}")
    print(f"   沿用 B3：IMAGE_DIR={b3.IMAGE_DIR}  GT_LABEL_DIR={b3.GT_LABEL_DIR}  "
          f"REMOVE_FRAGMENTS={b3.REMOVE_FRAGMENTS}")
    print(f"   沿用 B4：MM_XLSX={getattr(b4, 'MM_XLSX', '?')}  "
          f"SPLIT_SEED={getattr(b4, 'SPLIT_SEED', '?')}  N_FOLDS={getattr(b4, 'N_FOLDS', '?')}")
    apply_scale_override(b3)

    # ---- 1) B3 _v2：YOLO 換成 GT，輸出換位置 ----
    log = {"init": 0, "calls": 0, "single": [], "multi": [], "no_gt": []}
    b3.YOLO = make_gt_yolo(b3.GT_LABEL_DIR, log)
    b3.OUTPUT_XLSX = str(b3_xlsx)
    b3.VIS_DIR = str(out / "B3_mask視覺化")
    banner("B3 _v2(mask = GT 多邊形；畫面與圖例的 seg = GT)")
    b3.main()
    if log["init"] == 0 or log["calls"] == 0:
        raise RuntimeError("GT 替身沒有被呼叫 → B3 可能用了別的方式推論，結果不是 GT，勿採用")
    tag_source(b3_xlsx)
    print(f"\n🧪 GT 替身：predict 被呼叫 {log['calls']} 次；單一多邊形 {len(log['single'])}、"
          f"多個 {len(log['multi'])}(B3 用 GT 框挑)、無 seg 標註 {len(log['no_gt'])}")
    if log["no_gt"]:
        print(f"   無 seg 標註 → B3 標成 ❌seg無輸出、B4 會排除：{', '.join(log['no_gt'][:5])}"
              f"{' …' if len(log['no_gt']) > 5 else ''}")

    # ---- 2) B4：主檔 = GT，不併其他來源，不做牙位方法 ----
    b4.B3_XLSX = str(b3_xlsx)
    b4.METHODS = {METHOD_LABEL: LENGTH_COL}
    for name, val in (("EXTRA_B3", {}), ("USE_POSITION_METHODS", False)):
        if hasattr(b4, name):                    # 舊版 B4 沒有這兩項 = 本來就不會產生
            setattr(b4, name, val)
    b4.OUTPUT_PREFIX = str(out / "根管填充物像素長度_已配對")
    b4.DIAGNOSTIC_XLSX = str(out / "B4_配對診斷.xlsx")
    banner(f"B4(只有 {METHOD_LABEL}，輸出到 {out})")
    b4.main()

    # B4 實際產出的 C1 輸入檔：必須只有 GT 這一份
    made = sorted(p.name for p in out.glob("根管填充物像素長度_已配對_*.xlsx"))
    expect = f"根管填充物像素長度_已配對_{METHOD_LABEL}.xlsx"
    if expect not in made:
        raise RuntimeError(f"B4 沒產出 {expect}(實際：{made})。這版 B4 的輸出規則不同，把它上傳給我看")
    if len(made) > 1:
        print(f"⚠️ B4 另外產出了 {[m for m in made if m != expect]}：這些不是本次 oracle 的結果，忽略即可")

    # ---- 3) 常數平均基準 ----
    c1_input = out / f"根管填充物像素長度_已配對_{METHOD_LABEL}.xlsx"
    base = constant_mean_baseline(c1_input, out / "常數平均基準.json")

    json.dump({"B3_FILE": B3_FILE, "B4_FILE": B4_FILE, "mask": "GT polygons via GtAsYOLO",
               "B3_IMAGE_DIR": str(b3.IMAGE_DIR), "B3_GT_LABEL_DIR": str(b3.GT_LABEL_DIR),
               "B3_REMOVE_FRAGMENTS": bool(b3.REMOVE_FRAGMENTS),
               "B3_SCALE_SOURCE": str(getattr(b3, "SCALE_SOURCE", "")),
               "B3_SCALE_XLSX": str(getattr(b3, "SCALE_XLSX", "")),
               "B4_MM_XLSX": str(getattr(b4, "MM_XLSX", "")),
               "B4_SPLIT_SEED": getattr(b4, "SPLIT_SEED", None),
               "B4_N_FOLDS": getattr(b4, "N_FOLDS", None), "METHOD_LABEL": METHOD_LABEL},
              open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    banner("結果與下一步")
    print(f"📏 常數平均基準(不看影像，同 fold、n={base['n']})："
          f"MAE {base['MAE_mm']:.3f} mm，RMSE {base['RMSE_mm']:.3f} mm")
    print("   → 跟 C1 輸出「A_Test set表現」列的 MAE_mm 比(未扣 offset 的那列)\n")
    print("🔎 健全性：B3_mask輔助對照_gt.xlsx 的「長度vsGT」原圖 MAE 應為 0(或極接近)。")
    print("   不是 0 → GT 有多個多邊形且挑到別的那個，查「兩法選取一致」與視覺化。\n")
    print("👉 MATLAB(C1/C2 讀寫「當前資料夾」，在這裡跑就不會混到舊結果)：")
    print(f"   addpath('{HERE.as_posix()}');")
    print(f"   cd('{out.resolve().as_posix()}');")
    print(f"   C1 的 METHOD = '{METHOD_LABEL}'，其餘設定不動")


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)
        print(f"待執行設定：B3={B3_FILE}, B4={B4_FILE}, OUT_ROOT={OUT_ROOT}, "
              f"METHOD={METHOD_LABEL}")
        print("\nRUN_NOW=False：未執行。")