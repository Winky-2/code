"""
2c_predict_ab_and_measure.py
============================
對應新流程圖階段：（stage 2 推論）用2b訓練好的YOLO11-pose模型
-> 預測33張test圖的A(切端)/B(根尖)關鍵點
-> 把座標還原回「原始X光片」像素空間
-> 計算AB像素長度（餵給MATLAB ANN用）
-> 跟醫師標的舊標註(ground truth)比對，算關鍵點定位誤差

*** 為什麼推論、長度、誤差三件事放同一支 ***
成本全在推論；長度與誤差都只是加減乘除，而且三者需要的幾何換算資訊
(scale / pad / 裁切座標)完全一樣，拆開會變成各讀一次manifest、
各自維護一份換算邏輯，容易改一邊忘一邊。
折衷：這支同時輸出 RAW_KEYPOINT_CSV（模型原始座標，未換算），
之後想改量法或改誤差定義，直接拿它重算即可，不用重跑推論。

================================================================
座標換算方法
================================================================
模型吃的是1c letterbox後的640x640圖，吐出來的座標在「那張640圖」的
空間裡，每顆牙縮放倍率不同，不能直接比較。換算兩步：

  第1步 去letterbox -> 裁切後圖片座標
      x_crop = (x_resized - pad_left) / scale
      y_crop = (y_resized - pad_top ) / scale
  第2步 加回裁切位移 -> 原始X光片座標
      x_orig = x_crop + crop_x1
      y_orig = y_crop + crop_y1

  長度 L = sqrt((xA-xB)^2 + (yA-yB)^2)

  第2步是純平移，不改變兩點距離，所以 L_原圖 == L_裁切後，真正影響
  長度的只有第1步除以scale。等價捷徑：L_原圖 = L_resize後 / scale。
  本腳本兩種都算並互相驗證，對不起來就示警(自檢換算有沒有寫錯、
  manifest的scale有沒有跟圖片對不上)。
  捷徑成立的前提是1c用等比例縮放(x/y同一個scale)；當初若是硬拉伸，
  斜的連線一律失真，這就是1c堅持letterbox的原因。

  注意：長度不受平移影響，但「點位誤差」會，所以裁切座標crop_x1/crop_y1
  一定要正確，否則GT比對會整組平移掉、誤差全部爆掉。

================================================================
關鍵點誤差怎麼算
================================================================
舊標註(33-all-test.yolov8/test/labels)是YOLO-pose格式，座標是相對
「原始X光片」的正規化值，乘上原圖寬高就換回像素。預測點也換算到
同一個空間，兩者才能直接相減。

預設只留下真正會影響決策的幾個數字：

  點位誤差   err_A = ||A_pred - A_gt||,  err_B = ||B_pred - B_gt||
             報中位數(不被離群點拉走，代表典型表現)與最大值(找出最爛
             的那一顆)。A、B分開報，因為根尖B通常比切端A難定位，
             分開才看得出問題出在哪一端。
  長度誤差   MAE  = 平均|預測長度 - 真實長度|，整體準度
             bias = 平均(預測長度 - 真實長度)，跟MAE分開看很重要：
                    bias是系統性偏移(正=一律量太長)，可以事後校正；
                    MAE裡的隨機成分才是真的沒救的雜訊。
             相對% = 跨圖可比(不同X光片解析度不同，絕對px不公平)
             最大值 = 最壞情況，決定要不要人工挑掉哪幾顆

下游ANN只吃長度、不吃點位，所以長度那組才是主指標；點位誤差是用來
判斷「長度準是不是矇到的」的佐證。

想深入除錯時把 DETAILED_ERROR_METRICS 打開，會多出標準差、平均值、
正規化點位誤差、命中率(PCK)、以及誤差的沿軸/垂直分解(沿軸分量會直接
變成長度誤差，垂直分量幾乎不影響)。平常看那些只是干擾。

*** A/B順序自檢 ***
如果2b訓練資料的關鍵點順序，跟舊33張的標註順序不一致(一邊A在前、
一邊B在前)，點位誤差會爆炸但不是模型爛(長度反而不受影響，因為AB
對調距離一樣)。所以每顆牙都會另外算一次「把A/B對調再比」的誤差，
對調後明顯較小就標記，超過半數被標記時印大警告。

安裝需求：
    pip install ultralytics pandas opencv-python openpyxl pillow
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ---------------- 設定區 ----------------
POSE_WEIGHTS = "yolo11_pose_run/weights_ready.pt"     # 👈 2b訓練完成的pose權重

# 要跑推論的圖片資料夾(預設1c letterbox後的33張test圖)
IMAGE_DIR = "resized_test_teeth"

# 1c輸出的manifest(已合併1b裁切資訊)，提供scale / pad / 裁切座標
GEOMETRY_MANIFEST_XLSX = "33張resize結果.xlsx"

# 舊標註(ground truth)；不想比對就把GT_LABEL_DIR設成None
GT_LABEL_DIR = "33-all-test.yolov8/test/labels"
GT_IMAGE_DIR = "33-all-test.yolov8/test/images"

OUTPUT_XLSX = "yolo像素預測.xlsx"
RAW_KEYPOINT_CSV = "yolo關鍵點原始座標.csv"
VIS_DIR = "pose_預測視覺化"
SAVE_VISUALIZATION = True

IMG_SIZE = 640
CONF_THRESHOLD = 0.15        # 跟2b一致
FALLBACK_CONF = 0.05         # 主門檻抓不到時降門檻重試(會在狀態欄標註)
KPT_CONF_WARN = 0.5          # 關鍵點信心度低於此值標「建議人工複查」

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CONSISTENCY_TOL = 1e-3               # 兩種長度算法的相對差容忍值
SWAP_MARGIN = 0.8                    # 對調後誤差 < 原本誤差*此值，才判定疑似順序顛倒

# 平常關著就好，只有在「數字怪怪的、想知道誤差是哪來的」時才打開
DETAILED_ERROR_METRICS = False
PCK_ABS_THRESHOLDS_PX = [5, 10, 20]  # 僅DETAILED時使用：絕對門檻(原圖px)
PCK_REL_THRESHOLDS_PCT = [2, 5, 10]  # 僅DETAILED時使用：相對門檻(佔GT AB長度的%)


# ============================================================
# 第1部分：幾何換算資訊
# ============================================================

def _pick_column(df, candidates):
    """1c merge之後欄名可能被加 _裁切/_resize 後綴，這裡容忍幾種可能寫法。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_geometry_manifest():
    """回傳 dict：圖片檔名 -> {scale, pad_left, pad_top, crop_x1, crop_y1}"""
    path = Path(GEOMETRY_MANIFEST_XLSX) if GEOMETRY_MANIFEST_XLSX else None
    if path is None or not path.exists():
        print(f"⚠️ 找不到幾何manifest {GEOMETRY_MANIFEST_XLSX}，所有圖都會退回"
              f"「圖片自身像素空間」，長度不可跨圖比較、也無法跟GT比對")
        return {}

    df = pd.read_excel(path)
    name_col = _pick_column(df, ["圖片檔名"])
    if name_col is None:
        print(f"⚠️ {path} 裡沒有『圖片檔名』欄位，無法對應")
        return {}

    scale_col = _pick_column(df, ["縮放倍率scale"])
    padl_col = _pick_column(df, ["補邊pad_left"])
    padt_col = _pick_column(df, ["補邊pad_top"])
    crop_col = _pick_column(df, ["裁切像素座標_x1y1x2y2"])

    geo = {}
    for _, row in df.iterrows():
        fname = str(row[name_col]).strip()
        if not fname or fname.lower() == "nan":
            continue
        scale = row[scale_col] if scale_col else None
        if scale is None or pd.isna(scale):
            continue

        crop_x1, crop_y1 = 0, 0
        if crop_col and not pd.isna(row[crop_col]):
            try:
                parts = [int(float(v)) for v in str(row[crop_col]).split(",")]
                crop_x1, crop_y1 = parts[0], parts[1]
            except (ValueError, IndexError):
                print(f"   ⚠️ {fname} 的裁切座標格式看不懂：{row[crop_col]}，當作(0,0)"
                      f"(不影響長度，但GT比對會整組平移掉，請務必修正)")

        pad_left = row[padl_col] if padl_col else None
        pad_top = row[padt_col] if padt_col else None
        geo[fname] = {
            "scale": float(scale),
            "pad_left": float(pad_left) if pad_left is not None and not pd.isna(pad_left) else 0.0,
            "pad_top": float(pad_top) if pad_top is not None and not pd.isna(pad_top) else 0.0,
            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
        }

    print(f"✅ 幾何manifest載入完成，{len(geo)} 張圖有完整的scale/pad資訊")
    return geo


# ============================================================
# 第2部分：座標換算與幾何小工具
# ============================================================

def resized_to_cropped(x, y, geo):
    return (x - geo["pad_left"]) / geo["scale"], (y - geo["pad_top"]) / geo["scale"]


def cropped_to_original(x, y, geo):
    return x + geo["crop_x1"], y + geo["crop_y1"]


def original_to_resized(x, y, geo):
    """反方向換算，把GT點畫到resize後的圖上做視覺化比對用。"""
    return ((x - geo["crop_x1"]) * geo["scale"] + geo["pad_left"],
            (y - geo["crop_y1"]) * geo["scale"] + geo["pad_top"])


def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def decompose_error(pred_pt, gt_pt, axis_from, axis_to):
    """把誤差向量分解成「沿AB軸向」與「垂直AB」兩個分量。
    沿軸分量會直接變成長度誤差，垂直分量幾乎不影響長度。
    回傳 (沿軸分量帶正負, 垂直分量絕對值)。"""
    ex = pred_pt[0] - gt_pt[0]
    ey = pred_pt[1] - gt_pt[1]
    ax = axis_to[0] - axis_from[0]
    ay = axis_to[1] - axis_from[1]
    norm = math.hypot(ax, ay)
    if norm < 1e-9:
        return 0.0, math.hypot(ex, ey)
    ux, uy = ax / norm, ay / norm
    along = ex * ux + ey * uy
    perp = abs(ex * (-uy) + ey * ux)
    return along, perp


# ============================================================
# 第3部分：pose推論
# ============================================================

def predict_one_image(model, img_path):
    """回傳dict(A/B在resize後空間的座標與信心度)，完全沒偵測到回傳None。
    一張圖若有多個實例(裁切圖理論上只有一顆牙)，取box信心度最高者。"""
    used_conf = CONF_THRESHOLD
    pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                         save=False, verbose=False)[0]

    if pred.boxes is None or len(pred.boxes) == 0:
        pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=FALLBACK_CONF,
                             save=False, verbose=False)[0]
        used_conf = FALLBACK_CONF
        if pred.boxes is None or len(pred.boxes) == 0:
            return None

    if pred.keypoints is None or pred.keypoints.xy is None or len(pred.keypoints.xy) == 0:
        return None

    box_confs = pred.boxes.conf.cpu().numpy()
    best_i = int(np.argmax(box_confs))
    kxy = pred.keypoints.xy.cpu().numpy()[best_i]
    if kxy.shape[0] < 2:
        return None

    kconf = pred.keypoints.conf.cpu().numpy()[best_i] if pred.keypoints.conf is not None else None

    return {
        "偵測到牙齒數": int(len(box_confs)),
        "box信心度": float(box_confs[best_i]),
        "使用的conf門檻": used_conf,
        "A_x_resize": float(kxy[0][0]),
        "A_y_resize": float(kxy[0][1]),
        "B_x_resize": float(kxy[1][0]),
        "B_y_resize": float(kxy[1][1]),
        "A點信心度": float(kconf[0]) if kconf is not None else None,
        "B點信心度": float(kconf[1]) if kconf is not None else None,
    }


# ============================================================
# 第4部分：讀取ground truth
# ============================================================

def get_original_image_size(stem):
    """只需要寬高，用PIL開檔比cv2.imread快很多(不用把整張影像解碼進記憶體)。"""
    if not GT_IMAGE_DIR:
        return None
    for ext in IMAGE_EXTENSIONS:
        candidate = Path(GT_IMAGE_DIR) / f"{stem}{ext}"
        if candidate.exists():
            with Image.open(candidate) as im:
                return im.size  # (W, H)
    return None


def load_gt_points(img_name):
    """讀舊pipeline格式標註(class cx cy w h kx1 ky1 v1 kx2 ky2 v2)。
    座標是相對原始X光片的正規化值，乘上原圖寬高換回像素。
    回傳 (gt_A, gt_B) 或 None。"""
    if not GT_LABEL_DIR:
        return None

    stem = Path(img_name).stem
    label_path = Path(GT_LABEL_DIR) / f"{stem}.txt"
    if not label_path.exists():
        return None

    with open(label_path, "r") as f:
        line = f.readline().strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 11:
        return None
    parts = list(map(float, parts))

    size = get_original_image_size(stem)
    if size is None:
        return None
    W, H = size

    ax, ay, av = parts[5], parts[6], parts[7]
    bx, by, bv = parts[8], parts[9], parts[10]
    if av == 0 or bv == 0:
        return None   # 有點沒被標到，不當GT

    return (ax * W, ay * H), (bx * W, by * H)


# ============================================================
# 第5部分：誤差計算
# ============================================================

def compute_keypoint_errors(pred_A, pred_B, gt_A, gt_B, scale):
    """算單顆牙的所有誤差指標，欄名已是中文，可直接塞進DataFrame。"""
    gt_len = euclidean(gt_A, gt_B)
    pred_len = euclidean(pred_A, pred_B)

    err_A = euclidean(pred_A, gt_A)
    err_B = euclidean(pred_B, gt_B)

    # A/B順序自檢：對調後有沒有明顯變好
    err_swapped = euclidean(pred_A, gt_B) + euclidean(pred_B, gt_A)
    suspect_swap = err_swapped < (err_A + err_B) * SWAP_MARGIN

    out = {
        "GT_A_x": round(gt_A[0], 3), "GT_A_y": round(gt_A[1], 3),
        "GT_B_x": round(gt_B[0], 3), "GT_B_y": round(gt_B[1], 3),
        "真實AB像素長度_原圖px": round(gt_len, 4),

        "A點誤差_原圖px": round(err_A, 3),
        "B點誤差_原圖px": round(err_B, 3),

        "長度誤差px": round(pred_len - gt_len, 4),
        "長度相對誤差%": round(abs(pred_len - gt_len) / max(gt_len, 1e-9) * 100, 3),
        "A_B順序疑似顛倒": "⚠️是" if suspect_swap else "否",
    }

    if DETAILED_ERROR_METRICS:
        along_A, perp_A = decompose_error(pred_A, gt_A, gt_A, gt_B)
        along_B, perp_B = decompose_error(pred_B, gt_B, gt_A, gt_B)
        out.update({
            "A點誤差_resize後px": round(err_A * scale, 3),
            "B點誤差_resize後px": round(err_B * scale, 3),
            "A點誤差dx": round(pred_A[0] - gt_A[0], 3),
            "A點誤差dy": round(pred_A[1] - gt_A[1], 3),
            "B點誤差dx": round(pred_B[0] - gt_B[0], 3),
            "B點誤差dy": round(pred_B[1] - gt_B[1], 3),
            "A點誤差_佔GT長度%": round(err_A / max(gt_len, 1e-9) * 100, 3),
            "B點誤差_佔GT長度%": round(err_B / max(gt_len, 1e-9) * 100, 3),
            "A點沿軸誤差px": round(along_A, 3),
            "A點垂直誤差px": round(perp_A, 3),
            "B點沿軸誤差px": round(along_B, 3),
            "B點垂直誤差px": round(perp_B, 3),
        })

    return out


def build_error_summary(df):
    """把逐顆牙的誤差彙總成「指標 / 數值 / 說明」的長表，比一列超寬的表好讀。"""
    cols = ["指標", "數值", "說明"]
    if df.empty or "A點誤差_原圖px" not in df.columns:
        return pd.DataFrame(columns=cols)

    sub = df[df["A點誤差_原圖px"].notna()]
    n = len(sub)
    if n == 0:
        return pd.DataFrame(columns=cols)

    err_A = sub["A點誤差_原圖px"]
    err_B = sub["B點誤差_原圖px"]
    err_all = pd.concat([err_A, err_B])

    rows = []

    def add(name, value, note=""):
        rows.append({"指標": name, "數值": value, "說明": note})

    len_err = sub["長度誤差px"]

    add("有GT可比對的牙齒數", n, "沒有標註、或標註可見度=0的不計入")

    # --- 點位誤差：判斷「長度準是不是矇到的」 ---
    add("A點中位數誤差(px)", round(err_A.median(), 3), "切端；中位數不被離群點拉走")
    add("B點中位數誤差(px)", round(err_B.median(), 3), "根尖；通常比A難定位，分開看才知道問題在哪端")
    add("兩點最大誤差(px)", round(err_all.max(), 3), "最壞情況，去視覺化資料夾找是哪一張")

    # --- 長度誤差：下游ANN真正吃到的東西 ---
    add("長度誤差 MAE(px)", round(len_err.abs().mean(), 4), "整體準度")
    add("長度誤差 bias(px)", round(len_err.mean(), 4),
        "系統性偏移(正=一律量太長)，跟MAE分開看：bias可事後校正，MAE的隨機成分不行")
    add("長度平均相對誤差(%)", round(sub["長度相對誤差%"].mean(), 3), "跨圖可比")
    add("長度最大絕對誤差(px)", round(len_err.abs().max(), 4), "決定要不要人工挑掉哪幾顆")

    # --- 正確性防呆，不是效能指標 ---
    n_swap = int((sub["A_B順序疑似顛倒"] == "⚠️是").sum())
    add("A/B順序疑似顛倒的牙齒數", n_swap,
        "若佔多數，代表2b訓練標註跟舊33張的關鍵點順序不一致，先修這個再看其他數字")

    if DETAILED_ERROR_METRICS:
        add("--- 以下為除錯用細項 ---", "", "")
        add("A點平均誤差(px)", round(err_A.mean(), 3), "")
        add("B點平均誤差(px)", round(err_B.mean(), 3), "")
        add("A點誤差標準差(px)", round(err_A.std(ddof=1), 3) if n > 1 else None, "")
        add("B點誤差標準差(px)", round(err_B.std(ddof=1), 3) if n > 1 else None, "")
        add("兩點合併平均誤差(px)", round(err_all.mean(), 3), "A、B所有點一起算，等同MPJPE")
        add("A點平均誤差_resize後(px)", round(sub["A點誤差_resize後px"].mean(), 3),
            "模型實際在看的640空間，跟一般pose benchmark可比")
        add("B點平均誤差_resize後(px)", round(sub["B點誤差_resize後px"].mean(), 3), "")
        add("A點平均誤差_佔GT長度(%)", round(sub["A點誤差_佔GT長度%"].mean(), 3), "")
        add("B點平均誤差_佔GT長度(%)", round(sub["B點誤差_佔GT長度%"].mean(), 3), "")
        add("長度誤差 RMSE(px)", round(float(np.sqrt((len_err ** 2).mean())), 4),
            "比MAE更放大離群點")

        # 命中率(PCK)：平均值容易被離群點拉走，比例更能反映「大部分有多好」
        for t in PCK_ABS_THRESHOLDS_PX:
            hit = ((err_A < t).sum() + (err_B < t).sum()) / (2 * n) * 100
            add(f"命中率 誤差<{t}px (%)", round(hit, 2), "A、B兩點合計")
        for t in PCK_REL_THRESHOLDS_PCT:
            hit = ((sub["A點誤差_佔GT長度%"] < t).sum()
                   + (sub["B點誤差_佔GT長度%"] < t).sum()) / (2 * n) * 100
            add(f"命中率 誤差<GT長度的{t}% (%)", round(hit, 2), "A、B兩點合計")

        along = pd.concat([sub["A點沿軸誤差px"].abs(), sub["B點沿軸誤差px"].abs()])
        perp = pd.concat([sub["A點垂直誤差px"], sub["B點垂直誤差px"]])
        add("平均沿軸誤差絕對值(px)", round(along.mean(), 3), "這部分會直接變成長度誤差")
        add("平均垂直誤差絕對值(px)", round(perp.mean(), 3), "這部分幾乎不影響長度")

    return pd.DataFrame(rows, columns=cols)


# ============================================================
# 第6部分：視覺化
# ============================================================

def draw_visualization(img_path, rec, gt_resized=None):
    """預測點(實心)與GT點(空心)畫在同一張resize後的圖上，白線連起來就是誤差。"""
    out_dir = Path(VIS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        return

    pa = (int(round(rec["A_x_resize後"])), int(round(rec["A_y_resize後"])))
    pb = (int(round(rec["B_x_resize後"])), int(round(rec["B_y_resize後"])))

    cv2.line(image, pa, pb, (0, 255, 255), 2)
    cv2.circle(image, pa, 5, (0, 255, 0), -1)     # 預測A(切端) 綠實心
    cv2.circle(image, pb, 5, (0, 0, 255), -1)     # 預測B(根尖) 紅實心
    cv2.putText(image, "A", (pa[0] + 8, pa[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(image, "B", (pb[0] + 8, pb[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if gt_resized is not None:
        ga, gb = gt_resized
        ga = (int(round(ga[0])), int(round(ga[1])))
        gb = (int(round(gb[0])), int(round(gb[1])))
        cv2.line(image, ga, gb, (200, 200, 200), 1)
        cv2.circle(image, ga, 8, (0, 255, 0), 2)      # GT空心圈
        cv2.circle(image, gb, 8, (0, 0, 255), 2)
        cv2.line(image, pa, ga, (255, 255, 255), 1)   # 預測->GT的誤差連線
        cv2.line(image, pb, gb, (255, 255, 255), 1)

    lines = []
    if rec.get("AB像素長度_原圖px") is not None:
        lines.append(f"pred {rec['AB像素長度_原圖px']:.1f}px")
    if rec.get("真實AB像素長度_原圖px") is not None:
        lines.append(f"gt   {rec['真實AB像素長度_原圖px']:.1f}px")
    if rec.get("A點誤差_原圖px") is not None:
        lines.append(f"errA {rec['A點誤差_原圖px']:.1f} errB {rec['B點誤差_原圖px']:.1f}")
    for i, text in enumerate(lines):
        cv2.putText(image, text, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)

    cv2.imwrite(str(out_dir / Path(img_path).name), image)


# ============================================================
# 主流程
# ============================================================

def main():
    weights = Path(POSE_WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"❌ 找不到pose權重 {weights}，請先跑完 2b_train_yolo_pose.py")

    img_dir = Path(IMAGE_DIR)
    if not img_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {img_dir}，請確認 IMAGE_DIR 設定正確")

    img_paths = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not img_paths:
        raise FileNotFoundError(f"❌ {img_dir} 底下沒有任何圖片")

    geo_map = load_geometry_manifest()
    model = YOLO(str(weights))

    print(f"\n=== 對 {len(img_paths)} 張圖跑A/B關鍵點推論 + 長度計算 + GT誤差比對 ===")

    records, failures, raw_records = [], [], []
    n_no_geo = n_no_gt = n_inconsistent = 0

    for img_path in img_paths:
        fname = img_path.name
        pred = predict_one_image(model, img_path)

        if pred is None:
            failures.append({
                "圖片檔名": fname,
                "狀態": "❌ 完全沒偵測到牙齒/關鍵點",
                "已嘗試的最低conf門檻": FALLBACK_CONF,
            })
            print(f"   ❌ {fname}：沒偵測到任何牙齒(已降到conf={FALLBACK_CONF}仍失敗)")
            continue

        raw_records.append({"圖片檔名": fname, **pred})

        A_r = (pred["A_x_resize"], pred["A_y_resize"])
        B_r = (pred["B_x_resize"], pred["B_y_resize"])
        len_resized = euclidean(A_r, B_r)

        rec = {
            "圖片檔名": fname,
            "偵測到牙齒數": pred["偵測到牙齒數"],
            "box信心度": round(pred["box信心度"], 4),
            "A點信心度": round(pred["A點信心度"], 4) if pred["A點信心度"] is not None else None,
            "B點信心度": round(pred["B點信心度"], 4) if pred["B點信心度"] is not None else None,
            "A_x_resize後": round(A_r[0], 3),
            "A_y_resize後": round(A_r[1], 3),
            "B_x_resize後": round(B_r[0], 3),
            "B_y_resize後": round(B_r[1], 3),
            "AB像素長度_resize後px": round(len_resized, 4),
        }

        geo = geo_map.get(fname)
        gt_resized_for_vis = None

        if geo is None:
            n_no_geo += 1
            rec.update({
                "座標空間": "圖片自身像素空間(無幾何資訊)",
                "縮放倍率scale": None,
                "AB像素長度_原圖px": round(len_resized, 4),
                "狀態": "⚠️ manifest查不到scale/pad，長度未換算、也無法跟GT比對",
            })
        else:
            A_o = cropped_to_original(*resized_to_cropped(*A_r, geo), geo)
            B_o = cropped_to_original(*resized_to_cropped(*B_r, geo), geo)

            len_orig = euclidean(A_o, B_o)
            len_shortcut = len_resized / geo["scale"]
            if abs(len_orig - len_shortcut) / max(len_orig, 1e-9) > CONSISTENCY_TOL:
                n_inconsistent += 1
                consistency = f"⚠️ 兩種算法不一致({len_orig:.4f} vs {len_shortcut:.4f})"
            else:
                consistency = "✅ 一致"

            rec.update({
                "座標空間": "原始X光片像素空間",
                "縮放倍率scale": geo["scale"],
                "A_x_原圖": round(A_o[0], 3),
                "A_y_原圖": round(A_o[1], 3),
                "B_x_原圖": round(B_o[0], 3),
                "B_y_原圖": round(B_o[1], 3),
                "AB像素長度_原圖px": round(len_orig, 4),
                "AB像素長度_捷徑驗算px": round(len_shortcut, 4),
                "換算自檢": consistency,
                "狀態": "✅ 預測成功",
            })

            gt = load_gt_points(fname)
            if gt is None:
                n_no_gt += 1
                rec["GT比對"] = "⚠️ 找不到舊標註或標註不完整，跳過誤差計算"
            else:
                gt_A, gt_B = gt
                rec.update(compute_keypoint_errors(A_o, B_o, gt_A, gt_B, geo["scale"]))
                rec["GT比對"] = "✅ 已比對"
                gt_resized_for_vis = (original_to_resized(*gt_A, geo),
                                      original_to_resized(*gt_B, geo))

        low_conf = [c for c in (pred["A點信心度"], pred["B點信心度"])
                    if c is not None and c < KPT_CONF_WARN]
        rec["是否建議人工複查"] = "是(關鍵點信心度偏低)" if low_conf else "否"

        records.append(rec)

        if SAVE_VISUALIZATION:
            draw_visualization(img_path, rec, gt_resized_for_vis)

    # ---------------- 輸出 ----------------
    df = pd.DataFrame(records)

    # 偵測失敗清單即使是空的也一定寫表頭，避免下游腳本讀不到sheet直接爆掉
    fail_cols = ["圖片檔名", "狀態", "已嘗試的最低conf門檻"]
    df_fail = pd.DataFrame(failures, columns=fail_cols) if failures else pd.DataFrame(columns=fail_cols)

    df_err = build_error_summary(df)

    run_info = pd.DataFrame([{
        "輸入資料夾": IMAGE_DIR,
        "使用權重": POSE_WEIGHTS,
        "總圖片數": len(img_paths),
        "預測成功數": len(records),
        "偵測失敗數": len(failures),
        "無幾何資訊(未換算)數": n_no_geo,
        "找不到GT數": n_no_gt,
        "換算自檢不一致數": n_inconsistent,
    }])

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="預測結果", index=False)
        df_err.to_excel(writer, sheet_name="誤差分析", index=False)
        df_fail.to_excel(writer, sheet_name="偵測失敗清單", index=False)
        run_info.to_excel(writer, sheet_name="執行摘要", index=False)

    pd.DataFrame(raw_records).to_csv(RAW_KEYPOINT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- 終端機報告 ----------------
    print(f"\n=======================================================")
    print(f"✨ 推論 + 長度計算 + 誤差比對完成！")
    for k, v in run_info.iloc[0].items():
        print(f"   {k}: {v}")

    if not df_err.empty:
        print(f"\n--- 關鍵點誤差 ---")
        for _, r in df_err.iterrows():
            print(f"   {r['指標']}: {r['數值']}")

    print(f"\n📍 主要輸出: {OUTPUT_XLSX}（預測結果 / 誤差分析 / 偵測失敗清單 / 執行摘要）")
    print(f"📍 原始關鍵點座標(可重算用): {RAW_KEYPOINT_CSV}")
    if SAVE_VISUALIZATION:
        print(f"📍 視覺化: {VIS_DIR}/ （實心=預測，空心=GT，白線=誤差）")

    if n_no_geo:
        print(f"⚠️ {n_no_geo} 張查不到scale/pad，那幾列長度未換算，不能跟其他張比較")
    if n_no_gt:
        print(f"⚠️ {n_no_gt} 張找不到可用的舊標註，沒有納入誤差統計")
    if n_inconsistent:
        print(f"⚠️ {n_inconsistent} 張兩種長度算法對不起來，通常是manifest的scale跟圖片對不上，請查")

    if not df.empty and "A_B順序疑似顛倒" in df.columns:
        n_swap = int((df["A_B順序疑似顛倒"] == "⚠️是").sum())
        n_cmp = int(df["A_B順序疑似顛倒"].notna().sum())
        if n_cmp and n_swap > n_cmp / 2:
            print(f"\n🚨 {n_swap}/{n_cmp} 顆牙「把A/B對調後誤差明顯更小」，這幾乎確定是")
            print(f"   2b的訓練標註跟舊33張的關鍵點順序相反(一邊切端在前、一邊根尖在前)。")
            print(f"   先把順序統一再重訓，不然上面的點位誤差數字沒有參考價值。")
            print(f"   (長度不受影響——AB對調距離一樣——但點位誤差會整個爆掉。)")

    print(f"=======================================================")
    print(f"\n👉 接下來：拿『AB像素長度_原圖px』對上 根管充填長度_20260803.xlsx 的")
    print(f"   『填充物長度(mm)』，就是MATLAB端ANN(像素->毫米)的輸入。")
    print(f"   合併前先看「誤差分析」跟「偵測失敗清單」，把明顯標歪的挑掉，")
    print(f"   否則壞點會直接污染ANN。")


if __name__ == "__main__":
    main()
