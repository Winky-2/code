"""
2c_test_pose_and_measure.py
============================
對應新流程圖階段：（單階段版推論）用2b_train_yolo_pose_fullimage.py
訓練好的模型，直接對「完整X光片」test圖跑推論
-> 用IoU比對舊標註bbox，從這張圖偵測到的所有牙齒框裡挑出目標牙
-> 取得A(切端)/B(根尖)關鍵點座標(已經是原圖像素空間，不用再換算)
-> 計算AB像素長度（餵給MATLAB ANN用）
-> 跟醫師標的舊標註(ground truth)比對，算關鍵點定位誤差

*** 單一class版本：認牙邏輯是這支的核心，不是細節 ***
2b訓練用的Roboflow標註只有一個class(tooth)，目標牙跟一般牙都是同一個
class，差別只在keypoint visibility(目標牙v=2、一般牙v=0)。這代表：
    - model.predict()回傳的每一個偵測框，class都一樣是tooth，
      沒有辦法像「兩個class版」那樣直接篩class==target_tooth。
    - 每個偵測框都會吐出一組keypoint預測，包括一般牙——但一般牙的
      keypoint訓練時v=0沒被監督過，數值基本上是雜訊，不能直接拿
      信心度最高的框當目標牙(信心度高不代表keypoint準，那是box
      信心度，不是keypoint信心度，兩者未必一致)。
    - 所以這支腳本改用「IoU比對舊標註bbox」來選框：這張test圖裡
      偵測到的所有框，跟舊pipeline留下來的目標牙GT bbox比對IoU，
      IoU最高的那一個才當作目標牙，取它的keypoint來用。
      這跟兩階段版1b_train_tooth_crop_yolov11.py選目標牙用的邏輯
      完全一樣，只是這裡把它從「事前裁切用」搬到「事後推論選框用」。

*** 這個做法有一個必須先知道的限制，務必讓學長姐/醫師知道 ***
IoU比對依賴「舊標註裡本來就有這顆目標牙的bbox」，這33張test圖之所以
能這樣做，是因為它們本來就是舊pipeline標過的資料，有ground truth
可以比對。未來如果要接完全沒有舊標註的全新病例，這條路完全走不通，
屆時需要另一種認牙方式(例如醫師在畫面上點一下星號位置、或另外訓練
一個「這是不是目標牙」的分類器)。這是選擇單一class(而非兩個class)
在部署階段要多付的代價，現在先讓IoU比對撐著跑通整條pipeline、
拿到初步準確度數字，之後上線前這一步必須被取代掉。

*** IoU比對失敗時的處理 ***
如果這張圖完全沒有偵測到任何框、或所有框跟GT bbox的IoU都低於
IOU_MATCH_THRESHOLD，代表沒有框可信、無法安全選出目標牙，這種情況
直接記錄為失敗、跳過，不要硬選一個IoU很低的框冒充目標牙(那樣量出來
的keypoint、長度全部沒有意義，還會污染下游ANN)。

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

from image_enhance import enhance_image  # 跟B1共用同一份CLAHE增強邏輯，train/test參數保證一致

# ---------------- 設定區 ----------------
POSE_WEIGHTS = "yolo11_pose_run-batch4_前處理/weights_ready.pt"   # 👈 2b_..._fullimage.py訓練完成的權重

# 要跑推論的圖片資料夾：完整X光片test圖(不是裁切/letterbox後的圖)，
# 通常就跟GT_IMAGE_DIR是同一份
IMAGE_DIR = "33-all-test.yolov8/test/images"

# 舊標註(ground truth)：這支必須要有，因為要靠它的bbox做IoU比對來認牙，
# 不是只拿來算誤差而已，所以跟兩階段版不同，這裡不支援設成None
GT_LABEL_DIR = "33-all-test.yolov8/test/labels"
GT_IMAGE_DIR = "33-all-test.yolov8/test/images"

OUTPUT_XLSX = "yolo像素預測_11_batch4_前處理.xlsx"
RAW_KEYPOINT_CSV = "yolo關鍵點原始座標_11_batch4_前處理.csv"
VIS_DIR = "pose_預測視覺化_11_batch4_前處理"  # 推論結果視覺化輸出資料夾
SAVE_VISUALIZATION = True

IMG_SIZE = 640               # 👈 要跟2b_..._fullimage.py訓練時的IMG_SIZE一致
CONF_THRESHOLD = 0.15         # 跟2b一致，篩box的信心度門檻
FALLBACK_CONF = 0.05          # 主門檻抓不到任何框時降門檻重試(會在狀態欄標註)
KPT_CONF_WARN = 0.5           # 關鍵點信心度低於此值標「建議人工複查」

IOU_MATCH_THRESHOLD = 0.3     # 選框門檻：跟GT bbox的IoU低於此值，視為沒有可信的目標牙框

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SWAP_MARGIN = 0.8             # 對調後誤差 < 原本誤差*此值，才判定疑似順序顛倒

# 平常關著就好，只有在「數字怪怪的、想知道誤差是哪來的」時才打開
DETAILED_ERROR_METRICS = False
PCK_ABS_THRESHOLDS_PX = [5, 10, 20]  # 僅DETAILED時使用：絕對門檻(原圖px)
PCK_REL_THRESHOLDS_PCT = [2, 5, 10]  # 僅DETAILED時使用：相對門檻(佔GT AB長度的%)


# ============================================================
# 第1部分：幾何小工具
# ============================================================

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def yolo_to_corners(cx, cy, w, h):
    """把YOLO正規化的(cx,cy,w,h)轉成(x1,y1,x2,y2)，仍在0~1正規化空間。"""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def compute_iou(boxA, boxB):
    """boxA, boxB皆為(x1,y1,x2,y2)正規化座標。跟1b_train_tooth_crop_yolov11.py
    裡的compute_iou是同一個函式，這裡是選框依據，不是診斷用而已。"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


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
# 第2部分：讀取ground truth(bbox要拿來做IoU選框，A/B要拿來算誤差)
# ============================================================

def get_original_image_size(stem):
    """只需要寬高，用PIL開檔比cv2.imread快很多(不用把整張影像解碼進記憶體)。"""
    for ext in IMAGE_EXTENSIONS:
        candidate = Path(GT_IMAGE_DIR) / f"{stem}{ext}"
        if candidate.exists():
            with Image.open(candidate) as im:
                return im.size  # (W, H)
    return None


def load_gt_record(img_name):
    """讀舊pipeline格式標註(class cx cy w h kx1 ky1 v1 kx2 ky2 v2)。
    回傳 dict{gt_A, gt_B, gt_bbox_corners(正規化x1y1x2y2)} 或 None。
    bbox一定要能讀到才有辦法做IoU選框；A/B若可見度=0則gt_A/gt_B為None
    (仍可用bbox選框，只是沒辦法算關鍵點誤差)。"""
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

    bbox_corners = yolo_to_corners(*parts[1:5])

    ax, ay, av = parts[5], parts[6], parts[7]
    bx, by, bv = parts[8], parts[9], parts[10]
    if av == 0 or bv == 0:
        return {"gt_A": None, "gt_B": None, "gt_bbox_corners": bbox_corners}

    return {
        "gt_A": (ax * W, ay * H),
        "gt_B": (bx * W, by * H),
        "gt_bbox_corners": bbox_corners,
    }


# ============================================================
# 第3部分：pose推論 + IoU選框(認牙)
# ============================================================

def select_target_instance(pred, gt_bbox_corners):
    """從一次推論結果裡，用跟GT bbox的IoU挑出目標牙instance。
    回傳 (instance index, 最佳IoU值)；沒有任何框、或最佳IoU低於
    IOU_MATCH_THRESHOLD，回傳 (None, 最佳IoU值或0.0)。

    *** 為什麼不是直接取信心度最高的框 ***
    box信心度只代表「這是不是一顆牙」，不代表「這是不是目標牙」——
    單一class設計下沒有class訊號可用，模型也沒有專門學「哪顆是目標
    牙」這件事，唯一能拿來認牙的依據就是跟舊GT bbox的位置吻合度(IoU)。
    """
    if pred.boxes is None or len(pred.boxes) == 0:
        return None, 0.0
    if pred.keypoints is None or pred.keypoints.xy is None or len(pred.keypoints.xy) == 0:
        return None, 0.0

    candidate_corners = pred.boxes.xyxyn.cpu().numpy().tolist()
    ious = [compute_iou(gt_bbox_corners, tuple(c)) for c in candidate_corners]
    best_idx = int(max(range(len(ious)), key=lambda i: ious[i]))
    best_iou = ious[best_idx]

    if best_iou < IOU_MATCH_THRESHOLD:
        return None, best_iou
    return best_idx, best_iou


def predict_one_image(model, img_path, gt_bbox_corners):
    """回傳dict(A/B在原圖像素空間的座標與信心度、IoU選框資訊)，
    完全沒有可信框(IoU比對失敗)回傳None。

    *** CLAHE增強放在這裡、而不是先另存一份增強後的圖片資料夾 ***
    直接讀原圖->enhance_image()->把numpy array丟給model.predict()，
    跟B1訓練時套用的是同一份image_enhance.py、同一組參數，確保
    模型訓練時看到的分布(增強後)跟推論時看到的分布(也是增強後)
    一致；draw_visualization()仍然用原始未增強的圖畫框，方便人工
    複查時看清楚原始X光片，這兩件事互不影響。"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_enhanced = enhance_image(img)

    used_conf = CONF_THRESHOLD
    pred = model.predict(source=img_enhanced, imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                         save=False, verbose=False)[0]
    target_idx, best_iou = select_target_instance(pred, gt_bbox_corners)

    if target_idx is None:
        pred = model.predict(source=img_enhanced, imgsz=IMG_SIZE, conf=FALLBACK_CONF,
                             save=False, verbose=False)[0]
        used_conf = FALLBACK_CONF
        target_idx, best_iou = select_target_instance(pred, gt_bbox_corners)
        if target_idx is None:
            return None

    kxy = pred.keypoints.xy.cpu().numpy()[target_idx]
    if kxy.shape[0] < 2:
        return None

    kconf = pred.keypoints.conf.cpu().numpy()[target_idx] if pred.keypoints.conf is not None else None

    box_conf = float(pred.boxes.conf.cpu().numpy()[target_idx])

    return {
        "偵測到牙齒總數": int(len(pred.boxes)),
        "box信心度": box_conf,
        "使用的conf門檻": used_conf,
        "IoU_vs舊GT框": round(best_iou, 3),
        "A_x_原圖": float(kxy[0][0]),
        "A_y_原圖": float(kxy[0][1]),
        "B_x_原圖": float(kxy[1][0]),
        "B_y_原圖": float(kxy[1][1]),
        "A點信心度": float(kconf[0]) if kconf is not None else None,
        "B點信心度": float(kconf[1]) if kconf is not None else None,
    }


# ============================================================
# 第4部分：誤差計算
# ============================================================

def compute_keypoint_errors(pred_A, pred_B, gt_A, gt_B):
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

    if "IoU_vs舊GT框" in sub.columns:
        add("選框IoU中位數", round(sub["IoU_vs舊GT框"].median(), 3),
            "IoU比對是唯一的認牙依據，這個值偏低代表選框本身就不準，keypoint誤差會連帶失真")

    if DETAILED_ERROR_METRICS:
        add("--- 以下為除錯用細項 ---", "", "")
        add("A點平均誤差(px)", round(err_A.mean(), 3), "")
        add("B點平均誤差(px)", round(err_B.mean(), 3), "")
        add("A點誤差標準差(px)", round(err_A.std(ddof=1), 3) if n > 1 else None, "")
        add("B點誤差標準差(px)", round(err_B.std(ddof=1), 3) if n > 1 else None, "")
        add("兩點合併平均誤差(px)", round(err_all.mean(), 3), "A、B所有點一起算，等同MPJPE")
        add("A點平均誤差_佔GT長度(%)", round(sub["A點誤差_佔GT長度%"].mean(), 3), "")
        add("B點平均誤差_佔GT長度(%)", round(sub["B點誤差_佔GT長度%"].mean(), 3), "")
        add("長度誤差 RMSE(px)", round(float(np.sqrt((len_err ** 2).mean())), 4),
            "比MAE更放大離群點")

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
# 第5部分：視覺化(直接畫在原圖上，不再有resize空間)
# ============================================================

def draw_visualization(img_path, rec, gt_points=None):
    """預測點(實心)與GT點(空心)直接畫在原始X光片上，白線連起來就是誤差。"""
    out_dir = Path(VIS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(img_path))
    if image is None:
        return

    pa = (int(round(rec["A_x_原圖"])), int(round(rec["A_y_原圖"])))
    pb = (int(round(rec["B_x_原圖"])), int(round(rec["B_y_原圖"])))

    cv2.line(image, pa, pb, (0, 255, 255), 2)
    cv2.circle(image, pa, 6, (0, 255, 0), -1)     # 預測A(切端) 綠實心
    cv2.circle(image, pb, 6, (0, 0, 255), -1)     # 預測B(根尖) 紅實心
    cv2.putText(image, "A", (pa[0] + 10, pa[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(image, "B", (pb[0] + 10, pb[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if gt_points is not None:
        ga, gb = gt_points
        ga = (int(round(ga[0])), int(round(ga[1])))
        gb = (int(round(gb[0])), int(round(gb[1])))
        cv2.line(image, ga, gb, (200, 200, 200), 1)
        cv2.circle(image, ga, 10, (0, 255, 0), 2)      # GT空心圈
        cv2.circle(image, gb, 10, (0, 0, 255), 2)
        cv2.line(image, pa, ga, (255, 255, 255), 1)    # 預測->GT的誤差連線
        cv2.line(image, pb, gb, (255, 255, 255), 1)

    lines = []
    if rec.get("AB像素長度_原圖px") is not None:
        lines.append(f"pred {rec['AB像素長度_原圖px']:.1f}px")
    if rec.get("真實AB像素長度_原圖px") is not None:
        lines.append(f"gt   {rec['真實AB像素長度_原圖px']:.1f}px")
    if rec.get("A點誤差_原圖px") is not None:
        lines.append(f"errA {rec['A點誤差_原圖px']:.1f} errB {rec['B點誤差_原圖px']:.1f}")
    if rec.get("IoU_vs舊GT框") is not None:
        lines.append(f"IoU  {rec['IoU_vs舊GT框']:.3f}")
    for i, text in enumerate(lines):
        cv2.putText(image, text, (15, 40 + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 0), 2)

    cv2.imwrite(str(out_dir / Path(img_path).name), image)


# ============================================================
# 主流程
# ============================================================

def main():
    weights = Path(POSE_WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"❌ 找不到pose權重 {weights}，請先跑完 2b_train_yolo_pose_fullimage.py")

    img_dir = Path(IMAGE_DIR)
    if not img_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {img_dir}，請確認 IMAGE_DIR 設定正確")

    if not Path(GT_LABEL_DIR).exists():
        raise FileNotFoundError(
            f"❌ 找不到 {GT_LABEL_DIR}。單一class設計下這支腳本必須靠舊標註bbox做IoU"
            f"選框才能認出目標牙，沒有GT就沒辦法跑，這點跟兩階段版不同(那邊GT只是拿來"
            f"算誤差，可以設成None跳過)。"
        )

    img_paths = sorted(p for p in img_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not img_paths:
        raise FileNotFoundError(f"❌ {img_dir} 底下沒有任何圖片")

    model = YOLO(str(weights))

    print(f"\n=== 對 {len(img_paths)} 張完整X光片跑推論，用IoU比對舊GT bbox選出目標牙 "
          f"+ A/B關鍵點 + 長度計算 + GT誤差比對 ===")

    records, failures, raw_records = [], [], []
    n_no_gt = n_low_iou = 0

    for img_path in img_paths:
        fname = img_path.name
        gt = load_gt_record(fname)

        if gt is None:
            n_no_gt += 1
            failures.append({
                "圖片檔名": fname,
                "狀態": "❌ 找不到舊標註(或標註格式異常)，無法IoU選框，跳過",
                "已嘗試的最低conf門檻": None,
            })
            print(f"   ❌ {fname}：找不到舊標註，無法做IoU選框")
            continue

        pred = predict_one_image(model, img_path, gt["gt_bbox_corners"])

        if pred is None:
            failures.append({
                "圖片檔名": fname,
                "狀態": f"❌ 沒有任何框的IoU達到{IOU_MATCH_THRESHOLD}，無法安全選出目標牙",
                "已嘗試的最低conf門檻": FALLBACK_CONF,
            })
            print(f"   ❌ {fname}：所有偵測框跟GT bbox的IoU都太低(已降到conf={FALLBACK_CONF}仍失敗)")
            continue

        raw_records.append({"圖片檔名": fname, **pred})

        A_o = (pred["A_x_原圖"], pred["A_y_原圖"])
        B_o = (pred["B_x_原圖"], pred["B_y_原圖"])
        len_orig = euclidean(A_o, B_o)

        rec = {
            "圖片檔名": fname,
            "偵測到牙齒總數": pred["偵測到牙齒總數"],
            "IoU_vs舊GT框": pred["IoU_vs舊GT框"],
            "box信心度": round(pred["box信心度"], 4),
            "A點信心度": round(pred["A點信心度"], 4) if pred["A點信心度"] is not None else None,
            "B點信心度": round(pred["B點信心度"], 4) if pred["B點信心度"] is not None else None,
            "A_x_原圖": round(A_o[0], 3),
            "A_y_原圖": round(A_o[1], 3),
            "B_x_原圖": round(B_o[0], 3),
            "B_y_原圖": round(B_o[1], 3),
            "AB像素長度_原圖px": round(len_orig, 4),
            "狀態": "✅ 預測成功",
        }

        if pred["IoU_vs舊GT框"] < 0.5:
            n_low_iou += 1
            rec["狀態"] = (f"⚠️ 選框IoU只有{pred['IoU_vs舊GT框']:.3f}(門檻{IOU_MATCH_THRESHOLD}以上才選)，"
                           f"雖然過了門檻但不算高，建議人工複查是否選對牙")

        gt_points_for_vis = None
        if gt["gt_A"] is None:
            rec["GT比對"] = "⚠️ 舊標註A/B可見度=0，跳過誤差計算(但IoU選框仍可正常進行)"
        else:
            rec.update(compute_keypoint_errors(A_o, B_o, gt["gt_A"], gt["gt_B"]))
            rec["GT比對"] = "✅ 已比對"
            gt_points_for_vis = (gt["gt_A"], gt["gt_B"])

        low_conf = [c for c in (pred["A點信心度"], pred["B點信心度"])
                    if c is not None and c < KPT_CONF_WARN]
        rec["是否建議人工複查"] = "是(關鍵點信心度偏低)" if low_conf else "否"

        records.append(rec)

        if SAVE_VISUALIZATION:
            draw_visualization(img_path, rec, gt_points_for_vis)

    # ---------------- 輸出 ----------------
    df = pd.DataFrame(records)

    fail_cols = ["圖片檔名", "狀態", "已嘗試的最低conf門檻"]
    df_fail = pd.DataFrame(failures, columns=fail_cols) if failures else pd.DataFrame(columns=fail_cols)

    df_err = build_error_summary(df)

    run_info = pd.DataFrame([{
        "輸入資料夾": IMAGE_DIR,
        "使用權重": POSE_WEIGHTS,
        "總圖片數": len(img_paths),
        "預測成功數": len(records),
        "偵測/選框失敗數": len(failures),
        "找不到GT數(含在失敗數內)": n_no_gt,
        "選框IoU偏低(<0.5，仍算成功但建議複查)數": n_low_iou,
    }])

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="預測結果", index=False)
        df_err.to_excel(writer, sheet_name="誤差分析", index=False)
        df_fail.to_excel(writer, sheet_name="偵測失敗清單", index=False)
        run_info.to_excel(writer, sheet_name="執行摘要", index=False)

    pd.DataFrame(raw_records).to_csv(RAW_KEYPOINT_CSV, index=False, encoding="utf-8-sig")

    # ---------------- 終端機報告 ----------------
    print(f"\n=======================================================")
    print(f"✨ 全片單階段推論 + 長度計算 + 誤差比對完成！")
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

    if n_no_gt:
        print(f"⚠️ {n_no_gt} 張找不到舊標註，完全無法處理(單一class設計下GT是選框的必要依據)")
    if n_low_iou:
        print(f"⚠️ {n_low_iou} 張選框IoU偏低(<0.5)，雖然過了{IOU_MATCH_THRESHOLD}門檻，建議人工複查是否選對牙")

    if not df.empty and "A_B順序疑似顛倒" in df.columns:
        n_swap = int((df["A_B順序疑似顛倒"] == "⚠️是").sum())
        n_cmp = int(df["A_B順序疑似顛倒"].notna().sum())
        if n_cmp and n_swap > n_cmp / 2:
            print(f"\n🚨 {n_swap}/{n_cmp} 顆牙「把A/B對調後誤差明顯更小」，這幾乎確定是")
            print(f"   B1的訓練標註跟舊33張的關鍵點順序相反(一邊切端在前、一邊根尖在前)。")
            print(f"   先把順序統一再重訓，不然上面的點位誤差數字沒有參考價值。")
            print(f"   (長度不受影響——AB對調距離一樣——但點位誤差會整個爆掉。)")

    print(f"=======================================================")
    print(f"\n👉 接下來：拿『AB像素長度_原圖px』對上 根管充填長度_20260826.xlsx 的")
    print(f"   『填充物長度(mm)』，就是MATLAB端ANN(像素->毫米)的輸入。")
    print(f"   合併前先看「誤差分析」跟「偵測失敗清單」，尤其是選框IoU偏低那幾張，")
    print(f"   把明顯選錯牙或標歪的挑掉，否則壞點會直接污染ANN。")
    print(f"\n⚠️ 提醒：這33張能跑，是因為它們本來就有舊標註bbox可以做IoU選框。")
    print(f"   未來接完全沒有舊標註的全新病例時，這條認牙路徑會失效，")
    print(f"   屆時需要另一種方式(醫師點選星號位置 / 額外的目標牙分類器)。")


if __name__ == "__main__":
    main()
