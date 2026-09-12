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

============================================================
*** v2 重大修正：letterbox縮放比(scale)換算 ***
============================================================
舊版有一個會讓下游ANN根本學不起來的問題：

各張原始X光片的尺寸不一樣，但Roboflow匯出時全部letterbox到640×640。
letterbox是「等比例縮到最長邊剛好640、短邊補黑」，所以

    scale = 640 / max(原圖寬, 原圖高)

每張圖的scale都不同。假設A片原圖1600px高、B片800px高，兩張都壓到
640，scale就是0.40跟0.80，差兩倍——同樣一顆20mm的真牙，在A片量到
約200px，在B片量到約400px。

而MATLAB端的ANN學的是「像素長度 -> 毫米」這一個單變數函數，
輸入只有一個數字，它無從得知這張圖被縮過多少倍。當同一個mm值
對應到200px也對應到400px時，這個映射在數學上就不是良定義的，
ANN只能學出一條被硬拉平的迴歸線，MAE壓不下來是必然結果。

舊版的欄位名叫「AB像素長度_原圖px」，但實際上裝的是letterbox 640
空間的長度，名字跟內容對不上，這個誤導很可能一直遮住了問題。

v2的處理方式：
  1. 從ORIGINAL_IMAGE_DIR找回這張test圖對應的原始X光片，讀它的
     真實尺寸，算出scale。
  2. 欄位拆成兩組，名稱誠實對應內容：
       「..._letterbox{N}px」= 模型實際工作的空間(640)
       「..._原圖px」        = 除以scale還原回原圖空間
  3. 餵給MATLAB ANN的必須是「AB像素長度_原圖px」這一欄。
  4. 找不到原圖 / scale驗證失敗的，明確標記出來，不做靜默fallback
     ——寧可少幾筆，也不要讓錯誤刻度的資料混進ANN。

*** 關鍵點誤差仍然留在letterbox空間 ***
評估模型定位能力時，640空間才是模型實際看到的空間，誤差也是在
這個空間產生的，所以A/B點誤差維持640空間不換算。只有「長度」
需要換回原圖空間，因為那是要交給ANN當輸入的物理量。

*** 還有一個這支腳本解決不了的問題，要問醫師 ***
除以scale只能修掉「letterbox造成的刻度不一致」。如果這些片子
原本就是不同機器/不同感測器拍的，各自的pixel pitch(每個像素代表
幾mm)本來就不同，那即使在原圖空間，px/mm也還是不一致，ANN依然
學不到穩定映射。請確認：這些片子的尺寸不同，是同一台機器的不同
裁切範圍(→ 除以scale就完全修好了)，還是不同機器拍的(→ ANN需要
額外的尺規資訊，例如已知長度的參考物或機器型號當作額外輸入)。

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
POSE_WEIGHTS = "yolo11_pose_run/weights_ready.pt"   # 👈 2b_..._fullimage.py訓練完成的權重

# 要跑推論的圖片資料夾：完整X光片test圖(不是裁切/letterbox後的圖)，
# 通常就跟GT_IMAGE_DIR是同一份
IMAGE_DIR = "test_enhance/test/images"

# 舊標註(ground truth)：這支必須要有，因為要靠它的bbox做IoU比對來認牙，
# 不是只拿來算誤差而已，所以跟兩階段版不同，這裡不支援設成None
GT_LABEL_DIR = "test_enhance/test/labels"
GT_IMAGE_DIR = "test_enhance/test/images"

OUTPUT_XLSX = "yolo像素預測_11.xlsx"
RAW_KEYPOINT_CSV = "yolo關鍵點原始座標_11.csv"
VIS_DIR = "pose_預測視覺化_11"  # 推論結果視覺化輸出資料夾
SAVE_VISUALIZATION = True

# ---------------- v2新增：letterbox scale換算設定 ----------------
# 原始X光片(尺寸各不相同、尚未letterbox)的資料夾。
# 這支腳本要靠它讀回每張圖的真實尺寸來算scale，沒有它就沒辦法把
# 長度換算回原圖空間，下游ANN會拿到刻度不一致的輸入。
ORIGINAL_IMAGE_DIR = "Data_test"

# Roboflow匯出時letterbox的目標邊長(正方形)。跟B1的IMG_SIZE無關，
# 這是「圖片檔案本身的尺寸」，不是「模型推論時縮放到的尺寸」。
LETTERBOX_SIZE = 640

# 是否驗證scale算得對：偵測letterbox圖裡非黑內容區的實際大小，
# 跟「原圖尺寸×scale」比對。差距超過門檻代表letterbox方式跟這裡
# 假設的不一樣(例如不是fit-within而是stretch)，會標記出來。
VERIFY_LETTERBOX = False
LETTERBOX_VERIFY_TOL_PX = 3      # 容許誤差(四捨五入造成的1~2px屬正常)
LETTERBOX_BLACK_THRESHOLD = 10   # 灰階低於此值視為letterbox黑邊

# 找不到原圖時要不要繼續。預設False：直接記為失敗，不輸出這一筆。
# 設成True的話會保留letterbox空間的長度、但「AB像素長度_原圖px」留空，
# 這種列絕對不可以餵進ANN。
ALLOW_MISSING_ORIGINAL = False

IMG_SIZE = 640              # 👈 要跟2b_..._fullimage.py訓練時的IMG_SIZE一致
CONF_THRESHOLD = 0.15         # 跟2b一致，篩box的信心度門檻
FALLBACK_CONF = 0.05          # 主門檻抓不到任何框時降門檻重試(會在狀態欄標註)
KPT_CONF_WARN = 0.5           # 關鍵點信心度低於此值標「建議人工複查」

IOU_MATCH_THRESHOLD = 0.3     # 選框門檻：跟GT bbox的IoU低於此值，視為沒有可信的目標牙框

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SWAP_MARGIN = 0.8             # 對調後誤差 < 原本誤差*此值，才判定疑似順序顛倒

# 平常關著就好，只有在「數字怪怪的、想知道誤差是哪來的」時才打開
DETAILED_ERROR_METRICS = False
PCK_ABS_THRESHOLDS_PX = [5, 10, 20]  # 僅DETAILED時使用：絕對門檻(letterbox px)
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
# 第1.5部分(v2新增)：找回原圖、算letterbox縮放比scale
# ============================================================

def _candidate_stems(stem: str):
    """Roboflow匯出時會改檔名，常見形式：
        原本      IMG_1234.jpg
        匯出後    IMG_1234_jpg.rf.9f3a1c8e2b....jpg
    這裡把可能的後綴一層層剝掉，產生候選的原圖檔名(不含副檔名)，
    由前到後依序嘗試。回傳list，順序代表優先度。
    """
    cands = [stem]

    # 剝掉 ".rf.<hash>"
    if ".rf." in stem:
        base = stem.split(".rf.")[0]
        cands.append(base)
        stem = base

    # 剝掉 Roboflow把原副檔名寫進檔名的 "_jpg" / "_png" / "_jpeg" 尾巴
    for suffix in ("_jpg", "_jpeg", "_png", "_bmp", "_JPG", "_PNG"):
        if stem.endswith(suffix):
            cands.append(stem[: -len(suffix)])

    # 去重但保持順序
    seen, out = set(), []
    for c in cands:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def build_original_index(original_dir: Path):
    """把原圖資料夾掃一遍，建立 {檔名stem: 路徑} 的索引，
    避免每張圖都重掃一次資料夾。"""
    index = {}
    if not original_dir.exists():
        return index
    for p in sorted(original_dir.rglob("*.*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(p.stem, p)
    return index


def resolve_original_size(stem: str, original_index: dict):
    """依檔名找回原圖尺寸。
    回傳 (W0, H0, 原圖路徑) 或 (None, None, None)。"""
    for cand in _candidate_stems(stem):
        p = original_index.get(cand)
        if p is not None:
            try:
                with Image.open(p) as im:
                    W0, H0 = im.size
                return W0, H0, p
            except Exception:
                return None, None, None
    return None, None, None


def detect_content_bbox(img_path: Path, threshold=LETTERBOX_BLACK_THRESHOLD):
    """偵測letterbox圖裡「非黑內容區」的寬高，用來驗證scale。
    回傳 (content_w, content_h) 或 None。"""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if img.ndim == 3:
        if img.shape[2] == 1:          # OpenCV 5.0：灰階回傳 (H, W, 1)
            img = img[:, :, 0]
        else:                          # 保險：萬一真的是彩色或帶 alpha
            code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
            img = cv2.cvtColor(img, code)
    ys, xs = np.where(img > threshold)
    if len(ys) == 0:
        return None
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def compute_scale(img_path: Path, stem: str, original_index: dict):
    """算這張letterbox圖相對於原圖的縮放比。

    letterbox = 等比例縮到最長邊剛好LETTERBOX_SIZE、短邊補黑，所以
        scale = LETTERBOX_SIZE / max(W0, H0)

    回傳 dict：
        scale             : float 或 None
        原圖寬 / 原圖高    : int 或 None
        原圖檔名          : str 或 None
        scale驗證         : 說明字串(給人看的，異常時要去查)
    """
    W0, H0, orig_path = resolve_original_size(stem, original_index)
    if W0 is None:
        return {
            "scale": None, "原圖寬": None, "原圖高": None, "原圖檔名": None,
            "scale驗證": "❌ 在ORIGINAL_IMAGE_DIR找不到對應原圖，無法換算",
        }

    scale = LETTERBOX_SIZE / max(W0, H0)

    verify_msg = "(未驗證)"
    if VERIFY_LETTERBOX:
        content = detect_content_bbox(img_path)
        if content is None:
            verify_msg = "⚠️ 讀不到letterbox圖或整張近乎全黑，無法驗證"
        else:
            cw, ch = content
            exp_w, exp_h = round(W0 * scale), round(H0 * scale)
            dw, dh = abs(cw - exp_w), abs(ch - exp_h)
            if dw <= LETTERBOX_VERIFY_TOL_PX and dh <= LETTERBOX_VERIFY_TOL_PX:
                verify_msg = f"✅ 內容區{cw}×{ch} 符合預期{exp_w}×{exp_h}"
            else:
                verify_msg = (f"🚨 內容區{cw}×{ch} 與預期{exp_w}×{exp_h}不符"
                              f"(差{dw}×{dh}px)，letterbox方式可能跟假設不同，"
                              f"這筆scale不可信")

    return {
        "scale": round(scale, 6),
        "原圖寬": W0, "原圖高": H0,
        "原圖檔名": orig_path.name if orig_path else None,
        "scale驗證": verify_msg,
    }


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
    完全沒有可信框(IoU比對失敗)回傳None。"""
    used_conf = CONF_THRESHOLD
    pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                         save=False, verbose=False)[0]
    target_idx, best_iou = select_target_instance(pred, gt_bbox_corners)

    if target_idx is None:
        pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=FALLBACK_CONF,
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
        "真實AB像素長度_letterbox px": round(gt_len, 4),

        # 點位誤差維持letterbox空間：模型是在這個空間看圖、產生誤差的，
        # 評估定位能力就該用這個空間，不需要(也不該)換算回原圖。
        "A點誤差_letterbox px": round(err_A, 3),
        "B點誤差_letterbox px": round(err_B, 3),

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
    if df.empty or "A點誤差_letterbox px" not in df.columns:
        return pd.DataFrame(columns=cols)

    sub = df[df["A點誤差_letterbox px"].notna()]
    n = len(sub)
    if n == 0:
        return pd.DataFrame(columns=cols)

    err_A = sub["A點誤差_letterbox px"]
    err_B = sub["B點誤差_letterbox px"]
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
    add("長度誤差 MAE(letterbox px)", round(len_err.abs().mean(), 4),
        "模型工作空間的準度")
    add("長度誤差 bias(letterbox px)", round(len_err.mean(), 4),
        "系統性偏移(正=一律量太長)，跟MAE分開看：bias可事後校正，MAE的隨機成分不行")
    add("長度平均相對誤差(%)", round(sub["長度相對誤差%"].mean(), 3),
        "唯一不受scale影響、跨圖真正可比的長度指標")
    add("長度最大絕對誤差(letterbox px)", round(len_err.abs().max(), 4),
        "決定要不要人工挑掉哪幾顆")

    # v2：原圖空間的長度誤差才跟mm同刻度，是ANN實際會吃到的量
    if "長度誤差_原圖px" in sub.columns:
        len_err_o = sub["長度誤差_原圖px"].dropna()
        if len(len_err_o):
            add("長度誤差 MAE(原圖px)", round(len_err_o.abs().mean(), 4),
                f"✅ ANN實際吃到的刻度，n={len(len_err_o)}；報告請用這個而非letterbox版")
            add("長度誤差 bias(原圖px)", round(len_err_o.mean(), 4),
                "原圖空間的系統性偏移")

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

    pa = (int(round(rec["A_x_letterbox"])), int(round(rec["A_y_letterbox"])))
    pb = (int(round(rec["B_x_letterbox"])), int(round(rec["B_y_letterbox"])))

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
    if rec.get("AB像素長度_letterbox px") is not None:
        lines.append(f"pred {rec['AB像素長度_letterbox px']:.1f}px")
    if rec.get("真實AB像素長度_letterbox px") is not None:
        lines.append(f"gt   {rec['真實AB像素長度_letterbox px']:.1f}px")
    if rec.get("AB像素長度_原圖px") is not None:
        lines.append(f"orig {rec['AB像素長度_原圖px']:.1f}px (ANN輸入)")
    if rec.get("A點誤差_letterbox px") is not None:
        lines.append(f"errA {rec['A點誤差_letterbox px']:.1f} errB {rec['B點誤差_letterbox px']:.1f}")
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

    # v2：先把原圖資料夾掃成索引，之後每張圖用檔名查回原圖尺寸算scale
    original_index = build_original_index(Path(ORIGINAL_IMAGE_DIR))
    if not original_index:
        msg = (f"❌ ORIGINAL_IMAGE_DIR「{ORIGINAL_IMAGE_DIR}」不存在或裡面沒有圖片。\n"
               f"   這支腳本必須靠原圖尺寸算出letterbox縮放比scale，才能把AB長度\n"
               f"   換算回原圖空間；沒有它，各張圖的像素長度刻度不一致，餵給ANN\n"
               f"   等於要它從矛盾的資料學一個不存在的映射。")
        if ALLOW_MISSING_ORIGINAL:
            print(msg + "\n   (ALLOW_MISSING_ORIGINAL=True，仍繼續執行，但原圖空間長度全部留空)")
        else:
            raise FileNotFoundError(msg)
    else:
        print(f"📁 原圖索引建立完成：{len(original_index)} 張（來源 {ORIGINAL_IMAGE_DIR}）")

    print(f"\n=== 對 {len(img_paths)} 張完整X光片跑推論，用IoU比對舊GT bbox選出目標牙 "
          f"+ A/B關鍵點 + 長度計算 + scale換算 + GT誤差比對 ===")

    records, failures, raw_records = [], [], []
    n_no_gt = n_low_iou = 0
    n_no_scale = n_scale_bad = 0
    scale_values = []

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

        # ---- v2：算這張圖的letterbox縮放比 ----
        sc = compute_scale(img_path, Path(fname).stem, original_index)
        scale = sc["scale"]

        if scale is None:
            n_no_scale += 1
            if not ALLOW_MISSING_ORIGINAL:
                failures.append({
                    "圖片檔名": fname,
                    "狀態": "❌ 找不到對應原圖，無法算scale，長度刻度不可比，跳過",
                    "已嘗試的最低conf門檻": None,
                })
                print(f"   ❌ {fname}：在 {ORIGINAL_IMAGE_DIR} 找不到對應原圖，"
                      f"無法換算回原圖空間")
                continue
        elif str(sc["scale驗證"]).startswith("🚨"):
            n_scale_bad += 1

        if scale is not None:
            scale_values.append(scale)

        raw_records.append({"圖片檔名": fname, "縮放比scale": scale, **pred})

        A_o = (pred["A_x_原圖"], pred["A_y_原圖"])
        B_o = (pred["B_x_原圖"], pred["B_y_原圖"])
        len_lb = euclidean(A_o, B_o)                      # letterbox(640)空間
        len_o = (len_lb / scale) if scale else None       # 還原回原圖空間

        rec = {
            "圖片檔名": fname,
            "偵測到牙齒總數": pred["偵測到牙齒總數"],
            "IoU_vs舊GT框": pred["IoU_vs舊GT框"],
            "box信心度": round(pred["box信心度"], 4),
            "A點信心度": round(pred["A點信心度"], 4) if pred["A點信心度"] is not None else None,
            "B點信心度": round(pred["B點信心度"], 4) if pred["B點信心度"] is not None else None,
            # 關鍵點座標在letterbox空間(模型直接輸出的空間)
            "A_x_letterbox": round(A_o[0], 3),
            "A_y_letterbox": round(A_o[1], 3),
            "B_x_letterbox": round(B_o[0], 3),
            "B_y_letterbox": round(B_o[1], 3),
            # 長度兩種空間都留，欄名誠實對應內容
            "AB像素長度_letterbox px": round(len_lb, 4),
            "AB像素長度_原圖px": round(len_o, 4) if len_o is not None else None,
            # scale相關追溯欄位
            "縮放比scale": scale,
            "原圖檔名": sc["原圖檔名"],
            "原圖寬": sc["原圖寬"],
            "原圖高": sc["原圖高"],
            "scale驗證": sc["scale驗證"],
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

            # GT長度與長度誤差也給一份原圖空間版本(這才是跟mm同刻度的量)
            if scale:
                rec["真實AB像素長度_原圖px"] = round(
                    rec["真實AB像素長度_letterbox px"] / scale, 4)
                rec["長度誤差_原圖px"] = round(rec["長度誤差px"] / scale, 4)
            else:
                rec["真實AB像素長度_原圖px"] = None
                rec["長度誤差_原圖px"] = None

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
        "原圖資料夾": ORIGINAL_IMAGE_DIR,
        "letterbox邊長": LETTERBOX_SIZE,
        "找不到原圖(無法算scale)數": n_no_scale,
        "scale驗證不通過數": n_scale_bad,
        "scale最小值": round(min(scale_values), 4) if scale_values else None,
        "scale最大值": round(max(scale_values), 4) if scale_values else None,
        "scale最大/最小倍數": (round(max(scale_values) / min(scale_values), 2)
                              if scale_values and min(scale_values) > 0 else None),
    }])

    # scale明細獨立一張表，方便一眼看出刻度分歧有多嚴重
    scale_cols = ["圖片檔名", "原圖檔名", "原圖寬", "原圖高", "縮放比scale",
                  "AB像素長度_letterbox px", "AB像素長度_原圖px", "scale驗證"]
    df_scale = (df[[c for c in scale_cols if c in df.columns]]
                if not df.empty else pd.DataFrame(columns=scale_cols))

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="預測結果", index=False)
        df_err.to_excel(writer, sheet_name="誤差分析", index=False)
        df_scale.to_excel(writer, sheet_name="scale換算明細", index=False)
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

    print(f"\n📍 主要輸出: {OUTPUT_XLSX}（預測結果 / 誤差分析 / scale換算明細 / 偵測失敗清單 / 執行摘要）")
    print(f"📍 原始關鍵點座標(可重算用): {RAW_KEYPOINT_CSV}")
    if SAVE_VISUALIZATION:
        print(f"📍 視覺化: {VIS_DIR}/ （實心=預測，空心=GT，白線=誤差）")

    if n_no_gt:
        print(f"⚠️ {n_no_gt} 張找不到舊標註，完全無法處理(單一class設計下GT是選框的必要依據)")
    if n_low_iou:
        print(f"⚠️ {n_low_iou} 張選框IoU偏低(<0.5)，雖然過了{IOU_MATCH_THRESHOLD}門檻，建議人工複查是否選對牙")

    # ---------------- v2：scale 診斷 ----------------
    if n_no_scale:
        print(f"⚠️ {n_no_scale} 張在 {ORIGINAL_IMAGE_DIR} 找不到對應原圖。")
        print(f"   Roboflow匯出常會把檔名改成 原名_jpg.rf.<hash>.jpg，這支已經會自動")
        print(f"   剝掉這兩層後綴去比對；若還是找不到，請確認原圖資料夾路徑，或手動")
        print(f"   建一份「letterbox檔名 -> 原圖檔名」對照表。")
    if n_scale_bad:
        print(f"🚨 {n_scale_bad} 張的scale驗證不通過（letterbox圖裡的內容區大小跟")
        print(f"   『原圖尺寸×scale』對不上）。這代表Roboflow的縮放方式跟這裡假設的")
        print(f"   fit-within letterbox不同(例如是拉伸變形、或有額外裁切)，這些列的")
        print(f"   原圖px長度不可信，請先到「scale換算明細」工作表查是哪幾張。")

    if scale_values:
        s_min, s_max = min(scale_values), max(scale_values)
        ratio = s_max / s_min if s_min > 0 else float("inf")
        print(f"\n--- letterbox縮放比(scale)分布 ---")
        print(f"   最小 {s_min:.4f} / 最大 {s_max:.4f} / 最大是最小的 {ratio:.2f} 倍")
        if ratio > 1.05:
            print(f"   🚨 各張圖的縮放比不一致（差 {ratio:.2f} 倍）。這正是舊版把")
            print(f"      letterbox長度直接餵ANN會學不起來的原因：同一個mm值在不同")
            print(f"      片子上對應到差 {ratio:.2f} 倍的像素值，映射不是良定義的。")
            print(f"      ✅ 請務必改用「AB像素長度_原圖px」這一欄餵ANN。")
        else:
            print(f"   ✅ 各張圖縮放比幾乎一致，letterbox本身沒有造成刻度分歧。")
            print(f"      (仍建議用原圖px欄位，語意比較清楚。)")

    if not df.empty and "A_B順序疑似顛倒" in df.columns:
        n_swap = int((df["A_B順序疑似顛倒"] == "⚠️是").sum())
        n_cmp = int(df["A_B順序疑似顛倒"].notna().sum())
        if n_cmp and n_swap > n_cmp / 2:
            print(f"\n🚨 {n_swap}/{n_cmp} 顆牙「把A/B對調後誤差明顯更小」，這幾乎確定是")
            print(f"   B1的訓練標註跟舊33張的關鍵點順序相反(一邊切端在前、一邊根尖在前)。")
            print(f"   先把順序統一再重訓，不然上面的點位誤差數字沒有參考價值。")
            print(f"   (長度不受影響——AB對調距離一樣——但點位誤差會整個爆掉。)")

    print(f"=======================================================")
    print(f"\n👉 接下來：拿『AB像素長度_原圖px』(不是letterbox那一欄！)對上")
    print(f"   根管充填長度_20260826.xlsx 的")
    print(f"   『填充物長度(mm)』，就是MATLAB端ANN(像素->毫米)的輸入。")
    print(f"   合併前先看「誤差分析」跟「偵測失敗清單」，尤其是選框IoU偏低那幾張，")
    print(f"   把明顯選錯牙或標歪的挑掉，否則壞點會直接污染ANN。")
    print(f"\n⚠️ scale換算只修掉「letterbox造成的刻度不一致」。如果這些片子原本是")
    print(f"   不同機器/不同感測器拍的，各自的pixel pitch(一個像素代表幾mm)本來就")
    print(f"   不同，那即使換算回原圖空間，px/mm還是不一致，ANN依然學不到穩定映射。")
    print(f"   請跟醫師確認片子來源，這會決定要不要幫ANN補一個尺規輸入。")
    print(f"\n⚠️ 提醒：這33張能跑，是因為它們本來就有舊標註bbox可以做IoU選框。")
    print(f"   未來接完全沒有舊標註的全新病例時，這條認牙路徑會失效，")
    print(f"   屆時需要另一種方式(醫師點選星號位置 / 額外的目標牙分類器)。")


if __name__ == "__main__":
    main()
