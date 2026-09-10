"""
A1b_unsharp_after_export.py
============================
對「Roboflow已經匯出、已經標註完成」的資料集做unsharp mask銳化。

*** 為什麼是這支、而不是回去改A1 ***
你們的流程是：原圖 -> A1做CLAHE -> 上傳Roboflow標註 -> 匯出時letterbox到640。
如果把unsharp加在A1裡，就得重新上傳、重新標註一輪，成本很高。

但銳化(unsharp mask)跟CLAHE一樣是「逐點/局部的灰階運算」，
完全不改變任何像素的幾何位置——同一顆牙的根尖在第幾個像素，
銳化前後一模一樣。所以：

    labels/*.txt 完全不需要重做，直接複製過來配對即可。

這支腳本只動images/裡的圖片，labels/原樣複製，data.yaml也一併帶過去，
輸出的資料夾結構跟輸入完全一致，可以直接餵給B1訓練。

*** 附帶好處：順序其實比加在A1更正確 ***
降採樣(letterbox到640)本身就是一次低通濾波，會把高頻細節抹掉。
如果先銳化再縮小(A1的位置)，銳化出來的高頻有一半會在縮小時被吃掉。
在縮小「之後」才銳化(這支的位置)，銳化的是最終要餵給模型的那個
解析度，效果保留得比較完整。

*** 使用方式 ***
1. 先跑預覽模式(PREVIEW_MODE = True)，會挑前幾張圖、對每張輸出
   多組(sigma, amount)參數組合，放大到根尖區肉眼比較。
2. 選定參數後填回下面的UNSHARP_SIGMA / UNSHARP_AMOUNT，
   把PREVIEW_MODE改成False，跑正式批次。
3. 33張test那份資料夾必須用「完全一樣的參數」再跑一次
   (只要改SRC_DIR / DST_DIR)，否則train/test前處理不一致，
   B2量出來的誤差不能反映模型真實能力。

安裝需求：
    pip install opencv-python numpy pandas
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ==================== 設定區 ====================

# 輸入：Roboflow匯出的資料夾(裡面有train/valid/test，或扁平的images/+labels/)
SRC_DIR = "132-train-second-model_enhanced960"

# 輸出：銳化後的新資料夾(結構會跟SRC_DIR一模一樣)
DST_DIR = "132-train-second-model_unsharp"

# ---------------- 銳化參數 ----------------
# sigma: 高斯模糊的標準差，決定「銳化影響的尺度」。
#        小(1.0~1.5) -> 只加強很細的紋理，對粗的邊界幫助小
#        大(3.0~4.0) -> 加強較粗的輪廓，但容易出現明顯的halo(黑白鑲邊)
#        根尖那種過渡帶寬度大概幾個像素，建議從2.0開始掃。
UNSHARP_SIGMA = 2.0

# amount: 銳化強度。out = original*(1+amount) - blurred*amount
#         0.3~0.6之間通常安全，超過1.0幾乎一定會有halo。
#         halo很危險：模型會學到那條假的白邊/黑邊當成「牙齒邊界」，
#         標註是人標在真實邊界上的，兩者對不齊反而讓誤差變大。
UNSHARP_AMOUNT = 0.4

# 灰階值低於此門檻視為letterbox黑邊或cone cut空白區，
# 銳化後還原成原樣，避免在黑白交界處產生一圈亮環。
BORDER_THRESHOLD = 10

# ---------------- 預覽模式 ----------------
PREVIEW_MODE = True          # 👈 第一次跑務必用True，確認參數再改False
PREVIEW_N_IMAGES = 4          # 預覽取前幾張圖
PREVIEW_DIR = "unsharp_參數預覽"
PREVIEW_SIGMAS = [1.5, 2.0, 3.0]
PREVIEW_AMOUNTS = [0.3, 0.5, 0.8]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "valid", "test")

MANIFEST_CSV = "unsharp處理清單.csv"


# ==================== 核心處理 ====================

def unsharp_mask(img, sigma=UNSHARP_SIGMA, amount=UNSHARP_AMOUNT,
                 border_threshold=BORDER_THRESHOLD):
    """
    對影像做unsharp mask銳化，回傳跟輸入同channel數的uint8影像。

    原理：原圖減掉模糊版本 = 高頻成分(邊界、細節)，
          把這個高頻成分加回原圖，邊界對比就被放大。
          out = original + amount * (original - blurred)
              = original*(1+amount) - blurred*amount

    黑邊處理：letterbox黑邊與原圖內容的交界是一條非常陡的階躍，
    銳化會在這條線的亮側製造一圈過亮的環(overshoot)。這裡用亮度
    門檻找出真正有訊號的區域，銳化後把黑區還原，避免這個假邊界。
    """
    if img is None:
        return None

    is_color = (img.ndim == 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img.copy()

    mask = gray > border_threshold

    # sigmaX指定後ksize可以填(0,0)，OpenCV會自動推算適當的核大小
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)

    # addWeighted內部會自動做飽和運算(clip到0~255)，不會overflow
    out = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)

    # 黑區還原成原樣
    out[~mask] = gray[~mask]

    if is_color:
        return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out


def measure_sharpness(img):
    """用Laplacian變異數當銳利度的粗略指標，方便量化比較不同參數。

    ⚠️ 這個數字只能輔助，不能當判準：它對雜訊同樣敏感，
    把雜訊放大也會讓數值上升。最終還是要肉眼看根尖區有沒有halo，
    以及B2跑出來的「B點中位數誤差」有沒有下降。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = gray > BORDER_THRESHOLD
    if mask.sum() == 0:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap[mask].var())


# ==================== 資料夾走訪 ====================

def find_split_dirs(src_root: Path):
    """回傳[(split名稱, images資料夾, labels資料夾), ...]。
    同時相容Roboflow的train/valid/test結構與扁平的images/+labels/結構。"""
    found = []
    for split in SPLITS:
        img_dir = src_root / split / "images"
        lbl_dir = src_root / split / "labels"
        if img_dir.exists():
            found.append((split, img_dir, lbl_dir))

    if not found:
        img_dir = src_root / "images"
        lbl_dir = src_root / "labels"
        if img_dir.exists():
            found.append(("(flat)", img_dir, lbl_dir))

    return found


def collect_images(img_dir: Path):
    return sorted(p for p in img_dir.glob("*.*")
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


# ==================== 預覽模式 ====================

def run_preview(src_root: Path):
    """挑前幾張圖，對每張輸出original + 各種(sigma, amount)組合，
    檔名帶參數，方便並排比較。"""
    split_dirs = find_split_dirs(src_root)
    if not split_dirs:
        raise FileNotFoundError(f"❌ 在 {src_root} 底下找不到images/資料夾")

    _, first_img_dir, _ = split_dirs[0]
    img_paths = collect_images(first_img_dir)[:PREVIEW_N_IMAGES]
    if not img_paths:
        raise FileNotFoundError(f"❌ {first_img_dir} 底下沒有圖片")

    out_dir = Path(PREVIEW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 預覽模式：{len(img_paths)} 張圖 × "
          f"{len(PREVIEW_SIGMAS)}種sigma × {len(PREVIEW_AMOUNTS)}種amount ===\n")

    rows = []
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"   ⚠️ 讀不到 {img_path.name}，跳過")
            continue

        base_sharp = measure_sharpness(img)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_00_original.png"), img)
        rows.append({
            "檔名": img_path.name, "sigma": "-", "amount": "-",
            "銳利度指標": round(base_sharp, 1), "相對原圖": "1.00x",
        })

        for sigma in PREVIEW_SIGMAS:
            for amount in PREVIEW_AMOUNTS:
                sharpened = unsharp_mask(img, sigma=sigma, amount=amount)
                name = f"{img_path.stem}_s{sigma}_a{amount}.png"
                cv2.imwrite(str(out_dir / name), sharpened)

                s = measure_sharpness(sharpened)
                rows.append({
                    "檔名": img_path.name, "sigma": sigma, "amount": amount,
                    "銳利度指標": round(s, 1),
                    "相對原圖": f"{s / max(base_sharp, 1e-9):.2f}x",
                })

        print(f"   ✅ {img_path.name} 已輸出 original + "
              f"{len(PREVIEW_SIGMAS) * len(PREVIEW_AMOUNTS)} 種參數組合")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "銳利度指標比較.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n=======================================================")
    print(f"📍 預覽圖輸出到: {out_dir}")
    print(f"📍 銳利度指標: {csv_path}")
    print(f"=======================================================")
    print(f"\n👉 挑選參數的方法：")
    print(f"   1. 用看圖軟體把根尖(牙根最上端)區域放大到200%以上")
    print(f"   2. 找「牙根邊界有沒有出現一條白邊緊貼一條黑邊」——")
    print(f"      這叫halo，出現就代表amount太大，退回小一號")
    print(f"   3. 在沒有halo的前提下，挑邊界看起來最清楚的那組")
    print(f"   4. 銳利度指標只是輔助，數字最高的那組通常已經過頭了，")
    print(f"      不要直接照數字選")
    print(f"\n   選好後把值填回 UNSHARP_SIGMA / UNSHARP_AMOUNT，")
    print(f"   並將 PREVIEW_MODE 改成 False，再跑一次這支腳本。")


# ==================== 正式批次 ====================

def run_batch(src_root: Path, dst_root: Path):
    split_dirs = find_split_dirs(src_root)
    if not split_dirs:
        raise FileNotFoundError(f"❌ 在 {src_root} 底下找不到images/資料夾")

    if dst_root.exists() and any(dst_root.iterdir()):
        print(f"⚠️ 輸出資料夾 {dst_root} 已存在且非空，內容會被覆蓋。")

    print(f"=== 正式批次：sigma={UNSHARP_SIGMA}, amount={UNSHARP_AMOUNT} ===\n")

    records = []
    total_img = total_lbl = total_fail = 0

    for split, src_img_dir, src_lbl_dir in split_dirs:
        dst_img_dir = dst_root / (split if split != "(flat)" else "") / "images"
        dst_lbl_dir = dst_root / (split if split != "(flat)" else "") / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = collect_images(src_img_dir)
        n_ok = n_fail = 0

        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                n_fail += 1
                records.append({
                    "split": split, "檔名": img_path.name,
                    "狀態": "❌ 讀取失敗", "有對應label": "",
                })
                print(f"   ❌ {img_path.name}：讀取失敗")
                continue

            sharpened = unsharp_mask(img)
            cv2.imwrite(str(dst_img_dir / img_path.name), sharpened)
            n_ok += 1

            # labels原樣複製(幾何沒變，標註完全沿用)
            src_lbl = src_lbl_dir / (img_path.stem + ".txt")
            has_label = src_lbl.exists()
            if has_label:
                shutil.copy2(src_lbl, dst_lbl_dir / src_lbl.name)

            records.append({
                "split": split, "檔名": img_path.name,
                "狀態": "✅ 已銳化",
                "有對應label": "是" if has_label else "⚠️ 否",
            })

        n_lbl = len(list(dst_lbl_dir.glob("*.txt")))
        total_img += n_ok
        total_lbl += n_lbl
        total_fail += n_fail

        print(f"   {split}: 圖片 {n_ok} 張 / label {n_lbl} 個"
              + (f" / 失敗 {n_fail} 張" if n_fail else ""))

        if n_ok != n_lbl:
            print(f"      ⚠️ 圖片數({n_ok})跟label數({n_lbl})對不上，"
                  f"請檢查原資料夾是不是本來就有圖片沒標註")

    # data.yaml等設定檔一併帶過去
    for extra in src_root.glob("*.*"):
        if extra.is_file():
            shutil.copy2(extra, dst_root / extra.name)

    # data.yaml裡的path如果寫絕對路徑，指到的還是舊資料夾，提醒一下
    yaml_path = dst_root / "data.yaml"
    if yaml_path.exists():
        content = yaml_path.read_text(encoding="utf-8", errors="ignore")
        if str(src_root) in content or "path:" in content:
            print(f"\n   ⚠️ 已複製 data.yaml，但裡面的路徑可能還指向舊資料夾，"
                  f"請開啟 {yaml_path} 確認")

    manifest_path = dst_root / MANIFEST_CSV
    pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"\n=======================================================")
    print(f"✨ 銳化完成！")
    print(f"   圖片: {total_img} 張（失敗 {total_fail} 張）")
    print(f"   label: {total_lbl} 個（原樣複製，未修改）")
    print(f"   參數: sigma={UNSHARP_SIGMA}, amount={UNSHARP_AMOUNT}, "
          f"black_threshold={BORDER_THRESHOLD}")
    print(f"📍 輸出資料夾: {dst_root}")
    print(f"📍 處理清單: {manifest_path}")
    print(f"=======================================================")
    print(f"\n⚠️ 接下來務必做的事：")
    print(f"   1. 33張test那份資料夾也要用「完全一樣的參數」跑一次")
    print(f"      (只改SRC_DIR / DST_DIR，銳化參數一個字都不要動)，")
    print(f"      否則train銳化過、test沒有，B2量出來的誤差不能反映真實能力。")
    print(f"   2. B1的POSE_DATA_DIR改指到 {dst_root}，重新訓練。")
    print(f"   3. B2的IMAGE_DIR / GT_IMAGE_DIR改指到銳化後的test資料夾，")
    print(f"      GT_LABEL_DIR用哪一份都可以(labels沒改過，內容一樣)。")
    print(f"   4. 判斷有沒有效看「B點中位數誤差」——那是根尖定位的直接指標。")
    print(f"      長度MAE會被A、B兩端誤差互相抵消，單看容易誤判。")


def main():
    src_root = Path(SRC_DIR)
    if not src_root.exists():
        raise FileNotFoundError(f"❌ 找不到輸入資料夾 {src_root}，請確認SRC_DIR設定正確")

    if PREVIEW_MODE:
        run_preview(src_root)
    else:
        run_batch(src_root, Path(DST_DIR))


if __name__ == "__main__":
    main()
