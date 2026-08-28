"""
01d_resize_cropped_teeth.py
============================
對應新流程圖階段：裁切後的目標牙圖片 -> 統一尺寸（等比例縮放 + 補邊）
銜接在 01c_train_tooth_crop.py 之後執行。

*** 為什麼需要這支 ***
01c裁出來的33張目標牙照片，每張的像素尺寸都不一樣
(牙齒大小不同、偵測框大小也不同)。如果之後要用YOLO-pose去點A/B兩個
關鍵點、再拿像素距離去「比較」不同牙齒的根管長度，
每張圖的「每個pixel代表多少實際長度」必須是公平、可比較的，
不能一張大張的照片跟一張小張的照片，直接拿像素距離互相比。

*** 這支腳本用的做法：等比例縮放 + 補邊(letterbox)，不是直接硬拉伸 ***
如果直接把長方形圖片"硬拉伸"成正方形(例如640x640)，
x方向、y方向的縮放倍率會不一樣(一張很扁的圖被拉伸後，
水平、垂直方向被壓縮/拉長的程度不同)，這樣之後只要A、B兩點連線
不是完全水平或完全垂直，量出來的像素距離就會被扭曲、長度不準。

改用等比例縮放：整張圖用「同一個倍率」縮放，讓長邊符合目標尺寸，
短邊不足的部分補黑邊(padding)湊成正方形。這樣x、y方向的縮放倍率
永遠相同(scale_x == scale_y)，之後不管兩點連線是什麼角度，
只要把量到的像素距離除以這個倍率，就能換算回「裁切後原圖」的
像素距離，比例尺是公平一致的。

*** 這支腳本做的事 ***
1. 讀取01c輸出的裁切圖片資料夾(CROPPED_INPUT_DIR)。
2. 每張圖：等比例縮放到長邊=TARGET_SIZE，短邊補邊(置中或靠左上，
   由PAD_MODE控制)，輸出成TARGET_SIZE x TARGET_SIZE的正方形圖片。
3. 輸出一份Excel，記錄每張圖的：
   - 原始(裁切後)尺寸
   - 縮放倍率scale (單一數值，因為等比例縮放x/y倍率相同)
   - 補邊量(pad_left, pad_top)，因為之後座標換算要先扣掉padding、
     再除以scale，才能換回裁切後原圖的座標
   - 如果01c的Excel(CROP_MANIFEST_XLSX)存在，會自動合併進來，
     方便你在同一份表裡看到「裁切資訊 + resize資訊」全部對照

*** 之後座標換算公式(供YOLO-pose那支之後串接用) ***
給定resize後圖片上量到的像素座標(x_resized, y_resized)：
    x_cropped = (x_resized - pad_left) / scale
    y_cropped = (y_resized - pad_top)  / scale
(x_cropped, y_cropped 就是裁切後原圖上的像素座標，
 如果要再換回整張X光片座標，還要加上01c記錄的裁切像素座標x1,y1)

安裝需求：
    pip install pandas opencv-python openpyxl
"""

from pathlib import Path

import cv2
import pandas as pd

# ---------------- 設定區 ----------------
CROPPED_INPUT_DIR = "cropped_test_teeth"          # 👈 01c輸出的裁切圖片資料夾
RESIZED_OUTPUT_DIR = "resized_test_teeth"         # 統一尺寸後的圖片輸出處
TARGET_SIZE = 640                                 # 統一輸出成 TARGET_SIZE x TARGET_SIZE (正方形)

# 補邊時，短邊不足的部分要往哪裡補：
#   "center" = 兩邊平均補(圖片置中)，比較常見、不偏向任何一側
#   "top_left" = 全部補在右邊/下邊(圖片靠左上角)，換算座標時比較好心算
PAD_MODE = "center"

# 補邊顏色(黑色)，YOLO系列模型的letterbox慣例也是用黑邊，維持一致
PAD_COLOR = (0, 0, 0)

# 如果01c有輸出裁切manifest，這裡會自動讀進來合併，設None則跳過合併
PREVIOUS_CROP_MANIFEST_XLSX = "33張test裁切結果.xlsx"

RESIZE_MANIFEST_XLSX = "33張resize結果.xlsx"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def letterbox_resize(image, target_size, pad_mode, pad_color):
    """等比例縮放 + 補邊，回傳 (resized_image, scale, pad_left, pad_top)。
    scale為單一數值(x/y共用同一倍率)，pad_left/pad_top為補邊像素量，
    這兩者之後拿來把resize後座標換算回裁切後原圖座標。"""
    h, w = image.shape[:2]

    scale = target_size / max(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_size - new_w
    pad_h = target_size - new_h

    if pad_mode == "center":
        pad_left = pad_w // 2
        pad_top = pad_h // 2
    elif pad_mode == "top_left":
        pad_left = 0
        pad_top = 0
    else:
        raise ValueError(f"未知的PAD_MODE: {pad_mode}，只接受 'center' 或 'top_left'")

    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    canvas = cv2.copyMakeBorder(
        resized,
        top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
        borderType=cv2.BORDER_CONSTANT, value=pad_color,
    )
    return canvas, scale, pad_left, pad_top


def process_all_images():
    input_dir = Path(CROPPED_INPUT_DIR)
    if not input_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {input_dir}，請確認 CROPPED_INPUT_DIR 設定正確，"
                                 f"或先跑過 01c_train_tooth_crop.py")

    out_dir = Path(RESIZED_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in input_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not img_paths:
        raise FileNotFoundError(f"❌ {input_dir} 底下沒有找到任何圖片")

    print(f"=== 對 {len(img_paths)} 張裁切後圖片做等比例縮放+補邊 (目標尺寸 {TARGET_SIZE}x{TARGET_SIZE}) ===")

    records = []
    for img_path in img_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️ {img_path.name} 讀取失敗，跳過")
            records.append({
                "圖片檔名": img_path.name,
                "狀態": "❌ 讀取失敗，跳過",
            })
            continue

        orig_h, orig_w = image.shape[:2]
        canvas, scale, pad_left, pad_top = letterbox_resize(image, TARGET_SIZE, PAD_MODE, PAD_COLOR)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), canvas)

        records.append({
            "圖片檔名": img_path.name,
            "狀態": "✅ resize完成",
            "裁切後原始寬": orig_w,
            "裁切後原始高": orig_h,
            "resize輸出尺寸": f"{TARGET_SIZE}x{TARGET_SIZE}",
            "縮放倍率scale": round(scale, 6),
            "補邊pad_left": pad_left,
            "補邊pad_top": pad_top,
            "resize後檔案路徑": str(out_path),
        })

    df = pd.DataFrame(records)

    # 如果找得到01c的裁切manifest，依「圖片檔名」合併起來，
    # 這樣一份Excel就能同時看到「裁切資訊」+「resize資訊」
    if PREVIOUS_CROP_MANIFEST_XLSX and Path(PREVIOUS_CROP_MANIFEST_XLSX).exists():
        prev_df = pd.read_excel(PREVIOUS_CROP_MANIFEST_XLSX)
        if "圖片檔名" in prev_df.columns:
            df = prev_df.merge(df, on="圖片檔名", how="outer", suffixes=("_裁切", "_resize"))
            print(f"✅ 已合併01c的裁切manifest: {PREVIOUS_CROP_MANIFEST_XLSX}")
        else:
            print(f"⚠️ {PREVIOUS_CROP_MANIFEST_XLSX} 裡沒有'圖片檔名'欄位，略過合併")

    df.to_excel(RESIZE_MANIFEST_XLSX, index=False)

    n_ok = (df["狀態"] == "✅ resize完成").sum() if "狀態" in df.columns else 0
    print(f"\n=======================================================")
    print(f"✨ resize完成！共處理 {len(records)} 張圖")
    print(f"   ✅ 成功: {n_ok} 張")
    print(f"📍 統一尺寸後圖片: {RESIZED_OUTPUT_DIR}/")
    print(f"📍 明細(含裁切+resize換算資訊): {RESIZE_MANIFEST_XLSX}")
    print(f"=======================================================")
    print(f"\n👉 之後YOLO-pose預測出A/B關鍵點座標後，換算回裁切後原圖座標的公式：")
    print(f"   x_cropped = (x_resized - pad_left) / scale")
    print(f"   y_cropped = (y_resized - pad_top)  / scale")
    print(f"   這樣不同張照片量出來的長度才是同一把尺，可以互相比較。")


def main():
    process_all_images()


if __name__ == "__main__":
    main()