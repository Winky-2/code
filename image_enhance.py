"""
image_enhance.py
============================
train(B1)、test(B2)共用的影像清晰化模組。

*** 為什麼要抽成獨立檔案共用 ***
CLAHE屬於「前處理(preprocessing)」而不是「資料增強(augmentation)」：
augmentation(如B1裡的degrees/translate/scale/mosaic)只在訓練時隨機
套用，test/推論時完全不用；但CLAHE這種清晰化如果train用、test不用，
等於讓模型在「清晰化後的分布」學會辨識A/B點特徵，test圖卻維持原始
分布，會變成OOD(分布外)輸入，量出來的誤差沒辦法反映模型真實能力。
所以這裡寫成一份函式，B1、B2各自import、用同一組參數呼叫，物理上
保證兩邊不會不一致，不用口頭提醒兩邊要記得對齊參數。

*** clip_limit / tile_grid_size 這兩個參數務必先做小規模預覽再定案 ***
Roboflow內建的Adaptive Equalization是寫死參數、不能調，套用在含有
letterbox黑邊的X光片上容易把邊緣近乎全黑的雜訊放大成雪花顆粒(這是
CLAHE在低訊號區域的已知副作用)。這裡改用OpenCV自己控制參數，預設值
故意抓保守(clip_limit=1.5)，正式用之前務必先跑一輪比較不同clip_limit
(可以參考本檔案最下面的if __name__ == "__main__"小工具)，肉眼確認
是「變清晰」而不是「變雜訊」，再回來調這個預設值。

*** 排除近黑padding區域再做CLAHE ***
letterbox/裁切留下的黑邊如果整片tile都算進去，CLAHE會把邊界那圈
極小的雜訊起伏硬拉伸到看得見，這裡先偵測非近黑像素的bounding box，
只在這個ROI裡做CLAHE，黑邊部分維持原樣不處理。
"""

import cv2
import numpy as np

# ---------------- 共用參數設定區(train/test都吃這裡的預設值) ----------------
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)
BLACK_BORDER_THRESHOLD = 5   # 灰階值低於此門檻視為letterbox/裁切留下的黑邊，不做CLAHE


def enhance_image(img, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    """
    對灰階或BGR影像做CLAHE，回傳跟輸入同樣channel數的影像(dtype uint8)。

    Args:
        img: cv2.imread()讀出來的numpy array，灰階(H,W)或BGR(H,W,3)皆可。
        clip_limit: CLAHE的clipLimit，越高對比增強越強、雜訊也放大越多。
        tile_grid_size: CLAHE分tile的行列數，tile太小在近黑區域容易產生雜訊顆粒。

    Returns:
        增強後的影像，型別/channel數與輸入一致。輸入為None或整張近乎全黑
        時，直接原樣回傳(不處理，避免除以近乎全黑造成的偽影)。
    """
    if img is None:
        return img

    is_color = (img.ndim == 3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if is_color else img.copy()

    # 找出非近黑(有實際訊號)的區域，只在這塊ROI裡做CLAHE，
    # 避免letterbox黑邊被局部equalize放大成雜訊
    mask = gray > BLACK_BORDER_THRESHOLD
    ys, xs = np.where(mask)
    if len(ys) == 0:
        # 整張圖幾乎全黑，沒有訊號可增強，原樣回傳
        return img

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    roi_enhanced = clahe.apply(gray[y0:y1, x0:x1])

    out_gray = gray.copy()
    out_gray[y0:y1, x0:x1] = roi_enhanced

    if is_color:
        return cv2.cvtColor(out_gray, cv2.COLOR_GRAY2BGR)
    return out_gray


if __name__ == "__main__":
    # ------------------------------------------------------------
    # 小型預覽工具：正式接進B1/B2 pipeline前，先用這個比較不同
    # clip_limit的效果，肉眼確認是「變清晰」而不是「變雜訊」。
    #
    # 用法：
    #   python image_enhance.py 你的X光片資料夾 輸出資料夾
    #
    # 會對資料夾內前幾張圖分別跑 clip_limit = 0.5 / 1.0 / 1.5 / 2.0，
    # 各自存一份出來，方便並排比較。
    # ------------------------------------------------------------
    import sys
    from pathlib import Path

    if len(sys.argv) != 3:
        print("用法: python image_enhance.py <輸入圖片資料夾> <輸出資料夾>")
        sys.exit(1)

    src_dir = Path(sys.argv[1])
    dst_dir = Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    img_paths = sorted(p for p in src_dir.glob("*.*") if p.suffix.lower() in exts)[:5]
    if not img_paths:
        print(f"⚠️ {src_dir} 底下沒有找到圖片")
        sys.exit(1)

    test_clip_limits = [0.5, 1.0, 1.5, 2.0]
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 讀不到 {img_path}，跳過")
            continue
        cv2.imwrite(str(dst_dir / f"{img_path.stem}_original.png"), img)
        for cl in test_clip_limits:
            enhanced = enhance_image(img, clip_limit=cl)
            out_name = f"{img_path.stem}_clip{cl}.png"
            cv2.imwrite(str(dst_dir / out_name), enhanced)
        print(f"✅ {img_path.name} 已輸出 original + {len(test_clip_limits)} 種clip_limit版本")

    print(f"\n📍 全部輸出到: {dst_dir}，比較過後再回來調整 image_enhance.py 裡的 CLAHE_CLIP_LIMIT 預設值")
