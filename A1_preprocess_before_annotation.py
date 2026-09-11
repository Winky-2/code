"""
A1_preprocess_before_annotation_standalone.py
============================================
在「上傳Roboflow標註」之前，先對整批原始X光片套用CLAHE清晰化，
把處理過的圖片存到新資料夾，之後拿這個資料夾的圖去Roboflow標註
（而不是標原圖）。

*** 這是「獨立版」：不import image_enhance.py ***
CLAHE的參數與enhance_image()函式已經整段內嵌在這支檔案裡，
只要有 opencv-python + pandas 就能單獨執行、單獨搬到別台電腦跑，
不需要旁邊還放著 image_enhance.py。

⚠️ 但獨立的代價是「參數來源不再唯一」，這件事一定要記住：
    B1/B2 那邊如果還是 import image_enhance.py，那CLAHE參數就變成
    兩份（這裡一份、image_enhance.py一份）。哪天你調了其中一邊、
    忘了同步另一邊，train/test的前處理就會悄悄不一致，而且不會報錯，
    只會表現成「誤差莫名其妙變差」，非常難查。
    -> 每次改 CLIP_LIMIT / TILE_GRID_SIZE / BLACK_BORDER_THRESHOLD，
       務必同時檢查 image_enhance.py 裡的對應常數是不是同一個值。

*** 這支跟下面 --preview 模式的差別 ***
--preview 是「探索用」的：只挑前幾張圖、每張輸出好幾種clip_limit
版本，讓你並排比較、肉眼決定哪個clip_limit最好。
正常模式是「量產用」的：clip_limit已經決定好了，對整個資料夾的每
一張圖都套用同一組參數，輸出「一張圖一個結果」，資料夾內容可以直接
拖去Roboflow建立新的標註專案。

*** 讀我：這會改變你們的pipeline，務必想清楚再動手 ***
目前B1/B2的設計是「原圖存放、CLAHE在train/test當下即時套用」
(B1寫成硬碟檔案、B2只在記憶體套用)，這樣設計的理由是保證train/test
用同一份參數、不會不一致。

如果改成「先增強、再標註」，代表：
  1. Roboflow裡的標註座標，是標在「增強後的圖」上，不是原圖。
     這樣做的好處正是你要的——邊界更清楚，學長標根尖(根管長度)
     的座標會更準。
  2. 但這也表示，這批「增強後的圖」以後就是你的新「原始資料」。
     B1訓練時如果再對它套用一次CLAHE（也就是雙重增強），對比度
     會被過度拉伸、可能反而變成雜訊，所以之後接手訓練時，B1讀到
     這批圖時要把它的CLAHE開關關掉（或者：把這批圖直接當成
     這裡 enhance_image() 的輸出來源，B1不要再處理一次）。
  3. B2推論全新病例時，如果新病例給的是「未增強的原圖」，那麼
     推論前仍然要用同一組clip_limit跑過這支腳本的enhance_image()，
     這樣模型看到的分布才會跟訓練時一致。也就是說：這支腳本產生
     的參數(CLIP_LIMIT、TILE_GRID_SIZE)之後在B1/B2整條線上都要
     沿用同一份，不能各自調整。

用法：
    # 量產模式（用設定區的INPUT_DIR / OUTPUT_DIR）
    python A1_preprocess_before_annotation_standalone.py

    # 量產模式（用命令列指定資料夾，會蓋掉設定區）
    python A1_preprocess_before_annotation_standalone.py <輸入資料夾> <輸出資料夾>

    # 預覽模式（比較不同clip_limit，先看再決定參數）
    python A1_preprocess_before_annotation_standalone.py --preview <輸入資料夾> <預覽輸出資料夾>

安裝需求：
    pip install opencv-python pandas
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ============================================================
# 第0部分：CLAHE參數與函式（原本在 image_enhance.py，現在內嵌）
# ============================================================

# ---------------- CLAHE參數設定區 ----------------
# ⚠️ 這三個值必須跟 B1/B2 實際使用的那一份保持一致（見檔頭警告）。
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = (8, 8)
BLACK_BORDER_THRESHOLD = 5   # 灰階值低於此門檻視為letterbox/裁切留下的黑邊，不做CLAHE


def enhance_image(img, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    """
    對灰階或BGR影像做CLAHE，回傳跟輸入同樣channel數的影像(dtype uint8)。

    *** 為什麼要排除近黑padding區域再做CLAHE ***
    letterbox/裁切留下的黑邊如果整片tile都算進去，CLAHE會把邊界那圈
    極小的雜訊起伏硬拉伸到看得見。這裡先偵測非近黑像素的bounding box，
    只在這個ROI裡做CLAHE，黑邊部分維持原樣不處理。
    (Roboflow內建的Adaptive Equalization是寫死參數、不能調，正是因為
     這個原因才改用OpenCV自己控制。)

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

    # 找出非近黑(有實際訊號)的區域，只在這塊ROI裡做CLAHE
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


# ============================================================
# 第1部分：批次前處理設定區
# ============================================================

INPUT_DIR = "Data_RCTKKK"        # 👈 原始、尚未標註的X光片資料夾
OUTPUT_DIR = "rctkkk_A1"   # 👈 增強後、要拿去Roboflow標註的輸出資料夾

CLIP_LIMIT = CLAHE_CLIP_LIMIT
TILE_GRID_SIZE = CLAHE_TILE_GRID_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# 已經處理過的檔名如果在OUTPUT_DIR已存在，預設跳過（方便中斷後續跑）。
# 想強制全部重跑就設成True。
OVERWRITE_EXISTING = False

MANIFEST_CSV = "前處理清單_增強前後對照.csv"  # 留一份可追溯紀錄，report用得到

# 預覽模式要比較的clip_limit清單
PREVIEW_CLIP_LIMITS = [0.5, 1.0, 1.5, 2.0]
PREVIEW_MAX_IMAGES = 5


# ============================================================
# 第2部分：批次前處理（量產模式）
# ============================================================

def collect_images(input_dir: Path):
    return sorted(p for p in input_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def process_all(input_dir_str=INPUT_DIR, output_dir_str=OUTPUT_DIR):
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    if not input_dir.exists():
        raise FileNotFoundError(f"❌ 找不到輸入資料夾 {input_dir}，請確認 INPUT_DIR 設定正確")

    output_dir.mkdir(parents=True, exist_ok=True)

    img_paths = collect_images(input_dir)
    if not img_paths:
        raise FileNotFoundError(f"❌ {input_dir} 底下沒有找到任何圖片({IMAGE_EXTENSIONS})")

    print(f"=== 開始批次前處理：{len(img_paths)} 張圖，clip_limit={CLIP_LIMIT}, "
          f"tile_grid_size={TILE_GRID_SIZE} ===")

    records = []
    n_ok = n_skip = n_fail = 0

    for img_path in img_paths:
        out_path = output_dir / img_path.name

        if out_path.exists() and not OVERWRITE_EXISTING:
            n_skip += 1
            records.append({
                "檔名": img_path.name,
                "狀態": "⏭️ 已存在，跳過(OVERWRITE_EXISTING=False)",
                "clip_limit": None,
            })
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            n_fail += 1
            records.append({
                "檔名": img_path.name,
                "狀態": "❌ 讀取失敗(檔案損毀或格式不支援)",
                "clip_limit": None,
            })
            print(f"   ❌ {img_path.name}：讀取失敗，跳過")
            continue

        enhanced = enhance_image(img, clip_limit=CLIP_LIMIT, tile_grid_size=TILE_GRID_SIZE)
        cv2.imwrite(str(out_path), enhanced)

        n_ok += 1
        records.append({
            "檔名": img_path.name,
            "狀態": "✅ 已增強",
            "clip_limit": CLIP_LIMIT,
        })

    manifest_path = output_dir / MANIFEST_CSV
    pd.DataFrame(records).to_csv(manifest_path, index=False, encoding="utf-8-sig")

    print(f"\n=======================================================")
    print(f"✨ 批次前處理完成！")
    print(f"   成功增強: {n_ok} 張")
    print(f"   跳過(已存在): {n_skip} 張")
    print(f"   失敗: {n_fail} 張")
    print(f"📍 輸出資料夾: {output_dir}（把這個資料夾拖去Roboflow建立新標註專案）")
    print(f"📍 對照清單: {manifest_path}")
    print(f"=======================================================")
    print(f"\n👉 提醒：正式標註前，建議先用本檔案的預覽模式")
    print(f"   (python {Path(__file__).name} --preview <少量圖片資料夾> <預覽輸出資料夾>)")
    print(f"   肉眼確認 clip_limit={CLIP_LIMIT} 是「變清晰」而不是「變雜訊」，")
    print(f"   確定滿意再跑這支批次處理全部資料，避免標註完才發現參數要重調。")
    print(f"\n⚠️ 這批增強後的圖以後就是你的新「原始資料」，之後接上B1訓練時，")
    print(f"   要確認B1不會對它再套用一次CLAHE（雙重增強），且B2推論全新病例的")
    print(f"   原圖時，也要用同一組clip_limit跑過這裡的enhance_image()，")
    print(f"   train/test/未來推論三邊的前處理狀態才會一致。")
    print(f"\n⚠️ 這支是獨立版：CLAHE參數在本檔案自己有一份。改參數時記得同步")
    print(f"   B1/B2那邊實際使用的那一份，否則兩邊會悄悄不一致且不會報錯。")


# ============================================================
# 第3部分：預覽模式（決定clip_limit用）
# ============================================================

def preview(src_dir_str, dst_dir_str):
    """對前幾張圖分別跑多種clip_limit，各存一份，方便並排比較。"""
    src_dir = Path(src_dir_str)
    dst_dir = Path(dst_dir_str)

    if not src_dir.exists():
        raise FileNotFoundError(f"❌ 找不到輸入資料夾 {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    img_paths = collect_images(src_dir)[:PREVIEW_MAX_IMAGES]
    if not img_paths:
        print(f"⚠️ {src_dir} 底下沒有找到圖片")
        return

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ 讀不到 {img_path}，跳過")
            continue
        cv2.imwrite(str(dst_dir / f"{img_path.stem}_original.png"), img)
        for cl in PREVIEW_CLIP_LIMITS:
            enhanced = enhance_image(img, clip_limit=cl, tile_grid_size=TILE_GRID_SIZE)
            cv2.imwrite(str(dst_dir / f"{img_path.stem}_clip{cl}.png"), enhanced)
        print(f"✅ {img_path.name} 已輸出 original + {len(PREVIEW_CLIP_LIMITS)} 種clip_limit版本")

    print(f"\n📍 全部輸出到: {dst_dir}")
    print(f"   比較過後回來調整本檔案最上面的 CLAHE_CLIP_LIMIT 預設值，")
    print(f"   並且記得同步B1/B2那邊使用的同一個參數。")


# ============================================================
# 主流程
# ============================================================

def main():
    args = sys.argv[1:]

    if args and args[0] == "--preview":
        if len(args) != 3:
            print(f"用法: python {Path(__file__).name} --preview <輸入圖片資料夾> <預覽輸出資料夾>")
            sys.exit(1)
        preview(args[1], args[2])
        return

    if len(args) == 2:
        process_all(args[0], args[1])
    elif not args:
        process_all()
    else:
        print(f"用法:")
        print(f"  python {Path(__file__).name}                            # 用設定區的資料夾")
        print(f"  python {Path(__file__).name} <輸入資料夾> <輸出資料夾>")
        print(f"  python {Path(__file__).name} --preview <輸入資料夾> <預覽輸出資料夾>")
        sys.exit(1)


if __name__ == "__main__":
    main()
