"""
0_preprocess_before_annotation.py
============================
在「上傳Roboflow標註」之前，先對整批原始X光片套用CLAHE清晰化，
把處理過的圖片存到新資料夾，之後拿這個資料夾的圖去Roboflow標註
（而不是標原圖）。

*** 這支跟 image_enhance.py 最下面那個 __main__ 小工具的差別 ***
那個小工具是「探索用」的：只挑前5張圖、每張輸出好幾種clip_limit
版本，讓你並排比較、肉眼決定哪個clip_limit最好。
這支是「量產用」的：clip_limit已經決定好了，對整個資料夾的每一張
圖都套用同一組參數，輸出「一張圖一個結果」，資料夾內容可以直接
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
     image_enhance.enhance_image() 的輸出來源，B1不要再處理一次）。
  3. B2推論全新病例時，如果新病例給的是「未增強的原圖」，那麼
     推論前仍然要用同一組clip_limit跑過這支腳本的enhance_image()，
     這樣模型看到的分布才會跟訓練時一致。也就是說：這支腳本產生
     的參數(CLIP_LIMIT、TILE_GRID_SIZE)之後在B1/B2整條線上都要
     沿用同一份，不能各自調整。

安裝需求：
    pip install opencv-python
"""

from pathlib import Path

import cv2
import pandas as pd

from image_enhance import enhance_image, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE

# ---------------- 設定區 ----------------
INPUT_DIR = "Data_second"       # 👈 原始、尚未標註的X光片資料夾
OUTPUT_DIR = "raw_xrays_enhanced_1"   # 👈 增強後、要拿去Roboflow標註的輸出資料夾

# 沿用 image_enhance.py 裡的預設值，確保跟B1/B2未來會用的是同一組參數。
# 如果你已經用 image_enhance.py 的預覽小工具比較過、決定了不同的
# clip_limit，記得同步回去改 image_enhance.py 的 CLAHE_CLIP_LIMIT，
# 而不是只改這裡——這樣才能保持「單一參數來源」，不會日後訓練/推論
# 時漏改、兩邊對不齊。
CLIP_LIMIT = CLAHE_CLIP_LIMIT
TILE_GRID_SIZE = CLAHE_TILE_GRID_SIZE

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# 已經處理過的檔名如果在OUTPUT_DIR已存在，預設跳過（方便中斷後續跑）。
# 想強制全部重跑就設成True。
OVERWRITE_EXISTING = False

MANIFEST_CSV = "前處理清單_增強前後對照.csv"  # 留一份可追溯紀錄，report用得到


def collect_images(input_dir: Path):
    paths = sorted(p for p in input_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    return paths


def process_all():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

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
    print(f"\n👉 提醒：正式標註前，建議先用 image_enhance.py 的預覽小工具")
    print(f"   (python image_enhance.py <少量圖片資料夾> <預覽輸出資料夾>)")
    print(f"   肉眼確認 clip_limit={CLIP_LIMIT} 是「變清晰」而不是「變雜訊」，")
    print(f"   確定滿意再跑這支批次處理全部資料，避免標註完才發現參數要重調。")
    print(f"\n⚠️ 這批增強後的圖以後就是你的新「原始資料」，之後接上B1訓練時，")
    print(f"   要確認B1不會對它再套用一次CLAHE（雙重增強），且B2推論全新病例的")
    print(f"   原圖時，也要用同一組clip_limit跑過這裡的enhance_image()，")
    print(f"   train/test/未來推論三邊的前處理狀態才會一致。")


if __name__ == "__main__":
    process_all()