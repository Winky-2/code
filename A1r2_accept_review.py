"""
A1r_accept_review.py
====================
方案D 第二步：把「目視確認 OK」的候選還原圖收回無CLAHE資料集。

前置：
  1. 已用新版 A1r 跑過，輸出資料夾裡有 _review_未還原/候選還原/ 與 _review_未還原/比對圖/
  2. 打開「比對圖」逐張看（第3格棋盤格：邊緣連續＝對準；邊緣錯開＝對歪）
  3. 對歪的 → 把 _review_未還原/候選還原/ 裡「同名」的圖刪掉
     （不確定的也刪掉，寧可剔除不要收錯）

本程式：把 候選還原/ 裡剩下的圖放回資料集原位置、label 移回原位置，
        更新 A1r_配對QC.csv 與 未還原清單.txt。
"""
import shutil
from pathlib import Path

import pandas as pd

# ---------------- 設定 ----------------
OUTPUT_DIR = "train_single_seg-268_no_resize_noCLAHE"     # 👈 A1r 的 OUTPUT_DIR
REVIEW_DIR = "_review_未還原"
QC_CSV = "A1r_配對QC.csv"


def main():
    out = Path(OUTPUT_DIR)
    rv = out / REVIEW_DIR
    cand_root = rv / "候選還原"
    if not cand_root.exists():
        print(f"❌ 找不到 {cand_root}（要先用新版 A1r 跑）")
        return
    df = pd.read_csv(out / QC_CSV, encoding="utf-8-sig")
    n_ok = 0
    for i, r in df.iterrows():
        if not str(r["狀態"]).startswith("❌"):
            continue
        rel = Path(str(r["資料集圖片"]))
        cand = cand_root / rel
        if not cand.exists():
            continue                                         # 被你刪掉（判定對歪）或本來就沒有候選
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cand, out / rel)
        lrel = str(r.get("label原位置", "") or "")
        if lrel and lrel != "nan":
            src = rv / "label" / lrel
            if src.exists():
                (out / lrel).parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(out / lrel))
            else:
                print(f"⚠️ {rel}: 找不到暫存的 label {src}，請手動確認")
        df.at[i, "狀態"] = "✅ 人工目視確認後收回"
        n_ok += 1
        print(f"✅ 收回 {rel}")

    df.to_csv(out / QC_CSV, index=False, encoding="utf-8-sig")
    bad = df.loc[df["狀態"].astype(str).str.startswith("❌"), "資料集圖片"].tolist()
    (out / "未還原清單.txt").write_text("\n".join(bad), encoding="utf-8")
    print("\n=======================================================")
    print(f"收回 {n_ok} 張；仍未還原 {len(bad)} 張（清單已更新：未還原清單.txt）")
    print("👉 CLAHE 組只需剔除「更新後」清單裡的圖")
    print("=======================================================")


if __name__ == "__main__":
    main()
