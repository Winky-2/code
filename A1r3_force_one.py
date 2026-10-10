"""
A1r_force_one.py
================
救「A1r 完全配不到原圖（最佳 None）」的單張圖：由你指定原圖，強制對位。

為什麼 A1r 會失敗：資料集檔名(train_XXXX_tN)看不出原圖編號，程式只能靠特徵點投票找原圖；
裁切區特徵太少時投票失敗，多尺度比對備援也不會啟動。這裡直接指定原圖，
同時跑「放寬門檻的特徵點比對」與「多尺度 template 比對（含翻轉）」，取 NCC 最高者再做次像素精修。

輸出（不會直接改資料集）：
  <OUTPUT_DIR>/_review_未還原/候選還原/...   候選還原圖
  <OUTPUT_DIR>/_review_未還原/比對圖/..._forced.png   四格比對圖（看第3格棋盤格邊緣是否連續）
確認 OK 後執行 A1r_accept_review.py 收回（label 會一起移回）。

前置：本檔與 A1r_restore_noCLAHE_dataset.py 放在同一資料夾，且 OUTPUT_DIR 是用「新版」A1r 跑出來的。
"""
from pathlib import Path

import cv2
import numpy as np

import A1r_restore_noCLAHE_dataset as A

# ---------------- 設定 ----------------
OUTPUT_DIR = "train_single_seg-268_no_resize_noCLAHE"          # 👈 A1r 的 OUTPUT_DIR
TARGET = "train_0040_t0"                               # 👈 要救的那張（檔名開頭即可）
RAW_IMGS = [r"data set\Data_train\images\146_jpg.rf.slzoDglxga9MIq4DwZuN.jpg"]   # 👈 候選原圖（可列多張）


def sift_candidates(sift, it, ds):
    kp_r, d_r = sift.detectAndCompute(it["clahe"], None)
    out = []
    if d_r is None:
        return out
    bf = cv2.BFMatcher(cv2.NORM_L2)
    h, w = ds.shape
    for flip in ["", "h", "v", "hv"]:
        q = A.apply_flip(ds, flip)
        kp_q, d_q = sift.detectAndCompute(q, None)
        if d_q is None or len(d_q) < 2:
            continue
        good = [m[0] for m in bf.knnMatch(d_q, d_r, k=2)
                if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 4:
            continue
        src = np.float32([kp_r[m.trainIdx].pt for m in good])
        dst = np.float32([kp_q[m.queryIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=4.0, maxIters=10000)
        if M is not None and inl is not None and inl.sum() >= 4:
            out.append(((A.flip_matrix(flip, w, h) @ A.to3x3(M))[:2], flip, f"SIFT(inl={int(inl.sum())})"))
    return out


def main():
    out = Path(OUTPUT_DIR)
    rv = out / A.REVIEW_DIR
    hits = [p for p in (rv / "原CLAHE圖").rglob("*") if p.is_file() and p.name.startswith(TARGET)]
    if len(hits) != 1:
        print(f"❌ 在 {rv / '原CLAHE圖'} 找到 {len(hits)} 張開頭為 {TARGET} 的圖（要剛好 1 張）")
        return
    ds_path = hits[0]
    rel = ds_path.relative_to(rv / "原CLAHE圖")
    ds = A.read_gray(ds_path)
    h, w = ds.shape
    valid = ds > A.BLACK_BORDER_THRESHOLD
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.01)

    best = None
    for rp in RAW_IMGS:
        raw = A.read_gray(rp)
        it = dict(path=Path(rp), raw=raw, clahe=A.enhance_gray(raw))
        cands = sift_candidates(sift, it, ds) + \
                [(M, f, "template") for M, f in A.template_fallback(it, ds, valid)]
        for M, flip, how in cands:
            M2 = A.refine_ecc(it["clahe"], ds, M, valid)
            s1, _ = A.score_M(it, ds, M, valid)
            s2, _ = A.score_M(it, ds, M2, valid)
            M_, s_ = (M2, s2) if s2 >= s1 else (M, s1)
            print(f"   {Path(rp).name}  {how:<16} 翻轉={flip or '-':<2}  NCC={s_:.4f}")
            if best is None or s_ > best[0]:
                best = (s_, it, M_, how)

    if best is None:
        print("❌ 所有方法都對不上，請確認 RAW_IMGS 是不是這張的原圖")
        return
    score, it, M, how = best
    _, foot = A.score_M(it, ds, M, valid)
    warped_cl = cv2.warpAffine(it["clahe"], M, (w, h), flags=cv2.INTER_LINEAR)
    warped_raw = cv2.warpAffine(it["raw"], M, (w, h), flags=cv2.INTER_LINEAR)
    cand = ds.copy()
    cand[foot] = warped_raw[foot]
    color = A.imread_u(ds_path)
    cand_out = cv2.cvtColor(cand, cv2.COLOR_GRAY2BGR) if (color is not None and color.ndim == 3) else cand
    (rv / "候選還原" / rel).parent.mkdir(parents=True, exist_ok=True)
    A.imwrite_u(rv / "候選還原" / rel, cand_out, [cv2.IMWRITE_JPEG_QUALITY, 100])

    yy, xx = np.mgrid[0:h, 0:w]
    checker = np.where(((yy // 32 + xx // 32) % 2) == 0, ds, warped_cl)
    vis = []
    for im, t in zip([ds, warped_cl, checker, cand],
                     ["1 CLAHE dataset", "2 CLAHE(raw) warped", "3 checker 1+2", "4 restored (noCLAHE)"]):
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.putText(im, t, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        vis.append(im)
    vis = np.hstack(vis)
    cv2.putText(vis, f"FORCED  NCC={score:.4f}  {how}  src={it['path'].name}", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    (rv / "比對圖").mkdir(parents=True, exist_ok=True)
    png = rv / "比對圖" / ("__".join(rel.with_suffix("").parts) + "_forced.png")
    A.imwrite_u(png, vis)

    print("\n=======================================================")
    print(f"最佳：{it['path'].name}  {how}  NCC={score:.4f}")
    if score >= 0.97:
        print("✅ 對位品質好（≥0.97），看一下比對圖確認後執行 A1r_accept_review.py 收回")
    elif score >= 0.90:
        print("⚠️ 對位品質普通，務必仔細看比對圖第3格；邊緣有錯開就不要收")
    else:
        print("❌ NCC 很低，大概不是這張原圖或無法對位；請刪掉候選圖，改用 GIMP 手動或換 RAW_IMGS")
    print(f"📍 比對圖：{png}")
    print(f"📍 候選圖：{rv / '候選還原' / rel}")
    print("=======================================================")


if __name__ == "__main__":
    main()
