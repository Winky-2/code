"""
A5_pad_letterbox_640.py
============================
A4 裁切後的單顆牙圖 -> 上下補黑邊 -> letterbox 到 640x640（左右再補黑邊）
並「同步換算」YOLO 標註（det / pose / seg 皆可）。

前置：先跑 A4，產生 cropped_train_teeth_yolov11/ 與 train裁切結果_yolov11.xlsx
產出：OUTPUT_ROOT/<run_tag>/images, labels, qc, A5_manifest.xlsx, config.json
      ※ 只新增不覆寫：同一個 run_tag 已存在就停止

*** LABEL_SPACE：你的 label 是相對於哪張圖正規化？ ***
  "crop"：label 是在「A4 裁切後的圖」上標的（例如裁切後才上 Roboflow 標）
  "full"：label 是在「整張原圖」上標的（A4 讀的那種 GT label）
          -> 會先用 A4 manifest 的「裁切像素座標_x1y1x2y2」換到裁切座標，再換到 640

*** 換算公式（B4 還原長度時要用，參數都存在 A5_manifest.xlsx） ***
  640 -> 裁切圖像素：
      X_crop = (X_640 - pad_left) / scale_x
      Y_crop = (Y_640 - pad_top)  / scale_y - pad_tb_px
  裁切圖 -> 原圖像素：X_full = X_crop + crop_x1，Y_full = Y_crop + crop_y1（A4 manifest）
  ⚠️ 長度請先還原到原圖像素空間再算，不要在 640 空間算（每張 scale 不同）

⚠️ 黑邊值 PAD_VALUE=0；Ultralytics 自己的 letterbox 是 114 灰。圖已經是 640x640，
   B1/B2 用 imgsz=640 時不會再補邊；若再上 Roboflow，Resize 請設不處理。
⚠️ train 與 test 必須用同一個 PAD_TB_RATIO 跑過，否則分布不一致。

安裝需求：
    pip install opencv-python pandas openpyxl
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# ==================== 可手動修改 ====================
RUN_NOW = True

INPUT_IMG_DIR = "test_single_seg-67_no_resize/test/images"          # A4 輸出的裁切圖

LABEL_SPACE = "crop"                                    # "crop" 或 "full"（見檔頭）
CROP_LABEL_DIR = "test_single_seg-67_no_resize/test/labels"   # LABEL_SPACE="crop" 時：裁切圖的 label 資料夾
FULL_LABEL_DIR = "train_crop.yolov11/train/labels"      # LABEL_SPACE="full" 時：原圖 label
FULL_IMG_DIR = "train_crop.yolov11/train/images"        # LABEL_SPACE="full" 時：讀原圖寬高
A4_MANIFEST_XLSX = "train裁切結果_yolov11.xlsx"           # LABEL_SPACE="full" 時：A4 裁切座標

LABEL_FORMAT = "seg"        # "det"(cls cx cy w h) / "pose"(det + N_KPT*(x y v)) / "seg"(cls x1 y1 ...)
N_KPT = 2                   # pose 才用到

PAD_TB_RATIO = 0.10         # 上下各補 H*ratio 像素黑邊；設 0 = 對照組（只做 letterbox）
OUT_SIZE = 640
PAD_VALUE = 0

OUTPUT_ROOT = "A5_test_padded"
QC_N = 10                   # 畫前 N 張疊圖供肉眼檢查框有沒有對齊（0 = 不畫）

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def make_run_tag():
    return f"tb{PAD_TB_RATIO:.2f}_{LABEL_SPACE}_{LABEL_FORMAT}_{OUT_SIZE}"


# ============================================================
# 幾何
# ============================================================

def compute_geom(W, H, ratio, out_size):
    """回傳 pad/scale 參數。scale 用實際取整後尺寸反推，避免次像素誤差。"""
    p = int(round(H * ratio))
    Hp = H + 2 * p
    s = out_size / max(W, Hp)
    nw = max(1, int(round(W * s)))
    nh = max(1, int(round(Hp * s)))
    return {
        "pad_tb_px": p,
        "scale_x": nw / W,
        "scale_y": nh / Hp,
        "new_w": nw,
        "new_h": nh,
        "pad_left": (out_size - nw) // 2,
        "pad_top": (out_size - nh) // 2,
    }


def pad_and_letterbox(img, g, out_size, pad_value):
    import cv2
    import numpy as np

    p = g["pad_tb_px"]
    padded = cv2.copyMakeBorder(img, p, p, 0, 0, cv2.BORDER_CONSTANT,
                                value=[pad_value] * (img.shape[2] if img.ndim == 3 else 1))
    shrink = g["new_w"] < padded.shape[1]
    resized = cv2.resize(padded, (g["new_w"], g["new_h"]),
                         interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
    canvas_shape = (out_size, out_size) + ((img.shape[2],) if img.ndim == 3 else ())
    canvas = np.full(canvas_shape, pad_value, dtype=img.dtype)
    y0, x0 = g["pad_top"], g["pad_left"]
    canvas[y0:y0 + g["new_h"], x0:x0 + g["new_w"]] = resized
    return canvas


class LabelMapper:
    """來源正規化座標 -> 640 正規化座標。
    src_w/src_h：label 正規化所依據的圖尺寸；ox/oy：該圖到裁切圖的像素位移（crop 模式為 0）。"""

    def __init__(self, g, W, H, src_w, src_h, ox, oy, out_size):
        self.g, self.W, self.H = g, W, H
        self.src_w, self.src_h, self.ox, self.oy = src_w, src_h, ox, oy
        self.out = out_size
        # 裁切圖內容在 640 畫布上的範圍（像素），用來 clip 與檢查越界
        self.x_lo = g["pad_left"]
        self.x_hi = g["pad_left"] + W * g["scale_x"]
        self.y_lo = g["pad_top"] + g["pad_tb_px"] * g["scale_y"]
        self.y_hi = g["pad_top"] + (g["pad_tb_px"] + H) * g["scale_y"]

    def pt_px(self, x, y):
        g = self.g
        xc = x * self.src_w - self.ox
        yc = y * self.src_h - self.oy
        return xc * g["scale_x"] + g["pad_left"], (yc + g["pad_tb_px"]) * g["scale_y"] + g["pad_top"]

    def inside(self, X, Y, tol=0.5):
        return (self.x_lo - tol <= X <= self.x_hi + tol) and (self.y_lo - tol <= Y <= self.y_hi + tol)

    def clip(self, X, Y):
        return min(max(X, self.x_lo), self.x_hi), min(max(Y, self.y_lo), self.y_hi)

    def transform_line(self, line, fmt, n_kpt):
        """回傳 (新的 label 行 or None, 警告字串清單)。"""
        v = line.split()
        if not v:
            return None, []
        cls, nums = v[0], [float(t) for t in v[1:]]
        warns = []
        n = self.out
        f6 = lambda t: f"{t:.6f}"

        if fmt == "seg":
            if len(nums) < 6 or len(nums) % 2:
                return None, [f"seg 座標數量異常({len(nums)})，略過此行"]
            out, n_out = [], 0
            for i in range(0, len(nums), 2):
                X, Y = self.pt_px(nums[i], nums[i + 1])
                if not self.inside(X, Y):
                    n_out += 1
                    X, Y = self.clip(X, Y)
                out += [f6(X / n), f6(Y / n)]
            if n_out:
                warns.append(f"cls{cls} 多邊形有 {n_out} 點超出裁切範圍，已 clip")
            return " ".join([cls] + out), warns

        expected = 4 + (3 * n_kpt if fmt == "pose" else 0)
        if len(nums) < expected:
            return None, [f"{fmt} 欄位數不足({len(nums)}<{expected})，略過此行"]

        cx, cy, w, h = nums[:4]
        X1, Y1 = self.pt_px(cx - w / 2, cy - h / 2)
        X2, Y2 = self.pt_px(cx + w / 2, cy + h / 2)
        if not (self.inside(X1, Y1) and self.inside(X2, Y2)):
            warns.append(f"cls{cls} 框超出裁切範圍，已 clip")
            X1, Y1 = self.clip(X1, Y1)
            X2, Y2 = self.clip(X2, Y2)
        if X2 - X1 < 1 or Y2 - Y1 < 1:
            return None, warns + [f"cls{cls} 框 clip 後面積為 0，略過此行"]
        out = [f6((X1 + X2) / 2 / n), f6((Y1 + Y2) / 2 / n), f6((X2 - X1) / n), f6((Y2 - Y1) / n)]

        if fmt == "pose":
            for k in range(n_kpt):
                kx, ky, kv = nums[4 + 3 * k: 7 + 3 * k]
                if kv == 0:
                    out += ["0.000000", "0.000000", "0"]   # 不可見點維持 (0,0,0)
                    continue
                X, Y = self.pt_px(kx, ky)
                if not self.inside(X, Y):
                    warns.append(f"cls{cls} 關鍵點{k} 超出裁切範圍(未 clip，請人工複查)")
                out += [f6(X / n), f6(Y / n), str(int(kv))]
        return " ".join([cls] + out), warns


# ============================================================
# QC 疊圖
# ============================================================

def draw_qc(img, label_lines, fmt, n_kpt, out_size):
    import cv2
    import numpy as np

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for line in label_lines:
        v = [float(t) for t in line.split()[1:]]
        if fmt == "seg":
            pts = (np.array(v).reshape(-1, 2) * out_size).round().astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 1)
            continue
        cx, cy, w, h = [t * out_size for t in v[:4]]
        cv2.rectangle(vis, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (0, 255, 0), 1)
        if fmt == "pose":
            for k in range(n_kpt):
                kx, ky, kv = v[4 + 3 * k: 7 + 3 * k]
                if kv > 0:
                    c = (int(kx * out_size), int(ky * out_size))
                    cv2.circle(vis, c, 4, (0, 0, 255), -1)
                    cv2.putText(vis, "AB"[k] if n_kpt == 2 else str(k), (c[0] + 5, c[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis


# ============================================================
# 主流程
# ============================================================

def load_a4_crop_coords():
    import pandas as pd

    df = pd.read_excel(A4_MANIFEST_XLSX)
    coords = {}
    for _, r in df.iterrows():
        s = r.get("裁切像素座標_x1y1x2y2")
        if isinstance(s, str) and s.count(",") == 3:
            coords[str(r["圖片檔名"])] = tuple(int(t) for t in s.split(","))
    print(f"✅ 讀到 A4 裁切座標 {len(coords)} 筆（{A4_MANIFEST_XLSX}）")
    return coords


def run():
    import cv2
    import pandas as pd

    if LABEL_SPACE not in ("crop", "full"):
        raise ValueError(f"LABEL_SPACE 只能是 'crop' 或 'full'，目前是 {LABEL_SPACE!r}")
    if LABEL_FORMAT not in ("det", "pose", "seg"):
        raise ValueError(f"LABEL_FORMAT 只能是 det/pose/seg，目前是 {LABEL_FORMAT!r}")

    in_dir = Path(INPUT_IMG_DIR)
    if not in_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 {in_dir}，請先跑 A4")
    lbl_dir = Path(CROP_LABEL_DIR if LABEL_SPACE == "crop" else FULL_LABEL_DIR)
    if not lbl_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 label 資料夾 {lbl_dir}")

    run_dir = Path(OUTPUT_ROOT) / make_run_tag()
    if run_dir.exists():
        print(f"⚠️ {run_dir} 已存在，為避免覆寫舊結果直接停止；要重跑請先刪除或改設定。")
        return
    out_img, out_lbl, out_qc = run_dir / "images", run_dir / "labels", run_dir / "qc"
    for d in (out_img, out_lbl):
        d.mkdir(parents=True)
    if QC_N > 0:
        out_qc.mkdir(parents=True)

    crop_coords = load_a4_crop_coords() if LABEL_SPACE == "full" else {}

    img_paths = sorted(p for p in in_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    print(f"\n=== A5：{len(img_paths)} 張，PAD_TB_RATIO={PAD_TB_RATIO}，"
          f"LABEL_SPACE={LABEL_SPACE}，LABEL_FORMAT={LABEL_FORMAT} ===")

    records = []
    n_qc = 0
    for img_path in img_paths:
        rec = {"圖片檔名": img_path.name}
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            rec["狀態"] = "❌ 讀圖失敗"
            records.append(rec)
            continue
        H, W = img.shape[:2]
        g = compute_geom(W, H, PAD_TB_RATIO, OUT_SIZE)
        rec.update({"原W": W, "原H": H, **g})

        # 決定 label 的來源尺寸與位移
        if LABEL_SPACE == "crop":
            src_w, src_h, ox, oy = W, H, 0, 0
        else:
            if img_path.name not in crop_coords:
                rec["狀態"] = "❌ A4 manifest 找不到裁切座標，跳過"
                records.append(rec)
                continue
            x1, y1, x2, y2 = crop_coords[img_path.name]
            if (x2 - x1, y2 - y1) != (W, H):
                rec["狀態"] = f"❌ A4 裁切座標尺寸({x2 - x1}x{y2 - y1})與裁切圖({W}x{H})不符，跳過"
                records.append(rec)
                continue
            full_img = cv2.imread(str(Path(FULL_IMG_DIR) / img_path.name), cv2.IMREAD_UNCHANGED)
            if full_img is None:
                rec["狀態"] = f"❌ 讀不到原圖 {FULL_IMG_DIR}/{img_path.name}（需要原圖寬高），跳過"
                records.append(rec)
                continue
            src_h, src_w = full_img.shape[:2]
            ox, oy = x1, y1
            rec.update({"crop_x1": x1, "crop_y1": y1})

        out = pad_and_letterbox(img, g, OUT_SIZE, PAD_VALUE)
        cv2.imwrite(str(out_img / img_path.name), out)

        label_path = lbl_dir / (img_path.stem + ".txt")
        warns, new_lines = [], []
        if not label_path.exists():
            warns.append("找不到 label（只輸出圖，視為背景圖）")
        else:
            mapper = LabelMapper(g, W, H, src_w, src_h, ox, oy, OUT_SIZE)
            for line in label_path.read_text(encoding="utf-8").splitlines():
                new, w = mapper.transform_line(line, LABEL_FORMAT, N_KPT)
                warns += w
                if new:
                    new_lines.append(new)
            (out_lbl / label_path.name).write_text("\n".join(new_lines) + ("\n" if new_lines else ""),
                                                   encoding="utf-8")

        if n_qc < QC_N:
            cv2.imwrite(str(out_qc / f"{img_path.stem}_qc.png"),
                        draw_qc(out, new_lines, LABEL_FORMAT, N_KPT, OUT_SIZE))
            n_qc += 1

        rec["label行數"] = len(new_lines)
        rec["狀態"] = "⚠️ " + "；".join(warns) if warns else "✅"
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_excel(run_dir / "A5_manifest.xlsx", index=False)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: globals()[k] for k in (
            "INPUT_IMG_DIR", "LABEL_SPACE", "CROP_LABEL_DIR", "FULL_LABEL_DIR", "FULL_IMG_DIR",
            "A4_MANIFEST_XLSX", "LABEL_FORMAT", "N_KPT", "PAD_TB_RATIO", "OUT_SIZE", "PAD_VALUE")}
                  | {"run_tag": make_run_tag(), "time": datetime.now().isoformat(timespec="seconds")},
                  f, ensure_ascii=False, indent=2)

    st = df["狀態"].astype(str)
    print(f"\n=======================================================")
    print(f"✨ A5 完成：{len(df)} 張")
    print(f"   ✅ 正常: {(st == '✅').sum()} 張")
    print(f"   ⚠️ 有警告: {st.str.startswith('⚠️').sum()} 張 👈 看 manifest 的「狀態」欄")
    print(f"   ❌ 跳過: {st.str.startswith('❌').sum()} 張")
    print(f"📍 輸出: {run_dir}/images, labels")
    print(f"📍 QC 疊圖: {out_qc}（先看框有沒有對齊再往下跑）")
    print(f"📍 還原參數: {run_dir / 'A5_manifest.xlsx'}")
    print(f"=======================================================")


if __name__ == "__main__":
    if RUN_NOW:
        run()
    else:
        print(__doc__)
        print(f"待執行設定：run_tag={make_run_tag()}，PAD_TB_RATIO={PAD_TB_RATIO}，"
              f"LABEL_SPACE={LABEL_SPACE}，LABEL_FORMAT={LABEL_FORMAT}")
        print("👉 確認後把 RUN_NOW 改成 True 再執行")
