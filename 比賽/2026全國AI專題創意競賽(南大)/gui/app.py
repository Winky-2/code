#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py — 根管工作長度輔助估測系統(Gradio)
用法：python app.py  → 瀏覽器開 http://127.0.0.1:7860
⚠️ 不要加 share=True：病人影像會經過公開網址。
"""
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

import pipeline as P

TITLE = "根管工作長度輔助估測系統"   # TODO 8：作品名稱


def single(img, age, position, true_mm):
    if img is None:
        raise gr.Error("請先上傳根尖 X 光")
    try:
        return P.run(img, age, position, true_mm)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        raise gr.Error(str(e))


def batch(files, table):
    if not files or table is None:
        raise gr.Error("請上傳影像和醫師資料表")
    try:
        tab = P.read_doctor_table(table)
        reg = P.get_reg()
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        raise gr.Error(str(e))

    rows = []
    for f in files:
        name = Path(f).name
        r = tab.loc[P.name_key(name)] if P.name_key(name) in tab.index else None
        rec = {"檔名": name}
        if r is None:
            rows.append({**rec, "狀態": "表格裡找不到這張"})
            continue
        age = None if pd.isna(r["年齡"]) else float(r["年齡"])
        pos = None if pd.isna(r["牙位"]) else int(r["牙位"])
        true_mm = None if pd.isna(r["實際長度_mm"]) else float(r["實際長度_mm"])
        img = P.imread_u(f)
        if img is None:
            rows.append({**rec, "狀態": "讀不到影像"})
            continue
        m = P.measure(img, pos)
        if not m["ok"]:
            rows.append({**rec, "狀態": m["msg"]})
            continue
        try:
            pred = reg.predict(m["feat"], age) - P.OFFSET_MM
        except ValueError as e:
            rows.append({**rec, "狀態": str(e)})
            continue
        status, err = P.judge(pred, true_mm)
        rows.append({**rec, "牙位": pos, "年齡": age, "mask數": len(m["polys"]),
                     "正規化特徵": round(m["feat"], 3), "校正後預測_mm": round(pred, 2),
                     "實際長度_mm": true_mm, "誤差_mm": None if err is None else round(err, 2),
                     "判讀": status, "狀態": "✅" + ("（多顆牙，請確認選牙）" if len(m["polys"]) > 1 else "")})

    df = pd.DataFrame(rows)
    summary = "—"
    if "誤差_mm" in df and df["誤差_mm"].notna().any():
        e = df["誤差_mm"].dropna().to_numpy(float)
        ok = df.dropna(subset=["誤差_mm"])
        r = np.corrcoef(ok["校正後預測_mm"], ok["實際長度_mm"])[0, 1] if len(ok) > 2 else float("nan")
        summary = (f"n = {len(e)}　MAE {np.abs(e).mean():.3f} mm　RMSE {np.sqrt((e**2).mean()):.3f} mm　"
                   f"R {r:.3f}　Bias {e.mean():+.3f} mm　理想比率 {(np.abs(e) <= P.IDEAL_TOLERANCE_MM).mean()*100:.1f}%")
    out = Path(tempfile.mkdtemp()) / "批次估測結果.xlsx"
    df.to_excel(out, index=False)
    return summary, df, str(out)


with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🦷 {TITLE}\n"
                "CLAHE 前處理 → YOLOv11-seg 牙齒分割 → 長軸幾何量測 → 長邊正規化 + 年齡迴歸 → 臨床判讀")

    with gr.Tab("單張估測"):
        with gr.Row():
            with gr.Column(scale=2):
                img_in = gr.Image(label="上傳根尖 X 光（治療前）", type="numpy", height=360)
            with gr.Column(scale=1):
                age = gr.Number(label="年齡", value=None, precision=0)
                pos = gr.Dropdown(["不指定"] + [str(p) for p in P.TOOTH_POSITIONS], value="不指定",
                                  label="牙位 FDI（只影響切端/根尖方向標示，不影響長度）")
                true_mm = gr.Number(label="實際充填長度 mm（選填，有填才判讀）", value=None)
                btn = gr.Button("開始估測", variant="primary")
        with gr.Row():
            o_raw = gr.Image(label="① 原始影像", height=300)
            o_enh = gr.Image(label="② CLAHE 前處理", height=300)
            o_seg = gr.Image(label="③ 分割與長軸量測（A=切端、B=根尖）", height=300)
        with gr.Row():
            o_pred = gr.Number(label=f"④ 估測工作長度 (mm，已 −{P.OFFSET_MM})")
            o_status = gr.Textbox(label="⑤ 臨床判讀")
        o_info = gr.JSON(label="計算明細")
        btn.click(single, [img_in, age, pos, true_mm], [o_raw, o_enh, o_seg, o_pred, o_status, o_info])

    with gr.Tab("批次估測"):
        gr.Markdown("上傳多張影像 ＋ 醫師資料表（跟 `根管充填長度_*.xlsx` 同格式：圖片檔名 / 填充物長度(mm) / 牙位 / 年齡）。"
                    "檔名用開頭編號對應（001.jpg ↔ 1）。")
        with gr.Row():
            files = gr.File(label="影像（可多選）", file_count="multiple", type="filepath")
            table = gr.File(label="醫師資料表 xlsx / csv", type="filepath")
        bbtn = gr.Button("批次執行", variant="primary")
        b_sum = gr.Textbox(label="整體指標（校正後）")
        b_df = gr.Dataframe(label="逐張結果", wrap=True)
        b_file = gr.File(label="下載結果 xlsx")
        bbtn.click(batch, [files, table], [b_sum, b_df, b_file])

    gr.Markdown("⚠️ 研究原型，僅供輔助參考，不可取代臨床判斷。")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
