#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py — 根管工作長度輔助估測系統(Gradio)
輸入：根尖 X 光 + 年齡 + 牙位 → 輸出：CLAHE 前處理圖、預測 mask 圖、校正後工作長度(mm)
用法：python app.py  → 瀏覽器開 http://127.0.0.1:7860
⚠️ 不要加 share=True：病人影像會經過公開網址。
"""
import gradio as gr

import pipeline_v2 as P

TITLE = "根管工作長度輔助估測系統"   # TODO 8：作品名稱


def single(img, age, position):
    if img is None:
        raise gr.Error("請先上傳根尖 X 光")
    try:
        return P.run(img, age, position)
    except (FileNotFoundError, ValueError, RuntimeError) as e:   # 多半是 seg 權重找不到
        raise gr.Error(str(e))


with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# 🦷 {TITLE}\n"
                "CLAHE 前處理 → YOLOv11-seg 牙齒分割 → 長軸幾何量測 → 長邊正規化 + 年齡迴歸 → 工作長度")

    with gr.Row():
        with gr.Column(scale=2):
            img_in = gr.Image(label="上傳根尖 X 光（治療前）", type="numpy", height=360)
        with gr.Column(scale=1):
            age = gr.Number(label="年齡", value=None, precision=0)
            pos = gr.Dropdown(["不指定"] + [str(p) for p in P.TOOTH_POSITIONS], value="不指定",
                              label="牙位 FDI（決定切端/根尖方向）")
            btn = gr.Button("開始估測", variant="primary")
            o_pred = gr.Number(label=f"估測工作長度 (mm，已校正 −{P.OFFSET_MM})")
            o_msg = gr.Textbox(label="狀態", lines=2)
    with gr.Row():
        o_enh = gr.Image(label="① CLAHE 前處理", height=380)
        o_seg = gr.Image(label="② 預測 mask 與長軸（A=切端、B=根尖）", height=380)
    with gr.Accordion("計算明細（像素長度等）", open=False):
        o_info = gr.JSON()

    btn.click(single, [img_in, age, pos], [o_enh, o_seg, o_pred, o_msg, o_info])

    gr.Markdown("⚠️ 研究原型，僅供輔助參考，不可取代臨床判斷。")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
