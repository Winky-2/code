#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
server.py — 根管工作長度智慧估測系統(本機網頁版)
用法：python server.py  → 自動開啟瀏覽器 http://127.0.0.1:7860
完全離線可用；只聽 127.0.0.1，病人影像不會離開這台電腦。
"""
import base64
import os
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

import pipeline as P
import records as R

HERE = Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 7860
MAX_SIDE = 1600                     # 回傳給瀏覽器的影像長邊上限(只影響顯示)

app = Flask(__name__, static_folder=str(HERE / "static"))
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024
_lock = threading.Lock()


def to_data_url(bgr, fmt=".jpg"):
    h, w = bgr.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(fmt, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/status")
def status():
    out = {"det": {"path": Path(P.DET_WEIGHTS).name, "ok": Path(P.DET_WEIGHTS).exists()},
           "seg": {"path": Path(P.SEG_WEIGHTS).name, "ok": Path(P.SEG_WEIGHTS).exists()},
           "preprocess": P.PREPROCESS_LABEL.get(P.PREPROCESS, P.PREPROCESS),
           "offset": P.OFFSET_MM, "device": P.device_name(), "teeth": P.TOOTH_POSITIONS}
    try:
        out["gpr"] = {"ok": True, **P.get_gpr().info()}
    except Exception as e:
        out["gpr"] = {"ok": False, "error": str(e)}
    return jsonify(out)


EXAMPLE_DIR = HERE / "examples"     # 放幾張去識別化的整張 X 光，展示時一鍵載入
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


@app.get("/api/examples")
def examples():
    if not EXAMPLE_DIR.is_dir():
        return jsonify([])
    return jsonify(sorted(p.name for p in EXAMPLE_DIR.iterdir() if p.suffix.lower() in IMG_EXT)[:12])


@app.get("/examples/<path:name>")
def example_file(name):
    return send_from_directory(EXAMPLE_DIR, name)


_cache = OrderedDict()                # token → (整張影像, 偵測結果)；只留最近幾張
_results = OrderedDict()              # rid → 待存入紀錄的分析結果(按「存入紀錄」才寫檔)
_rec_lock = threading.Lock()


def _read_upload():
    f = request.files.get("image")
    if f is None:
        return None, "請先上傳影像"
    bgr = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    return (bgr, None) if bgr is not None else (None, "無法讀取影像，請上傳 JPG / PNG")


@app.post("/api/detect")
def detect():
    bgr, err = _read_upload()
    if err:
        return jsonify({"ok": False, "msg": err}), 400
    t0 = time.perf_counter()
    try:
        with _lock:
            boxes = P.detect_teeth(bgr)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "msg": f"{type(e).__name__}：{e}"}), 500
    ms = round((time.perf_counter() - t0) * 1000, 1)
    token = uuid.uuid4().hex
    _cache[token] = (bgr, boxes)
    while len(_cache) > 20:
        _cache.popitem(last=False)
    h, w = bgr.shape[:2]
    return jsonify({"ok": True, "token": token, "boxes": boxes, "size": [w, h], "ms": ms,
                    "image": to_data_url(bgr),
                    "msg": "" if boxes else "偵測器沒有找到牙齒，請確認上傳的是根尖 X 光"})


@app.post("/api/predict")
def predict():
    js = request.get_json(silent=True) or {}
    item = _cache.get(js.get("token"))
    if item is None:
        return jsonify({"ok": False, "msg": "影像已過期，請重新上傳"}), 400
    bgr, boxes = item
    i = js.get("box")
    if not isinstance(i, int) or not 0 <= i < len(boxes):
        return jsonify({"ok": False, "msg": "請先在影像上點選目標牙"}), 400
    age = js.get("age")
    if age is None:
        return jsonify({"ok": False, "msg": "請輸入年齡"}), 400
    t0 = time.perf_counter()
    try:
        with _lock:
            r = P.run(bgr, boxes[i]["box"], float(age), js.get("position"), det_conf=boxes[i]["conf"])
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "msg": f"{type(e).__name__}：{e}"}), 500
    rid = uuid.uuid4().hex
    _results[rid] = {"overlay": r["overlay"], "saved": None, "rec": {
        "影像檔名": js.get("filename") or "", "年齡": float(age),
        "目標牙": f"#{i + 1}（共 {len(boxes)} 顆）",
        "估測長度_mm": r["pred_mm"], "95%區間下限_mm": r["pi_low"], "95%區間上限_mm": r["pi_high"],
        "預測標準差_mm": r["sd_mm"], "GPR原始預測_mm": r["raw_mm"],
        "提醒事項": "；".join(n for n in r["notes"] if not n.startswith("未指定牙位"))}}
    while len(_results) > 50:
        _results.popitem(last=False)
    for k in ("original", "enhanced", "overlay"):
        r[k] = to_data_url(r[k])
    r["rid"] = rid
    r["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return jsonify(r)


# ============================================================
# 病例紀錄(Excel)
# ============================================================
@app.post("/api/records")
def save_record():
    js = request.get_json(silent=True) or {}
    item = _results.get(js.get("rid"))
    if item is None:
        return jsonify({"ok": False, "msg": "找不到這次的分析結果，請重新分析"}), 400
    if item["saved"]:
        return jsonify({"ok": True, "row": item["saved"], "dup": True})
    case_id = str(js.get("case_id") or "").strip()[:40]
    try:
        with _rec_lock:
            row = R.append({**item["rec"], "病例代號": case_id}, item["overlay"])
    except R.RecordLocked as e:
        return jsonify({"ok": False, "msg": str(e)}), 409
    except Exception as e:
        traceback.print_exc()
        return jsonify({"ok": False, "msg": f"{type(e).__name__}：{e}"}), 500
    item["saved"] = row
    return jsonify({"ok": True, "row": row})


@app.get("/api/records")
def get_records():
    try:
        return jsonify({"ok": True, "rows": R.list_records()})
    except Exception as e:
        return jsonify({"ok": False, "msg": f"讀取紀錄表失敗：{e}", "rows": []})


@app.get("/api/records/download")
def download_records():
    if not R.XLSX.exists():
        return jsonify({"ok": False, "msg": "還沒有任何紀錄"}), 404
    return send_file(R.XLSX, as_attachment=True, download_name="分析紀錄.xlsx")


@app.post("/api/records/open")
def open_records():
    if not R.XLSX.exists():
        return jsonify({"ok": False, "msg": "還沒有任何紀錄"}), 404
    try:
        if sys.platform.startswith("win"):
            os.startfile(R.XLSX)                     # 用電腦預設程式(Excel)開啟
        else:
            return jsonify({"ok": False, "msg": "只支援在 Windows 直接開啟，請改用「下載紀錄表」"}), 400
    except Exception as e:
        return jsonify({"ok": False, "msg": f"無法開啟：{e}"}), 500
    return jsonify({"ok": True})


@app.get("/records/<path:name>")
def record_file(name):
    return send_from_directory(R.RECORD_DIR, name)


def warmup():
    """先載入模型並跑一次空白圖，第一次按下估測就不會卡好幾秒。"""
    try:
        blank = np.zeros((P.IMG_SIZE, P.IMG_SIZE, 3), np.uint8)
        P.get_det().predict(blank, verbose=False)
        P.get_seg().predict(blank, verbose=False)
        P.get_gpr()
        print("✅ 模型已載入")
    except Exception as e:
        print(f"⚠️ 預載模型失敗(介面仍可開啟，按估測時會顯示原因)：{e}")


if __name__ == "__main__":
    warmup()
    url = f"http://{HOST}:{PORT}"
    print(f"👉 {url}")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=HOST, port=PORT, debug=False)
