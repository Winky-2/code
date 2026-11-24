#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
records.py — 病例紀錄寫入 Excel
===============================================================
records/分析紀錄.xlsx   每按一次「存入紀錄」新增一列
records/images/        對應的結果圖(分割 + 長軸)，檔名 = 序號_病例代號.jpg

⚠️ Excel 開著紀錄表時寫不進去(Windows 會鎖檔)：會回報 RecordLocked，
   請使用者先關閉 Excel 再存一次；這時 Excel 與結果圖都不會被寫入。
⚠️ 只存去識別化代號，不要存姓名、病歷號。
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
RECORD_DIR = HERE / "records"
XLSX = RECORD_DIR / "分析紀錄.xlsx"
IMG_DIR = RECORD_DIR / "images"

COLUMNS = ["序號", "時間", "病例代號", "影像檔名", "年齡", "目標牙",
           "估測長度_mm", "95%區間下限_mm", "95%區間上限_mm", "預測標準差_mm",
           "GPR原始預測_mm", "提醒事項", "結果圖"]
WIDTHS = [6, 17, 12, 28, 6, 12, 12, 15, 15, 14, 15, 40, 26]


class RecordLocked(Exception):
    pass


def _safe(s):
    return re.sub(r"[^\w\-]", "_", str(s or "").strip())[:40]


def _new_book():
    wb = Workbook()
    ws = wb.active
    ws.title = "分析紀錄"
    ws.append(COLUMNS)
    for i, w in enumerate(WIDTHS, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
        c = ws.cell(1, i)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="0E7C86")
        c.alignment = Alignment(horizontal="center", vertical="center")
    ws.freeze_panes = "A2"
    return wb


def append(rec: dict, overlay_bgr) -> dict:
    """rec 需含 COLUMNS 中除 序號/時間/結果圖 以外的欄位。回傳寫入的那一列(dict)。"""
    RECORD_DIR.mkdir(exist_ok=True)
    IMG_DIR.mkdir(exist_ok=True)
    wb = load_workbook(XLSX) if XLSX.exists() else _new_book()
    ws = wb.active
    seq = max([c.value for c in ws["A"][1:] if isinstance(c.value, int)] or [0]) + 1
    img_name = f"{seq:04d}_{_safe(rec.get('病例代號')) or 'na'}.jpg"
    row = {**rec, "序號": seq, "時間": datetime.now().strftime("%Y-%m-%d %H:%M"),
           "結果圖": f"images/{img_name}"}
    ws.append([row.get(k) for k in COLUMNS])
    try:
        wb.save(XLSX)
    except PermissionError:
        raise RecordLocked("紀錄表正被 Excel 開啟，請先關閉 Excel 再按一次「存入紀錄」")
    ok, buf = cv2.imencode(".jpg", overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(str(IMG_DIR / img_name))         # tofile：中文路徑也寫得進去
    return row


def list_records(limit=100) -> list[dict]:
    """新到舊。檔案不存在回空 list。"""
    if not XLSX.exists():
        return []
    wb = load_workbook(XLSX, read_only=True)
    rows = list(wb.active.iter_rows(min_row=2, values_only=True))
    wb.close()
    out = [dict(zip(COLUMNS, r)) for r in rows if r and r[0] is not None]
    return out[::-1][:limit]
