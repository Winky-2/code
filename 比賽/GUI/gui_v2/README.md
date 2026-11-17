# 根管工作長度智慧估測系統 — GUI v4（整張 X 光 → 點選目標牙）

## 檔案
| 檔案 | 用途 |
|---|---|
| `server.py` / `啟動.bat` | 啟動 GUI（本機網頁，離線可用） |
| `pipeline.py` | 整張 X 光 → A3 偵測 → 點選目標牙 → A4 裁切 → top-hat → A5 letterbox → YOLO11n-seg → 長軸量測 → 長邊正規化（整張長邊）→ GPR → −0.5 mm |
| `gpr_core.py` | 與 MATLAB `fitrgp`（exact）相同的 GPR 數學 |
| `C1_export_final_gpr.m` | **首選**：MATLAB 用 fitrgp 訓練全部資料，匯出 `gpr_model.json` |
| `fit_gpr.py` | 備案：沒有 MATLAB 時用 Python 重現同一模型 |
| `check_consistency.py` | 驗證 GUI 算的特徵 = 訓練 Excel 的「像素長度」 |
| `examples/` | （自己建立）放去識別化的**整張** X 光，介面一鍵載入 |

## 執行順序
```powershell
python -m pip install -r requirements.txt
# 0. 把 A3 偵測器權重（A4 的 train_crop_yolov11_run/weights_ready.pt）複製進來，改名 detector.pt
# 1. MATLAB 執行 C1_export_final_gpr.m，把 gpr_model.json 複製到這個資料夾
#    （或改用：python fit_gpr.py）
python check_consistency.py
python server.py
```

## 已對齊你們的設定
- top-hat：照抄 `A1_tophat_before_annotation.py`（contrast、K=31、ellipse、STRENGTH=1、只處理非黑邊 ROI），已用程式比對輸出逐像素相同
- GPR：照 `C1_pxToMm_GPR.m`（squaredexponential、basis linear、X 手動標準化）
- `SEG_WEIGHTS = weights_ready.pt`、`LENGTH_SCALE_MODE = "b3_padded"`：沿用 pipeline_v2
- top-hat 在 letterbox 之前做；GPR 用 `mask幾何_長邊正規化` + 年齡（年齡為必填）
- 要不要做長邊正規化：自動讀 `gpr_model.json` 的 METHOD，不會跟訓練資料不一致

## 要確認
| 項目 | 說明 |
|---|---|
| 權重 | `detector.pt` = A3 偵測器；`weights_ready.pt` = top-hat 圖訓練的分割模型 |
| `check_consistency.py` | 填 `CROP_DIR`（A4 裁切後、top-hat 前的 67 張單顆牙圖）與 `FULL_IMAGE_DIR`（整張 X 光） |
