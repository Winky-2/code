# 根管工作長度輔助估測系統 — GUI

## 檔案
| 檔案 | 用途 |
|---|---|
| `pipeline.py` | 推論流程：A1 CLAHE → A5 letterbox → YOLO-seg → B3 幾何 → B7 正規化 → 迴歸 −0.5 mm |
| `app.py` | Gradio 介面（單張／批次） |
| `fit_regressor.py` | 產生 `reg_model.json`（線性迴歸，Python） |
| `C1_export_final_ann.m` | 產生 `reg_model.json`（C1 同設定的 ANN，MATLAB）— 二選一 |
| `check_consistency.py` | 驗證 GUI 算的特徵 = 訓練 Excel 的「像素長度」 |

## 第一次設定（照順序）
```bash
pip install -r requirements.txt
```
1. 把 `best.pt` 和 `根管填充物像素長度_已配對_26n_mask幾何_長邊正規化.xlsx` 放進這個資料夾
2. 改 `pipeline.py` 的 TODO 1–4、`check_consistency.py` 的 TODO 6
3. `python fit_regressor.py` → 產生 `reg_model.json`（印出的 CV MAE 應該接近 C1 輸出表的 D 列）
4. `python check_consistency.py` → 照它的建議設定 `LENGTH_SCALE_MODE`
5. `python app.py` → 瀏覽器開 http://127.0.0.1:7860

## TODO 清單
| # | 位置 | 要改什麼 |
|---|---|---|
| 1 | pipeline.py `SEG_WEIGHTS` | 論文用的 seg 權重路徑 |
| 2 | pipeline.py `USE_CLAHE` | test 圖進 A5 前有沒有做 CLAHE（A5 設定裡的資料夾名有 `noCLAHE`，請確認） |
| 3 | pipeline.py `PAD_TB_RATIO` | test 集跑 A5 時的值（預設 0.10） |
| 4 | pipeline.py `LENGTH_SCALE_MODE` | 用 check_consistency.py 決定 |
| 5 | fit_regressor.py `TRAIN_XLSX` | B7 的輸出檔名 |
| 6 | check_consistency.py `RAW_IMAGE_DIR` | 整張原圖資料夾（= B7 的 FULL_IMAGE_DIR） |
| 7 | 迴歸模型 | 要跟論文一樣用 ANN → 在 MATLAB 跑 `C1_export_final_ann.m` 覆寫 `reg_model.json` |
| 8 | app.py `TITLE` | 作品名稱 |

## 已知限制
- **挑牙**：B3 有 GT 時用 gt_iou 挑牙，GUI 沒有 GT，只能挑最靠近影像中心的那顆。多顆牙時畫面會提醒，必須用眼睛確認。
- **模型數字 ≠ 論文數字**：GUI 用全資料訓練的單一模型，論文報的是 k-fold 平均，兩者本來就會不同。
- **B3 的 scale 寫法**：`scale = 原圖長邊 ÷ 640` 沒扣掉 A5 上下補的 10% 黑邊，所以直立影像的「原圖 px」會被低估約 1/1.2。迴歸會吸收這個常數，但寬、高比例不同的影像吸收程度不一樣。這不是 GUI 的問題，論文方法章節可能要描述或修正。
