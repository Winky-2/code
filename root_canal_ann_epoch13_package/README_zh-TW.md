# 根管曲線像素轉毫米 ANN（epoch 13）

## 內容

- `data/預測結果與評估指標_像素轉mm.xlsx`：31 筆曲線像素長度與參考長度資料。
- `data/醫生照片.zip`：原始影像資料。
- `code/pixelToMmPredictorANN5Fold.m`：MATLAB 五折訓練與 OOF 評估程式。
- `trained_model/ann_epoch13_trainbr_models.mat`：已訓練的五折網路與完整資料重訓網路。
- `results/ann_5fold_report.json`：各折與整體 OOF 指標、每筆預測值。
- `results/ann_5fold_oof_plot.png`：OOF 預測圖。

## 模型設定

- 模型：MATLAB `fitnet(5, 'trainbr')`
- 隱藏層：5 個神經元
- 最大 epochs：13
- 輸入：根管中心線曲線長度（pixel）
- 輸出：參考工作長度（mm）
- 評估：固定 random seed 42 的 5-fold Out-of-Fold
- 每折標準化：只由該折訓練資料的平均值與標準差計算

## 重新訓練

在 MATLAB 將目前資料夾切換至此套件根目錄後執行：

```matlab
addpath('code');
inputFile = fullfile(pwd, 'data', '預測結果與評估指標_像素轉mm.xlsx');
outputDir = fullfile(pwd, 'rerun_results');
pixelToMmPredictorANN5Fold(13, outputDir, 5, 'trainbr', 0, inputFile);
```

此程式會重新產生 OOF 報告、圖檔與 `.mat` 模型檔。
