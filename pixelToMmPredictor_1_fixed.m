function pixelToMmPredictor()

    % ---------------- 資料切分設定區 ----------------
    % 對應流程圖「資料標註，並將資料分成8:2」：不做交叉驗證，改成單純
    % 一次性的 80% train / 20% test，跟整條pipeline其他地方(YOLO資料集)
    % 的切分方式維持一致。
    TRAIN_RATIO = 0.8;
    RANDOM_SEED = 42;   % 固定亂數種子，確保每次重跑切出來的train/test都一樣

    % ---------------- 臨床設定區 ----------------
    % 對應流程圖「轉換實際長度 (長度推估 -0.5mm)」：
    % 影像/像素轉換出來的長度會再扣掉這個臨床安全offset，才是實際建議
    % 使用的工作長度。目前先用 -0.5mm，正式值待醫師確認後在這裡改。
    OFFSET_MM = 0.5;

    % 對應流程圖「臨床與安全性評估 (MAE / 理想比率 / 過長率)」：
    % 「理想比率」= 校正後預測值與實際長度的絕對誤差在容忍範圍內的比例。
    % 容忍值待醫師確認臨床可接受誤差後再調整，先用 ±1.0mm 當預設值。
    IDEAL_TOLERANCE_MM = 1.0;

    %% 1. 讀取 Excel 資料
    filename = "C:\Users\MAOJIE\Desktop\fcu\像素轉公分\根管填充物像素長度.xlsx";
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve'; % 保留原始中文欄位名稱
    data = readtable(filename, opts);

    %% 2. 資料清理與特徵提取
    ids = data.('病歷號');

    % 取出「長度」，去除 'mm' 字樣 (保持為 mm)
    length_str = string(data.('長度'));
    Y = str2double(regexprep(length_str, '[^\d.]', ''));

    % 取出「像素長度」，去除 'px' 字樣
    pixel_str = string(data.('像素長度'));
    X = str2double(regexprep(pixel_str, '[^\d.]', ''));

    % 過濾掉可能的 NaN
    valid_idx = ~isnan(X) & ~isnan(Y);
    X = X(valid_idx);
    Y = Y(valid_idx);
    ids = ids(valid_idx);
    n = length(Y);

    %% 3. 資料標準化
    [Xn, ~, ~] = zscore(X);
    Yn = Y;  % 輸出的長度不標準化，維持 mm 單位

    %% 4. 切分 80% train / 20% test（單次切分，不做交叉驗證）
    rng(RANDOM_SEED);
    c = cvpartition(n, 'HoldOut', 1 - TRAIN_RATIO);
    trainIdx = training(c);
    testIdx  = test(c);

    n_train = sum(trainIdx);
    n_test  = sum(testIdx);
    fprintf('=== 資料切分：train %d 筆 (%.0f%%) / test %d 筆 (%.0f%%) ===\n', ...
        n_train, 100 * n_train / n, n_test, 100 * n_test / n);
    if n_test < 10
        fprintf('⚠️ test 只有 %d 筆，樣本數偏小，以下指標(尤其理想比率/過長率)僅供參考，\n', n_test);
        fprintf('   單一筆的對錯就會讓百分比大幅跳動，解讀時請留意這一點。\n');
    end

    %% 5. 訓練模型（只用 train，test 完全不參與訓練）
    net = fitnet(5, 'trainbr');
    net.trainParam.showWindow = false;
    net.trainParam.epochs     = 200;

    net.divideFcn            = 'divideind';
    net.divideParam.trainInd = find(trainIdx);
    net.divideParam.valInd   = [];
    net.divideParam.testInd  = find(testIdx);

    net = train(net, Xn', Yn');

    %% 6. 在 test set 上做唯一一次評估（held-out，不會拿來調參）
    Y_test  = Y(testIdx);
    Yp_test = net(Xn(testIdx)')';

    test_residuals = Yp_test - Y_test;
    test_abs_error = abs(test_residuals);

    test_mae  = mean(test_abs_error);
    test_rmse = sqrt(mean(test_residuals.^2));
    test_bias = mean(test_residuals);
    if n_test >= 2
        [R_test, ~] = corr(Y_test, Yp_test);
        test_r2 = R_test^2;
    else
        test_r2 = NaN;
    end

    fprintf('[Test set] RMSE: %.3f | MAE: %.3f | R²: %.3f | Bias: %.3f\n', ...
        test_rmse, test_mae, test_r2, test_bias);

    % --- 6b. 套用臨床offset，算出「實際建議工作長度」，這是流程圖裡
    %         真正會拿去跟醫師臨床判斷比較的數字（不是原始迴歸輸出）---
    Yp_test_offset      = Yp_test - OFFSET_MM;
    test_offset_residuals = Yp_test_offset - Y_test;
    test_offset_abs_error = abs(test_offset_residuals);

    test_offset_mae  = mean(test_offset_abs_error);
    test_offset_bias = mean(test_offset_residuals);

    % 理想比率：絕對誤差落在容忍範圍內的比例
    test_ideal_rate = mean(test_offset_abs_error <= IDEAL_TOLERANCE_MM) * 100;

    % 過長率：offset後的預測值仍然「比實際長度長」的比例
    % （臨床上高估工作長度風險較大，優先於低估，需要單獨追蹤）
    test_overest_rate = mean(test_offset_residuals > 0) * 100;

    %% 7. 用同一個模型預測全部資料（含train，只作參考，不能當成model表現指標）
    Y_pred_all    = net(Xn')';
    all_residuals = Y_pred_all - Y;
    all_abs_error = abs(all_residuals);

    all_mae  = mean(all_abs_error);
    all_rmse = sqrt(mean(all_residuals.^2));
    all_bias = mean(all_residuals);
    [R_all, ~] = corr(Y, Y_pred_all);
    all_r2   = R_all^2;

    Y_pred_all_offset    = Y_pred_all - OFFSET_MM;
    all_offset_residuals = Y_pred_all_offset - Y;
    all_offset_abs_error = abs(all_offset_residuals);

    all_offset_mae     = mean(all_offset_abs_error);
    all_offset_bias    = mean(all_offset_residuals);
    all_ideal_rate      = mean(all_offset_abs_error <= IDEAL_TOLERANCE_MM) * 100;
    all_overest_rate    = mean(all_offset_residuals > 0) * 100;

    %% 8. 整理預測結果，準備輸出到 Excel
    split_label = repmat("train", n, 1);
    split_label(testIdx) = "test";

    is_overest_all = all_offset_residuals > 0;

    ResultsTable = table(ids, split_label, X, Y, ...
        Y_pred_all, all_residuals, all_abs_error, ...
        Y_pred_all_offset, all_offset_residuals, all_offset_abs_error, is_overest_all, ...
        'VariableNames', {'病歷號', 'Train_or_Test', '像素長度_px', '實際長度_mm', ...
        '模型預測_mm', '誤差Bias_mm', '絕對誤差_mm', ...
        '校正後預測_mm', '校正後誤差Bias_mm', '校正後絕對誤差_mm', '是否過長'});

    MetricNames = {
        sprintf('A_Test set表現 (held-out %d筆，唯一一次的嚴格指標)', n_test);
        'B_全部資料表現 (含train，僅供參考，不可當model實際能力)';
        '--- 分界線 ---';
        sprintf('C_臨床安全性評估_Test set校正後 (offset=-%.2fmm)', OFFSET_MM);
        sprintf('D_臨床安全性評估_全部資料校正後 (offset=-%.2fmm，僅供參考)', OFFSET_MM)
    };

    MAE_val     = round([test_mae;  all_mae;  NaN; test_offset_mae;  all_offset_mae],  3);
    RMSE_val    = round([test_rmse; all_rmse; NaN; NaN;              NaN],             3);
    R2_val      = round([test_r2;   all_r2;   NaN; NaN;              NaN],             3);
    Bias_val    = round([test_bias; all_bias; NaN; test_offset_bias; all_offset_bias], 3);
    IdealRate_pct   = round([NaN; NaN; NaN; test_ideal_rate;   all_ideal_rate],   1);
    OverestRate_pct = round([NaN; NaN; NaN; test_overest_rate; all_overest_rate], 1);

    MetricsTable = table(MetricNames, MAE_val, RMSE_val, R2_val, Bias_val, IdealRate_pct, OverestRate_pct, ...
        'VariableNames', {'評估範圍_與_嚴格程度', 'MAE_mm', 'RMSE_mm', 'R_Square', 'Mean_Bias_mm', ...
        '理想比率_pct', '過長率_pct'});

    %% 9. 匯出至 Excel 檔案
    output_filename = '預測結果與評估指標_train80_test20_像素轉mm.xlsx';

    writetable(ResultsTable, output_filename, 'Sheet', '所有牙齒預測結果');
    writetable(MetricsTable, output_filename, 'Sheet', '模型評估指標');

    fprintf('\n✅ 所有結果已成功匯出至 Excel 檔案：「 %s 」\n\n', output_filename);
    disp('=== 最終模型評估指標 (已存入 Excel 第二工作表) ===');
    disp(MetricsTable);

    fprintf('\n=== 臨床與安全性評估 (對應流程圖最後一步) ===\n');
    fprintf('offset = -%.2f mm，理想誤差容忍 = ±%.2f mm（兩者皆為預設值，需醫師確認後調整）\n', OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('[Test set，唯一嚴格指標]  MAE(校正後): %.3f mm | 理想比率: %.1f%% | 過長率: %.1f%%\n', ...
        test_offset_mae, test_ideal_rate, test_overest_rate);
    fprintf('⚠️ 過長率是臨床風險較高的指標（預測工作長度仍比實際長 -> 有超出根尖的風險）。\n');
    fprintf('⚠️ 請以 Test set 的數字為準，全部資料(含train)的數字只是參考，不能拿來當model實際表現。\n');

    % 註：畫圖已統一移到 2_plotMyResults.m，避免跟該檔重複維護同一份繪圖程式碼。
    % 這支腳本只負責訓練 + 匯出 Excel，匯出後請執行 2_plotMyResults.m 看圖，
    % 它會自動讀取這裡輸出的 output_filename，不需要手動改檔名或抄指標數字。
    fprintf('👉 接下來請執行 2_plotMyResults.m 產生圖表\n');

end
