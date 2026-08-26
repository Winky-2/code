function plotMyResults()

    %% 1. 設定檔案路徑
    % 直接對接 1_pixelToMmPredictor.m 實際輸出的檔名，不用手動改成「最後版.xlsx」，
    % 兩支腳本改了檔名/指標都會自動同步，不用再抄一次。
    filename = '預測結果與評估指標_train80_test20_像素轉mm.xlsx';
    fprintf('正在從 %s 讀取預測資料...\n', filename);

    %% 2. 讀取工作表 1：所有牙齒預測結果，只取 test set 來畫圖
    % (train 的預測不能拿來評估model表現，因為模型已經看過這些資料)
    opts_res = detectImportOptions(filename);
    opts_res.Sheet = '所有牙齒預測結果';
    opts_res.VariableNamingRule = 'preserve';
    data_res = readtable(filename, opts_res);

    is_test = strcmp(string(data_res.('Train_or_Test')), 'test');
    data_test = data_res(is_test, :);

    Y                 = data_test.('實際長度_mm');
    Y_pred            = data_test.('模型預測_mm');
    Y_pred_offset     = data_test.('校正後預測_mm');
    is_overest        = logical(data_test.('是否過長'));
    n = length(Y);

    fprintf('圖表只使用 test set（%d 筆），train set 的預測不列入評估用圖。\n', n);
    if n < 10
        fprintf('⚠️ test set 樣本數偏小 (%d 筆)，圖表趨勢僅供參考。\n', n);
    end

    %% 3. 自動從「模型評估指標」工作表讀取 test set 指標
    % 不再手動打字輸入，避免模型重跑後數字沒同步更新、圖表跟實際結果對不上。
    opts_metrics = detectImportOptions(filename);
    opts_metrics.Sheet = '模型評估指標';
    opts_metrics.VariableNamingRule = 'preserve';
    data_metrics = readtable(filename, opts_metrics);

    row_label = '評估範圍_與_嚴格程度';

    % A列：test set 原始（未校正）指標
    is_test_row = contains(string(data_metrics.(row_label)), 'A_Test set表現');
    test_row = data_metrics(is_test_row, :);
    if height(test_row) ~= 1
        error('在「模型評估指標」工作表裡找不到唯一的 Test set 那一列，請確認 1_pixelToMmPredictor.m 有沒有改過 MetricNames 的文字');
    end
    test_mae  = test_row.('MAE_mm');      if iscell(test_mae);  test_mae  = str2double(string(test_mae));  end
    test_rmse = test_row.('RMSE_mm');     if iscell(test_rmse); test_rmse = str2double(string(test_rmse)); end
    test_r2   = test_row.('R_Square');    if iscell(test_r2);   test_r2   = str2double(string(test_r2));   end
    test_bias = test_row.('Mean_Bias_mm');if iscell(test_bias); test_bias = str2double(string(test_bias)); end

    % C列：test set 校正後（臨床安全性）指標
    is_clinical_row = contains(string(data_metrics.(row_label)), 'C_臨床安全性評估_Test set校正後');
    clinical_row = data_metrics(is_clinical_row, :);
    if height(clinical_row) ~= 1
        error('在「模型評估指標」工作表裡找不到唯一的臨床安全性(C列)，請確認 1_pixelToMmPredictor.m 有沒有改過 MetricNames 的文字');
    end
    clinical_mae     = clinical_row.('MAE_mm');      if iscell(clinical_mae);     clinical_mae     = str2double(string(clinical_mae));     end
    clinical_bias    = clinical_row.('Mean_Bias_mm');if iscell(clinical_bias);    clinical_bias    = str2double(string(clinical_bias));    end
    clinical_ideal   = clinical_row.('理想比率_pct'); if iscell(clinical_ideal);   clinical_ideal   = str2double(string(clinical_ideal));   end
    clinical_overest = clinical_row.('過長率_pct');   if iscell(clinical_overest); clinical_overest = str2double(string(clinical_overest)); end

    fprintf('資料讀取完畢！正在繪製圖表...\n');

    %% 4. 繪製圖表 1：實際長度 vs 預測長度 曲線對比圖 (test set)
    figure('Name','Curve Chart: Actual vs Predicted (Test Set)','NumberTitle','off');
    hold on;

    [Y_sorted, sort_idx] = sort(Y);
    Y_pred_sorted = Y_pred(sort_idx);

    plot(1:n, Y_sorted, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '實際長度 (Actual)');
    plot(1:n, Y_pred_sorted, 'b-*', 'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', 'Test set 預測長度 (Predicted)');

    xlabel('樣本排序編號 (依實際長度由小到大，僅test set)');
    ylabel('長度 (mm)');
    title('實際長度 vs 預測長度 曲線對比圖 (Test Set)');
    legend('Location', 'northwest');
    grid on;
    hold off;

    %% 5. 繪製圖表 2：Test set 預測 vs 實際值 (結合誤差容忍線與評估指標)
    figure('Name','Test Set: Actual vs Predicted with Error Margins','NumberTitle','off');
    hold on;

    scatter(Y, Y_pred, 40, 'b', 'filled', 'DisplayName', '預測結果 (Test Set)');

    min_val = floor(min([Y; Y_pred])) - 1;
    max_val = ceil(max([Y; Y_pred])) + 1;

    plot([min_val, max_val], [min_val, max_val], 'w-', 'LineWidth', 2, 'DisplayName', '完美預測線 (誤差 0)');

    plot([min_val, max_val], [min_val+0.5, max_val+0.5], 'r--', 'LineWidth', 1.5, 'DisplayName', '+0.5 mm 誤差線');
    plot([min_val, max_val], [min_val-0.5, max_val-0.5], 'r--', 'LineWidth', 1.5, 'DisplayName', '-0.5 mm 誤差線');

    plot([min_val, max_val], [min_val+1.0, max_val+1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '+1.0 mm 誤差線');
    plot([min_val, max_val], [min_val-1.0, max_val-1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '-1.0 mm 誤差線');

    xlabel('實際長度 (mm)');
    ylabel('預測長度 (mm)');
    title('實際長度 vs 預測長度 (Test Set，搭配誤差容忍區間)');
    legend('Location','southeast');
    grid on;

    metric_text = sprintf('【Test Set 模型評估指標】\nRMSE : %.3f mm\nMAE : %.3f mm\nR^2 : %.3f\nMean Bias : %.3f mm\n(n = %d)', ...
                           test_rmse, test_mae, test_r2, test_bias, n);

    x_lims = xlim;
    y_lims = ylim;
    text(x_lims(1) + 0.05*(x_lims(2)-x_lims(1)), ...
         y_lims(2) - 0.18*(y_lims(2)-y_lims(1)), ...
         metric_text, 'FontSize', 11, ...
         'BackgroundColor', 'k', ...  % 👈 'w' 改成 'k' (黑色背景)
         'EdgeColor', 'w', ...        % 👈 'k' 改成 'w' (白色外框，才不會在黑底隱形)
         'Color', 'w');               % 👈 補上這行，將文字顏色強制改成白色
    hold off;

    %% 6. 繪製圖表 3：校正後 (實際建議工作長度) vs 實際值 — 臨床安全性圖
    % 對應流程圖「轉換實際長度(-0.5mm)」+「臨床與安全性評估」這兩步，
    % 跟圖表2的差別是：這裡的預測值已經扣掉臨床offset，且用顏色標出
    % 「過長」(臨床風險較高) 的點，方便跟醫師討論。
    figure('Name','Clinical Safety: Offset-Adjusted Working Length (Test Set)','NumberTitle','off');
    hold on;

    scatter(Y(~is_overest), Y_pred_offset(~is_overest), 45, 'b', 'filled', ...
        'DisplayName', '校正後預測 (在容忍範圍內或偏短)');
    scatter(Y(is_overest), Y_pred_offset(is_overest), 45, 'r', 'filled', ...
        'Marker', '^', 'DisplayName', '校正後預測仍過長 (臨床風險)');

    min_val2 = floor(min([Y; Y_pred_offset])) - 1;
    max_val2 = ceil(max([Y; Y_pred_offset])) + 1;

    plot([min_val2, max_val2], [min_val2, max_val2], 'w-', 'LineWidth', 2, 'DisplayName', '完美預測線 (誤差 0)');
    plot([min_val2, max_val2], [min_val2+1.0, max_val2+1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '+1.0 mm 容忍線');
    plot([min_val2, max_val2], [min_val2-1.0, max_val2-1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '-1.0 mm 容忍線');

    xlabel('實際長度 (mm)');
    ylabel('校正後預測長度 = 迴歸輸出 - offset (mm)');
    title('臨床建議工作長度 vs 實際長度 (Test Set，校正後，含過長標示)');
    legend('Location', 'southeast');
    grid on;

    clinical_text = sprintf(['【臨床與安全性評估 (校正後, Test Set)】\nMAE : %.3f mm\nMean Bias : %.3f mm\n' ...
                              '理想比率 : %.1f%%\n過長率 : %.1f%%\n(n = %d)'], ...
                             clinical_mae, clinical_bias, clinical_ideal, clinical_overest, n);

    x_lims2 = xlim;
    y_lims2 = ylim;
    text(x_lims2(1) + 0.05*(x_lims2(2)-x_lims2(1)), ...
         y_lims2(2) - 0.22*(y_lims2(2)-y_lims2(1)), ...
         clinical_text, 'FontSize', 11, 'BackgroundColor', 'k', 'EdgeColor', 'w');
    hold off;

    fprintf('✅ 三張圖表已繪製完成（皆只用 test set，含臨床安全性評估圖）！\n');

end
