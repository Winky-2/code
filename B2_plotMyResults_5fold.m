function plotMyResults5Fold()
% B_plotMyResults_5fold.m
% ============================================================
% 對應流程圖階段：結果視覺化 / 臨床與安全性評估
% *** 5-fold 交叉驗證版本 ***
%
% *** 跟舊版 B_plotMyResults_fixed.m 的差異 ***
% 1. 舊版只畫 test set（4 顆牙），這一版畫的是全部 20 顆牙的
%    out-of-fold 預測——每顆牙都是「沒看過它的模型」預測出來的，
%    所以全部都可以拿來評估，樣本數直接變 5 倍，圖表才有意義。
% 2. 圖上的指標數字改成「用 computeMetrics.m 從預測值重算」，
%    不再從 Excel 抄數字。舊版是讀 Excel 的指標欄位，一旦
%    A_ 那邊改過算法而沒重跑、或有人手動編輯過 Excel，
%    圖上的數字就會跟散佈圖對不起來，而且看起來完全正常。
%    現在圖上的數字必然跟圖上的點一致。
%    另外仍會讀 Excel 的指標表做交叉檢查，兩邊對不上會明確示警。
% 3. offset / 容忍值改成從 Excel 的「設定」工作表讀，
%    不在這支腳本裡再打一次，避免兩邊參數不同步。
% 4. 新增第 4 張圖：各 fold 的 MAE 長條圖，用來看有沒有某一折特別離譜。
%
% 執行前請確認：
% - computeMetrics.m 跟這支放在同一個資料夾
% - A_pixelToMmPredictor_5fold.m 已跑過，產生了 filename 指定的 Excel

    %% 1. 設定
    filename = '預測結果與評估指標_5fold_像素轉mm.xlsx';

    % 完美預測線用中性灰，深色/淺色主題下都看得見。
    % （舊版寫死 'w-' 白線，在預設白底 figure 上會整條隱形。）
    LINE_COLOR = [0.5 0.5 0.5];

    if exist('computeMetrics', 'file') ~= 2
        error(['找不到 computeMetrics.m，請把它跟這支腳本放在同一個資料夾。' ...
               '圖上的指標必須跟 A_ 用同一份算法，才不會出現「圖跟數字對不上」。']);
    end
    if exist(filename, 'file') ~= 2
        error('找不到 %s，請先執行 A_pixelToMmPredictor_5fold.m。', filename);
    end
    fprintf('正在從 %s 讀取預測資料...\n', filename);

    %% 2. 讀取預測結果（全部都是 out-of-fold，不需要再篩選 train/test）
    opts_res = detectImportOptions(filename);
    opts_res.Sheet = '所有牙齒預測結果';
    opts_res.VariableNamingRule = 'preserve';
    data_res = readtable(filename, opts_res);

    Y      = data_res.('實際長度_mm');
    Y_pred = data_res.('OOF預測_mm');
    fold   = data_res.('Fold');

    valid  = ~isnan(Y) & ~isnan(Y_pred);
    Y      = Y(valid);
    Y_pred = Y_pred(valid);
    fold   = fold(valid);
    n      = numel(Y);

    fprintf('圖表使用全部 %d 顆牙的 out-of-fold 預測（每顆牙都由沒看過它的模型預測）。\n', n);
    if n < 10
        fprintf('⚠️ 有效樣本只有 %d 顆，圖表趨勢僅供參考。\n', n);
    end

    %% 3. 讀設定（offset / 容忍值），再用 computeMetrics 重算指標
    [OFFSET_MM, IDEAL_TOLERANCE_MM] = readSettings(filename);
    fprintf('臨床參數：offset = -%.2f mm，理想容忍 = ±%.2f mm（讀自 Excel 的「設定」工作表）\n', ...
        OFFSET_MM, IDEAL_TOLERANCE_MM);

    m = computeMetrics(Y, Y_pred, OFFSET_MM, IDEAL_TOLERANCE_MM);
    Y_pred_offset = m.PredOffset;
    is_overest    = m.IsOverest;

    % 交叉檢查：重算的結果應該跟 A_ 寫進 Excel 的 A_OOF_POOLED 那列一致
    crossCheckAgainstExcel(filename, m);

    fprintf('資料讀取完畢！正在繪製圖表...\n');

    %% 4. 圖表 1：實際長度 vs OOF 預測長度 曲線對比
    figure('Name','Curve: Actual vs Out-of-Fold Predicted','NumberTitle','off');
    hold on;

    [Y_sorted, sort_idx] = sort(Y);
    Y_pred_sorted = Y_pred(sort_idx);

    plot(1:n, Y_sorted, 'r-o', 'LineWidth', 1.5, 'MarkerSize', 4, 'DisplayName', '實際長度 (Actual)');
    plot(1:n, Y_pred_sorted, 'b-*', 'LineWidth', 1.2, 'MarkerSize', 4, 'DisplayName', 'OOF 預測長度 (Predicted)');

    xlabel('樣本排序編號（依實際長度由小到大，全部牙齒）');
    ylabel('長度 (mm)');
    title(sprintf('實際長度 vs OOF 預測長度 曲線對比圖 (5-fold, n=%d)', n));
    legend('Location', 'northwest');
    grid on;
    hold off;

    %% 5. 圖表 2：OOF 預測 vs 實際值（依 fold 上色 + 誤差容忍線 + 指標）
    figure('Name','Out-of-Fold: Actual vs Predicted with Error Margins','NumberTitle','off');
    hold on;

    foldList = unique(fold(:))';
    cmap = lines(max(numel(foldList), 1));
    for i = 1:numel(foldList)
        k = foldList(i);
        sel = fold == k;
        scatter(Y(sel), Y_pred(sel), 45, cmap(i,:), 'filled', ...
            'DisplayName', sprintf('fold%d (n=%d)', k, sum(sel)));
    end

    min_val = floor(min([Y; Y_pred])) - 1;
    max_val = ceil(max([Y; Y_pred])) + 1;

    plot([min_val, max_val], [min_val, max_val], '-', 'Color', LINE_COLOR, ...
        'LineWidth', 2, 'DisplayName', '完美預測線 (誤差 0)');
    plot([min_val, max_val], [min_val+0.5, max_val+0.5], 'r--', 'LineWidth', 1.5, 'DisplayName', '+0.5 mm 誤差線');
    plot([min_val, max_val], [min_val-0.5, max_val-0.5], 'r--', 'LineWidth', 1.5, 'HandleVisibility', 'off');
    plot([min_val, max_val], [min_val+1.0, max_val+1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '±1.0 mm 誤差線');
    plot([min_val, max_val], [min_val-1.0, max_val-1.0], 'g:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

    xlabel('實際長度 (mm)');
    ylabel('OOF 預測長度 (mm)');
    title('實際長度 vs OOF 預測長度 (5-fold，搭配誤差容忍區間)');
    legend('Location','southeast');
    grid on;

    metric_text = sprintf(['【5-fold OOF 指標】\nRMSE : %.3f mm\nMAE : %.3f mm\n' ...
                           'R^2 : %.3f\nPearson r^2 : %.3f\nMean Bias : %.3f mm\n(n = %d)'], ...
                           m.RMSE, m.MAE, m.R2, m.PearsonR2, m.Bias, m.n);
    placeMetricBox(metric_text, 0.18);
    hold off;

    %% 6. 圖表 3：校正後（實際建議工作長度）vs 實際值 — 臨床安全性圖
    % 對應流程圖「轉換實際長度(-0.5mm)」+「臨床與安全性評估」，
    % 跟圖表 2 的差別是預測值已扣掉臨床 offset，並用顏色標出「過長」的點。
    figure('Name','Clinical Safety: Offset-Adjusted Working Length (Out-of-Fold)','NumberTitle','off');
    hold on;

    scatter(Y(~is_overest), Y_pred_offset(~is_overest), 45, 'b', 'filled', ...
        'DisplayName', '校正後預測（在容忍範圍內或偏短）');
    scatter(Y(is_overest), Y_pred_offset(is_overest), 45, 'r', 'filled', ...
        'Marker', '^', 'DisplayName', '校正後預測仍過長（臨床風險）');

    min_val2 = floor(min([Y; Y_pred_offset])) - 1;
    max_val2 = ceil(max([Y; Y_pred_offset])) + 1;

    plot([min_val2, max_val2], [min_val2, max_val2], '-', 'Color', LINE_COLOR, ...
        'LineWidth', 2, 'DisplayName', '完美預測線 (誤差 0)');
    plot([min_val2, max_val2], [min_val2+IDEAL_TOLERANCE_MM, max_val2+IDEAL_TOLERANCE_MM], ...
        'g:', 'LineWidth', 1.5, 'DisplayName', sprintf('±%.1f mm 容忍線', IDEAL_TOLERANCE_MM));
    plot([min_val2, max_val2], [min_val2-IDEAL_TOLERANCE_MM, max_val2-IDEAL_TOLERANCE_MM], ...
        'g:', 'LineWidth', 1.5, 'HandleVisibility', 'off');

    xlabel('實際長度 (mm)');
    ylabel('校正後預測長度 = 模型輸出 - offset (mm)');
    title(sprintf('臨床建議工作長度 vs 實際長度 (5-fold OOF，offset = -%.2f mm)', OFFSET_MM));
    legend('Location', 'southeast');
    grid on;

    clinical_text = sprintf(['【臨床與安全性評估（校正後，5-fold OOF）】\nMAE : %.3f mm\n' ...
                             'Mean Bias : %.3f mm\n理想比率 : %.1f%%\n過長率 : %.1f%%\n(n = %d)'], ...
                             m.MAE_offset, m.Bias_offset, m.IdealRate, m.OverestRate, m.n);
    placeMetricBox(clinical_text, 0.22);
    hold off;

    %% 7. 圖表 4：各 fold 的 MAE（看穩定度）
    % 每折只有 3~4 顆牙，數字本來就會抖；這張圖是要看有沒有「某一折特別離譜」，
    % 如果有，通常代表那一折剛好分到幾顆難牙（重疊、彎曲、YOLO 抓不準）。
    figure('Name','Per-Fold MAE (Stability Check)','NumberTitle','off');
    hold on;

    foldMAE = nan(numel(foldList), 1);
    foldN   = zeros(numel(foldList), 1);
    for i = 1:numel(foldList)
        sel = fold == foldList(i);
        fm = computeMetrics(Y(sel), Y_pred(sel), OFFSET_MM, IDEAL_TOLERANCE_MM);
        foldMAE(i) = fm.MAE;
        foldN(i)   = fm.n;
    end

    bar(1:numel(foldList), foldMAE, 0.6, 'FaceColor', [0.3 0.6 0.9], 'DisplayName', '各 fold MAE');
    plot([0.4, numel(foldList)+0.6], [m.MAE, m.MAE], 'r--', 'LineWidth', 1.8, ...
        'DisplayName', sprintf('彙總 OOF MAE = %.3f mm', m.MAE));

    for i = 1:numel(foldList)
        text(i, foldMAE(i), sprintf('  %.3f\n  (n=%d)', foldMAE(i), foldN(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 9);
    end

    set(gca, 'XTick', 1:numel(foldList), ...
        'XTickLabel', arrayfun(@(k) sprintf('fold%d', k), foldList, 'UniformOutput', false));
    xlim([0.4, numel(foldList)+0.6]);
    ylabel('MAE (mm)');
    title('各 fold 的 MAE 與彙總 OOF MAE 比較（穩定度檢查）');
    legend('Location', 'best');
    grid on;
    hold off;

    fprintf('✅ 四張圖表已繪製完成（全部使用 out-of-fold 預測，含臨床安全性與穩定度檢查）！\n');

end


% ============================================================
% 以下為輔助函式
% ============================================================

function [offsetMm, tolMm] = readSettings(filename)
% 從 A_ 寫出的「設定」工作表讀臨床參數。
% 讀不到就退回預設值並示警——但預設值不保證跟 A_ 當時用的一樣，
% 所以一定要提醒使用者去確認，不然圖上的臨床指標會是錯的。
    offsetMm = 0.5;
    tolMm    = 1.0;
    try
        opts = detectImportOptions(filename);
        opts.Sheet = '設定';
        opts.VariableNamingRule = 'preserve';
        t = readtable(filename, opts);
        names = string(t.('參數'));
        vals  = t.('值');
        if ~isnumeric(vals)
            vals = str2double(string(vals));
        end
        idx = find(names == "OFFSET_MM", 1);
        if ~isempty(idx), offsetMm = vals(idx); end
        idx = find(names == "IDEAL_TOLERANCE_MM", 1);
        if ~isempty(idx), tolMm = vals(idx); end
    catch
        warning(['讀不到「設定」工作表，暫時使用預設值 offset=%.2f / 容忍=%.2f。' ...
                 '請確認 A_pixelToMmPredictor_5fold.m 是最新版（會輸出設定表），' ...
                 '否則圖上的臨床指標可能跟 A_ 實際使用的參數不同。'], offsetMm, tolMm);
    end
end


function crossCheckAgainstExcel(filename, m)
% 把「這裡重算的指標」跟「A_ 寫進 Excel 的 A_OOF_POOLED 那列」比對。
% 兩邊都走 computeMetrics，正常情況只會差在四捨五入，
% 差太多就代表 Excel 裡的預測值或指標被動過、或 A_ 沒重跑。
    try
        opts = detectImportOptions(filename);
        opts.Sheet = '模型評估指標';
        opts.VariableNamingRule = 'preserve';
        t = readtable(filename, opts);

        row = t(string(t.('Key')) == "A_OOF_POOLED", :);
        if height(row) ~= 1
            warning('在「模型評估指標」工作表裡找不到唯一的 A_OOF_POOLED 那列，略過交叉檢查。');
            return;
        end

        checks = { ...
            'MAE_mm',   m.MAE,   0.001; ...
            'RMSE_mm',  m.RMSE,  0.001; ...
            'R_Square', m.R2,    0.001; ...
            'Mean_Bias_mm', m.Bias, 0.001};

        bad = {};
        for i = 1:size(checks, 1)
            colName = checks{i,1};
            if ~ismember(colName, t.Properties.VariableNames), continue; end
            excelVal = row.(colName);
            if iscell(excelVal), excelVal = str2double(string(excelVal)); end
            if isnan(excelVal) && isnan(checks{i,2}), continue; end
            % Excel 裡是四捨五入到小數第 3 位，容忍半個單位再加一點餘裕
            if abs(excelVal - checks{i,2}) > checks{i,3}
                bad{end+1} = sprintf('%s: Excel=%.4f vs 重算=%.4f', colName, excelVal, checks{i,2}); %#ok<AGROW>
            end
        end

        if isempty(bad)
            fprintf('🧪 交叉檢查通過：圖上的指標與 Excel 的 A_OOF_POOLED 一致。\n');
        else
            warning(['圖上重算的指標跟 Excel 記錄的對不上：\n   %s\n' ...
                     '通常代表 Excel 被手動編輯過，或 A_pixelToMmPredictor_5fold.m ' ...
                     '改過之後沒有重跑。圖上顯示的是「從預測值重算」的版本，' ...
                     '跟圖上的點一定一致；建議重跑 A_ 讓 Excel 同步。'], strjoin(bad, '\n   '));
        end
    catch ME
        warning('交叉檢查失敗（%s），略過。', ME.message);
    end
end


function placeMetricBox(txt, yOffsetRatio)
% 把指標文字框放在左上角。用黑底白字，跟原本的樣式一致。
    x_lims = xlim;
    y_lims = ylim;
    text(x_lims(1) + 0.05*(x_lims(2)-x_lims(1)), ...
         y_lims(2) - yOffsetRatio*(y_lims(2)-y_lims(1)), ...
         txt, 'FontSize', 11, ...
         'BackgroundColor', 'k', ...
         'EdgeColor', 'w', ...
         'Color', 'w');
end
