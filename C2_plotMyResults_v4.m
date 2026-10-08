function C2_plotMyResults_v5()
%% C2_plotMyResults_v5.m
% ============================================================
% 搭配 C1_pixelToMmPredictor_v6.m。以 v3 為底，只做三件事：
%   (1) 圖 3 的文字框改成校正後那組：MAE / RMSE / Bias / R / R^2 / 理想比率
%   (2) 拿掉「過長率」，圖 3 不再把過長樣本標成紅色三角
%   (3) 不再讀「是否過長」欄(C1_v6 已不輸出)
% 其餘(排序點圖、散布圖、年齡/性別誤差分析)與 v3 相同。
% ============================================================

    %% 1. 設定(必須跟 C1 跑的那次一致)
    METHOD  = 'mask幾何_長邊正規化';   % 'mask幾何' | '冠寬比例尺'
    USE_AGE = false;
    USE_SEX = true;
    HIDDEN_SIZE = [3 3];
    SHOW_ERROR_BARS = false;

    filename = sprintf('預測結果與評估指標_kfold_%s%s_H%s.xlsx', METHOD, featTag(USE_AGE, USE_SEX), strrep(mat2str(HIDDEN_SIZE),' ','-'));
    if ~isfile(filename)
        error(['找不到 %s\n' ...
               '   請先用同樣的 METHOD / USE_AGE / USE_SEX 跑一次 C1_v6。'], filename);
    end
    fprintf('正在從 %s 讀取預測資料...\n', filename);

    %% 2. 讀取逐筆預測結果
    opts_res = detectImportOptions(filename, 'Sheet', '所有牙齒預測結果');
    opts_res.VariableNamingRule = 'preserve';
    if ismember('性別', opts_res.VariableNames)
        opts_res = setvartype(opts_res, '性別', 'string');
    end
    data_res = readtable(filename, opts_res);

    is_test = strcmp(string(data_res.('Train_or_Test')), 'test');
    data_test = data_res(is_test, :);

    Y             = data_test.('實際長度_mm');
    Y_pred        = data_test.('模型預測_mm');
    Y_pred_offset = data_test.('校正後預測_mm');
    n = length(Y);

    vn = data_test.Properties.VariableNames;
    hasDemo = ismember('年齡', vn) && ismember('性別', vn);
    if hasDemo
        age    = double(data_test.('年齡'));
        sexLbl = upper(strtrim(string(data_test.('性別'))));
        isM    = sexLbl == "M";
    end

    fprintf('圖表使用 %d 筆 out-of-fold 預測。\n', n);

    %% 3. 讀取評估指標與模型設定
    opts_metrics = detectImportOptions(filename, 'Sheet', '模型評估指標');
    opts_metrics.VariableNamingRule = 'preserve';
    data_metrics = readtable(filename, opts_metrics);
    labels = string(data_metrics.('評估範圍_與_嚴格程度'));

    test_row     = pickRow(data_metrics, labels, 'A_Test set表現');
    clinical_row = pickRow(data_metrics, labels, 'C_臨床安全性評估_Test set校正後');
    sd_row       = pickRowOptional(data_metrics, labels, 'B_重複間標準差');
    lin_row      = pickRowOptional(data_metrics, labels, 'E_線性迴歸對照組_校正後');
    if isempty(lin_row)
        lin_row  = pickRowOptional(data_metrics, labels, 'D_線性迴歸對照組');
    end

    % 未校正(圖 2 用)
    test_mae  = getNum(test_row, 'MAE_mm');
    test_rmse = getNum(test_row, 'RMSE_mm');
    test_r    = getNum(test_row, 'R_Pearson');
    test_r2   = getNum(test_row, 'R_Square');
    test_bias = getNum(test_row, 'Mean_Bias_mm');

    % 校正後(圖 3、報告用)
    clinical_mae   = getNum(clinical_row, 'MAE_mm');
    clinical_rmse  = getNum(clinical_row, 'RMSE_mm');
    clinical_bias  = getNum(clinical_row, 'Mean_Bias_mm');
    clinical_r     = getNum(clinical_row, 'R_Pearson');
    clinical_r2    = getNum(clinical_row, 'R_Square');
    clinical_ideal = getNum(clinical_row, '理想比率_pct');

    sd_mae = NaN; sd_ideal = NaN;
    if ~isempty(sd_row)
        sd_mae   = getNum(sd_row, 'MAE_mm');
        sd_ideal = getNum(sd_row, '理想比率_pct');
    end
    lin_mae = NaN;
    if ~isempty(lin_row)
        lin_mae = getNum(lin_row, 'MAE_mm');
    end

    featDesc = '像素長度';
    if any(sheetnames(filename) == "模型設定")
        st = readtable(filename, 'Sheet', '模型設定', 'VariableNamingRule', 'preserve', ...
                       'TextType', 'string');
        hit = string(st.('項目')) == "輸入特徵";
        if any(hit), featDesc = char(string(st.('值')(find(hit, 1)))); end
    end

    fprintf('資料讀取完畢！正在繪製圖表...\n');
    ttl = sprintf('%s，Out-of-Fold', METHOD);

    %% 4. 圖表 1：排序點圖
    figure('Name', ['Sorted: Actual vs Predicted (OOF) - ' METHOD], 'NumberTitle', 'off');
    hold on;
    [Y_sorted, sort_idx] = sort(Y);
    Y_pred_sorted = Y_pred(sort_idx);
    x = (1:n)';
    if SHOW_ERROR_BARS
        h_err = plot([x x]', [Y_sorted Y_pred_sorted]', '-', ...
            'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
        set(h_err, 'HandleVisibility', 'off');
    end
    scatter(x, Y_sorted, 28, 'r', 'filled', 'o', 'DisplayName', '實際長度 (Actual)');
    scatter(x, Y_pred_sorted, 28, 'b', 'filled', 's', 'DisplayName', 'Out-of-fold 預測長度');
    xlabel(sprintf('樣本排序編號 (依實際長度由小到大，n=%d)', n));
    ylabel('長度 (mm)');
    title({['實際長度 vs 預測長度 排序點圖 (' ttl ')'], ['輸入：' featDesc]});
    legend('Location', 'northwest');
    grid on; hold off;

    %% 5. 圖表 2：預測 vs 實際散布圖(未校正)
    figure('Name', ['OOF: Actual vs Predicted - ' METHOD], 'NumberTitle', 'off');
    hold on;
    if hasDemo
        scatter(Y(~isM), Y_pred(~isM), 40, 'b', 'filled', 'o', 'DisplayName', 'OOF 預測 (F)');
        scatter(Y(isM),  Y_pred(isM),  45, 'c', 'filled', 's', 'DisplayName', 'OOF 預測 (M)');
    else
        scatter(Y, Y_pred, 40, 'b', 'filled', 'DisplayName', 'Out-of-fold 預測');
    end
    min_val = floor(min([Y; Y_pred])) - 1;
    max_val = ceil(max([Y; Y_pred])) + 1;
    plot([min_val, max_val], [min_val, max_val], 'w-', 'LineWidth', 2, 'DisplayName', '完美預測線 (誤差 0)');
    plot([min_val, max_val], [min_val+0.5, max_val+0.5], 'r--', 'LineWidth', 1.5, 'DisplayName', '+0.5 mm 誤差線');
    plot([min_val, max_val], [min_val-0.5, max_val-0.5], 'r--', 'LineWidth', 1.5, 'DisplayName', '-0.5 mm 誤差線');
    plot([min_val, max_val], [min_val+1.0, max_val+1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '+1.0 mm 誤差線');
    plot([min_val, max_val], [min_val-1.0, max_val-1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '-1.0 mm 誤差線');
    xlabel('實際長度 (mm)'); ylabel('預測長度 (mm)');
    title(['實際長度 vs 預測長度 (' ttl '，未校正)']);
    legend('Location', 'southeast'); grid on;

    metric_text = sprintf(['【未校正 OOF 指標】\n量測法 : %s\n輸入 : %s\n' ...
        'MAE : %.3f mm%s\nRMSE : %.3f mm\nMean Bias : %.3f mm\n%sR^2 : %.3f\n(n = %d)'], ...
        METHOD, featDesc, test_mae, sdSuffix(sd_mae, 'mm'), test_rmse, test_bias, ...
        lineIf('R : %.3f\n', test_r), test_r2, n);
    placeText(metric_text, 0.18);
    hold off;

    %% 6. 圖表 3：校正後(報告用)
    figure('Name', ['Offset-corrected (OOF) - ' METHOD], 'NumberTitle', 'off');
    hold on;
    scatter(Y, Y_pred_offset, 45, 'b', 'filled', 'DisplayName', '校正後預測');
    min_val2 = floor(min([Y; Y_pred_offset])) - 1;
    max_val2 = ceil(max([Y; Y_pred_offset])) + 1;
    plot([min_val2, max_val2], [min_val2, max_val2], 'w-', 'LineWidth', 2, 'DisplayName', '完美預測線 (誤差 0)');
    plot([min_val2, max_val2], [min_val2+1.0, max_val2+1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '+1.0 mm 容忍線');
    plot([min_val2, max_val2], [min_val2-1.0, max_val2-1.0], 'g:', 'LineWidth', 1.5, 'DisplayName', '-1.0 mm 容忍線');
    xlabel('實際長度 (mm)');
    ylabel('校正後預測長度 = 迴歸輸出 - offset (mm)');
    title(['校正後預測長度 vs 實際長度 (' ttl ')']);
    legend('Location', 'southeast'); grid on;

    clinical_text = sprintf(['【校正後 OOF 指標 (報告用)】\n量測法 : %s\n輸入 : %s\n' ...
        'MAE : %.3f mm%s\nRMSE : %.3f mm\nMean Bias : %.3f mm\n%sR^2 : %.3f\n' ...
        '理想比率 : %.1f%%%s\n(n = %d)'], ...
        METHOD, featDesc, clinical_mae, sdSuffix(sd_mae, 'mm'), clinical_rmse, ...
        clinical_bias, lineIf('R : %.3f\n', clinical_r), clinical_r2, ...
        clinical_ideal, sdSuffix(sd_ideal, '%'), n);
    placeText(clinical_text, 0.26);
    hold off;

    %% 7. 圖表 4：誤差 vs 年齡 / 性別
    if hasDemo
        res = Y_pred - Y;

        figure('Name', ['Error Analysis: Age & Sex (OOF) - ' METHOD], 'NumberTitle', 'off');
        tiledlayout(1, 2, 'TileSpacing', 'compact');

        nexttile; hold on;
        scatter(age(~isM), res(~isM), 40, 'b', 'filled', 'o', 'DisplayName', 'F');
        scatter(age(isM),  res(isM),  45, 'c', 'filled', 's', 'DisplayName', 'M');
        pp = polyfit(age, res, 1);
        xa = [min(age) max(age)];
        plot(xa, polyval(pp, xa), 'y-', 'LineWidth', 1.8, ...
            'DisplayName', sprintf('趨勢 %.3f mm/歲', pp(1)));
        yline(0, 'w-', 'HandleVisibility', 'off');
        [r_age, p_age] = corr(age, res, 'Type', 'Spearman');
        xlabel('年齡 (歲)'); ylabel('誤差 = 預測 - 實際 (mm)');
        title(sprintf('誤差 vs 年齡 (Spearman ρ=%.2f, p=%.3f)', r_age, p_age));
        legend('Location', 'best'); grid on; hold off;

        nexttile;
        grp = categorical(sexLbl, ["F", "M"]);
        gx = double(grp);
        boxchart(gx, res, 'MarkerStyle', 'none');
        hold on;
        rng(0);
        jit = (rand(n,1) - 0.5) * 0.25;
        scatter(gx + jit, res, 25, 'w', 'filled', 'MarkerFaceAlpha', 0.6);
        yline(0, 'w-');
        hold off;
        xticks([1 2]); xticklabels({'F', 'M'}); xlim([0.5 2.5]);
        p_sex = NaN;
        if any(isM) && any(~isM)
            p_sex = ranksum(res(isM), res(~isM));
        end
        maeF = mean(abs(res(~isM))); maeM = mean(abs(res(isM)));
        ylabel('誤差 = 預測 - 實際 (mm)');
        title(sprintf('誤差依性別 (MAE F=%.2f / M=%.2f, 秩和 p=%.3f)', maeF, maeM, p_sex));
        grid on;

        sgtitle(sprintf('誤差分析 (%s，輸入：%s，n=%d)', ttl, featDesc, n));

        fprintf('\n=== 誤差分析 ===\n');
        fprintf('  誤差 vs 年齡：Spearman ρ=%.3f, p=%.3f, 斜率 %.4f mm/歲\n', r_age, p_age, pp(1));
        fprintf('  性別：F n=%d MAE=%.3f bias=%.3f | M n=%d MAE=%.3f bias=%.3f | 秩和 p=%.3f\n', ...
            sum(~isM), maeF, mean(res(~isM)), sum(isM), maeM, mean(res(isM)), p_sex);
        fprintf('  ⚠️ n 小、單次檢定，p 值只當參考。\n');
    else
        fprintf('ℹ️ 輸出檔沒有年齡/性別欄，略過圖表 4。\n');
    end

    %% 8. 終端機摘要(校正後)
    fprintf('\n=== 校正後 OOF 指標(報告用) ===\n');
    fprintf('  MAE       : %.3f mm%s\n', clinical_mae, sdSuffix(sd_mae, 'mm'));
    fprintf('  RMSE      : %.3f mm\n', clinical_rmse);
    fprintf('  Mean Bias : %.3f mm  (正 = 整體偏長)\n', clinical_bias);
    printIf('  R         : %.3f  (offset 不影響，與未校正同值)\n', clinical_r);
    fprintf('  R^2       : %.3f  (同上)\n', clinical_r2);
    fprintf('  理想比率  : %.1f%%%s\n', clinical_ideal, sdSuffix(sd_ideal, '%'));

    fprintf('\n✅ 圖表繪製完成。\n');
    fprintf('⚠️ 報告時務必附上 n=%d 與標準差。\n', n);
    if ~isnan(lin_mae) && lin_mae <= clinical_mae
        fprintf('ℹ️ 線性對照組 MAE(%.3f) 不輸 ANN(%.3f)，解讀時請一併說明。\n', lin_mae, clinical_mae);
    end
end


%% ============================================================
%  子函式
%  ============================================================

function tag = featTag(useAge, useSex)
% 必須跟 C1 的 featTag 規則一致
    parts = {};
    if useAge, parts{end+1} = '年齡'; end
    if useSex, parts{end+1} = '性別'; end
    if isempty(parts)
        tag = '';
    else
        tag = ['_' strjoin(parts, '')];
    end
end


function row = pickRow(tbl, labels, key)
    idx = contains(labels, key);
    if sum(idx) ~= 1
        error(['在「模型評估指標」裡找不到唯一的「%s」那一列(找到 %d 列)。\n' ...
               '   請確認 C1 的 MetricNames 沒有被改動。'], key, sum(idx));
    end
    row = tbl(idx, :);
end


function row = pickRowOptional(tbl, labels, key)
    idx = contains(labels, key);
    if sum(idx) == 1
        row = tbl(idx, :);
    else
        row = [];
    end
end


function v = getNum(row, colName)
% 欄位不存在就回 NaN，舊版 C1 的輸出也讀得起來
    if isempty(row) || ~ismember(colName, row.Properties.VariableNames)
        v = NaN; return;
    end
    x = row.(colName);
    if iscell(x)
        v = str2double(string(x));
    else
        v = double(x);
    end
    if isempty(v), v = NaN; else, v = v(1); end
end


function s = lineIf(fmtStr, v)
% 值是 NaN 就回空字串，文字框不會出現空洞的那一行
    if isnan(v)
        s = '';
    else
        s = sprintf(fmtStr, v);
    end
end


function printIf(fmtStr, v)
    if ~isnan(v), fprintf(fmtStr, v); end
end


function s = sdSuffix(sd, unit)
    if isnan(sd)
        s = '';
    elseif strcmp(unit, '%')
        s = sprintf(' ± %.1f', sd);
    else
        s = sprintf(' ± %.3f', sd);
    end
end


function placeText(txt, yOffsetRatio)
    x_lims = xlim;
    y_lims = ylim;
    text(x_lims(1) + 0.05*(x_lims(2)-x_lims(1)), ...
         y_lims(2) - yOffsetRatio*(y_lims(2)-y_lims(1)), ...
         txt, 'FontSize', 11, 'BackgroundColor', 'k', ...
         'EdgeColor', 'w', 'Color', 'w');
end
