function C2v2_plotMyResults()
%% C2v2_plotMyResults.m
% ============================================================
% 讀 C1v2_pixelToMmPredictor.m 的輸出，畫：
%   1. 實際長度 vs 預測長度 排序點圖
%   2. 預測 vs 實際散布圖(含 ±0.5 / ±1.0 mm 容忍線；有性別時分 M/F 標記)
%   3. 校正後工作長度 vs 實際值，含過長標示(臨床安全性圖)
%   4. (新增，有年齡/性別欄時) 誤差分析：誤差 vs 年齡、誤差依性別
%
% *** 2026-09-22：年齡 / 性別 ***
%   - 檔名跟 C1 一樣由 METHOD + USE_AGE/USE_SEX 組成，三個設定要跟 C1 那次一致
%   - 讀「模型設定」sheet 把輸入特徵印在文字框
%   - 圖 4 怎麼讀：
%       X only 模型的誤差若跟年齡明顯相關 → 年齡帶有影像沒抓到的資訊，值得進模型
%       X+年齡 模型的誤差仍跟年齡相關       → 關係可能非線性，或樣本太少學不到
%       性別兩組誤差中位數差很多            → 同理，看性別
%   - r/p 值是 n≈54 的單次檢定，只當參考
% ============================================================

    %% 1. 設定(必須跟 C1 跑的那次一致)
    METHOD  = '冠寬比例尺';   % 'mask幾何' | '冠寬比例尺' | '牙位基準'
    USE_AGE = true;
    USE_SEX = true;
    SHOW_ERROR_BARS = false;

    filename = sprintf('預測結果與評估指標_kfold_%s%s.xlsx', METHOD, featTag(USE_AGE, USE_SEX));
    if ~isfile(filename)
        error(['找不到 %s\n' ...
               '   請先用同樣的 METHOD / USE_AGE / USE_SEX 跑一次 C1v2。'], filename);
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
    is_overest    = logical(data_test.('是否過長'));
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
    lin_row      = pickRowOptional(data_metrics, labels, 'D_線性迴歸對照組');

    test_mae  = num(test_row.('MAE_mm'));
    test_rmse = num(test_row.('RMSE_mm'));
    test_r2   = num(test_row.('R_Square'));
    test_bias = num(test_row.('Mean_Bias_mm'));

    clinical_mae     = num(clinical_row.('MAE_mm'));
    clinical_bias    = num(clinical_row.('Mean_Bias_mm'));
    clinical_ideal   = num(clinical_row.('理想比率_pct'));
    clinical_overest = num(clinical_row.('過長率_pct'));

    sd_mae = NaN; sd_ideal = NaN; sd_over = NaN;
    if ~isempty(sd_row)
        sd_mae   = num(sd_row.('MAE_mm'));
        sd_ideal = num(sd_row.('理想比率_pct'));
        sd_over  = num(sd_row.('過長率_pct'));
    end
    lin_mae = NaN;
    if ~isempty(lin_row)
        lin_mae = num(lin_row.('MAE_mm'));
    end

    featDesc = '像素長度';   % 舊版輸出沒有「模型設定」sheet 時的預設
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

    %% 5. 圖表 2：預測 vs 實際散布圖
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
    title(['實際長度 vs 預測長度 (' ttl '，搭配誤差容忍區間)']);
    legend('Location', 'southeast'); grid on;

    metric_text = sprintf(['【Out-of-Fold 評估指標】\n量測法 : %s\n輸入 : %s\n' ...
        'RMSE : %.3f mm\nMAE : %.3f mm%s\nR^2 : %.3f\nMean Bias : %.3f mm\n(n = %d)'], ...
        METHOD, featDesc, test_rmse, test_mae, sdSuffix(sd_mae, 'mm'), test_r2, test_bias, n);
    if ~isnan(lin_mae)
        metric_text = sprintf('%s\n線性對照組 MAE : %.3f mm', metric_text, lin_mae);
    end
    placeText(metric_text, 0.18);
    hold off;

    %% 6. 圖表 3：臨床安全性圖
    figure('Name', ['Clinical Safety (OOF) - ' METHOD], 'NumberTitle', 'off');
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
    title(['臨床建議工作長度 vs 實際長度 (' ttl '，含過長標示)']);
    legend('Location', 'southeast'); grid on;

    clinical_text = sprintf(['【臨床與安全性評估 (校正後, OOF)】\n量測法 : %s\n輸入 : %s\n' ...
        'MAE : %.3f mm%s\nMean Bias : %.3f mm\n理想比率 : %.1f%%%s\n過長率 : %.1f%%%s\n(n = %d)'], ...
        METHOD, featDesc, clinical_mae, sdSuffix(sd_mae, 'mm'), clinical_bias, ...
        clinical_ideal, sdSuffix(sd_ideal, '%'), ...
        clinical_overest, sdSuffix(sd_over, '%'), n);
    placeText(clinical_text, 0.22);
    hold off;

    %% 7. 圖表 4：誤差 vs 年齡 / 性別
    if hasDemo
        res = Y_pred - Y;   % 未校正誤差(offset 只是整體平移，不影響跟年齡/性別的關係)

        figure('Name', ['Error Analysis: Age & Sex (OOF) - ' METHOD], 'NumberTitle', 'off');
        tiledlayout(1, 2, 'TileSpacing', 'compact');

        % --- 左：誤差 vs 年齡 ---
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

        % --- 右：誤差依性別 ---
        nexttile;
        grp = categorical(sexLbl, ["F", "M"]);
        gx = double(grp);   % 用數值 x，才能疊 scatter(categorical 軸不能疊數值點)
        boxchart(gx, res, 'MarkerStyle', 'none');
        hold on;
        rng(0);             % 抖動固定，重畫一樣
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
        fprintf('  ⚠️ n 小、單次檢定，p 值只當參考；重點看不同特徵組合下這些數字怎麼變。\n');
    else
        fprintf('ℹ️ 輸出檔沒有年齡/性別欄，略過圖表 4(請用新版 C1 + B4 重跑)。\n');
    end

    fprintf('✅ 圖表繪製完成。\n');
    fprintf('⚠️ 報告時務必附上 n=%d 與標準差。\n', n);
    if ~isnan(lin_mae) && lin_mae <= test_mae
        fprintf('ℹ️ 線性對照組的 MAE(%.3f) 不輸 ANN(%.3f)，解讀時請一併說明。\n', lin_mae, test_mae);
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
               '   請確認 C1v2 的 MetricNames 沒有被改動。'], key, sum(idx));
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


function v = num(x)
    if iscell(x)
        v = str2double(string(x));
    else
        v = double(x);
    end
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
