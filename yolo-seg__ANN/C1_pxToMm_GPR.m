function C1_pixelToMmPredictor_GPR()
%% C1_pixelToMmPredictor_GPR.m
% ============================================================
% 以 C1 v6 為底，只換一個變因：後端 ANN(fitnet + trainbr) → GPR(fitrgp)。
% 輸入檔、分摺、標準化、offset、指標定義、線性對照組、輸出 sheet 名稱
% 與列關鍵字全部沒動 → C2 可以直接讀(只要改 C2 的檔名那一行，見最後提示)。
%
% 跟 v6 不同的地方(都是換模型帶來的，不是另外加的變因)：
%   (1) GPR 沒有隨機初始化，摺固定時重跑結果完全相同
%       → N_REPEATS 預設 1；B 列(重複間標準差)填 NaN，C2 會自動不顯示 ±。
%   (2) 逐筆表多一欄「預測標準差σ_mm」(GPR 的預測不確定性)，C2 不讀這欄。
%   (3) 新增 sheet「GPR每摺超參數」：長度尺度 σL、訊號 σf、雜訊 σn。
%       σL(輸入已標準化)遠大於 1(例如 >10) = GP 退化成「線性迴歸 + 常數誤差帶」，
%       n 小、特徵少時很常見，不是跑錯；這時 GPR 跟線性對照組的 MAE 會很接近。
%
% ⚠️ offset 只平移預測值：R、R^2 校正前後同值；MAE、RMSE、Bias 會變(同 v6)。
% 需要：Statistics and Machine Learning Toolbox(fitrgp)
% ============================================================

    %% ---------------- 輸入設定區 ----------------
    METHOD = 'mask幾何_長邊正規化';   % 'mask幾何_長邊正規化' | '冠寬比例尺'
    filename = ['根管填充物像素長度_已配對_11n_tophat_' METHOD '.xlsx'];

    FOLD_COLUMN  = '折數fold';
    SPLIT_COLUMN = '資料夾來源';   % 只在 USE_KFOLD=false 時才用

    %% ---------------- 特徵設定區 ----------------
    USE_AGE = true;
    USE_SEX = false;
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    %% ---------------- 評估設定區 ----------------
    USE_KFOLD  = true;
    N_REPEATS  = 1;       % GPR 是確定性的，>1 只會得到一模一樣的數字
    BASE_SEED  = 42;

    %% ---------------- GPR 設定區 ----------------
    GPR_KERNEL = 'squaredexponential';   % 'ardsquaredexponential' = 每個特徵各自的長度尺度
                                         % 'matern52' = 比 SE 粗糙一點的曲線
    GPR_BASIS  = 'linear';               % 平均函數：'linear' = GP 只學線性迴歸剩下的部分
                                         %           'constant' = fitrgp 預設

    RUN_LINEAR_BASELINE = true;

    %% ---------------- 臨床設定區 ----------------
    OFFSET_MM          = 0.5;
    IDEAL_TOLERANCE_MM = 1.0;

    %% --- 1. 讀取資料 ---
    if ~isfile(filename)
        error(['找不到 %s\n' ...
               '   這份檔案由 B4_build_C1_input.py 產生，請先跑那支腳本。'], filename);
    end
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    if ismember(SEX_COLUMN, opts.VariableNames)
        opts = setvartype(opts, SEX_COLUMN, 'string');
    end
    data = readtable(filename, opts);

    ids = data.('圖片檔名');
    Y   = data.('填充物長度(mm)');
    X   = data.('像素長度');

    if ~isnumeric(Y) || ~isnumeric(X)
        error('實際長度或像素長度欄位含非數值資料，請檢查 Excel。');
    end

    vn = data.Properties.VariableNames;
    hasDemo = ismember(AGE_COLUMN, vn) && ismember(SEX_COLUMN, vn);
    if (USE_AGE || USE_SEX) && ~hasDemo
        error(['輸入檔沒有「%s」「%s」欄。\n' ...
               '   請在 B4 的 MM_EXTRA_COLS 加上這兩欄、MM_XLSX 指向新版醫師表，重跑 B4。'], ...
               AGE_COLUMN, SEX_COLUMN);
    end

    if hasDemo
        age = toNum(data.(AGE_COLUMN));
        sex = parseSex(data.(SEX_COLUMN));
        if all(isnan(sex))
            error('「%s」欄全部無法解析(只認 M/F/男/女)，請檢查 Excel。', SEX_COLUMN);
        end
    else
        age = nan(height(data), 1);
        sex = nan(height(data), 1);
    end

    valid = ~isnan(X) & ~isnan(Y);
    if hasDemo
        valid = valid & ~isnan(age) & ~isnan(sex);
    end
    if any(~valid)
        fprintf('⚠️ 有 %d 筆含缺值(長度/年齡/性別)，已剔除。\n', sum(~valid));
        ids = ids(valid); Y = Y(valid); X = X(valid);
        age = age(valid); sex = sex(valid);
        data = data(valid, :);
    end
    n = numel(Y);

    % --- 組特徵矩陣 ---
    Xmat = X;
    featNames = {'像素長度'};
    if USE_AGE, Xmat = [Xmat age]; featNames{end+1} = '年齡'; end
    if USE_SEX, Xmat = [Xmat sex]; featNames{end+1} = '性別(M=1)'; end
    p = size(Xmat, 2);
    featDesc = strjoin(featNames, ' + ');
    FEAT_TAG = featTag(USE_AGE, USE_SEX);

    fprintf('=== C1 GPR：%s ===\n', METHOD);
    fprintf('載入 %d 筆樣本。輸入特徵：%s\n', n, featDesc);
    fprintf('GPR：kernel = %s，basis = %s\n', GPR_KERNEL, GPR_BASIS);
    if hasDemo
        fprintf('性別：M %d / F %d；年齡 %g–%g 歲(中位數 %g)\n', ...
            sum(sex == 1), sum(sex == 0), min(age), max(age), median(age));
    end

    %% --- 2. 決定每一筆屬於哪一摺(與 v6 相同) ---
    if USE_KFOLD
        if ismember(FOLD_COLUMN, vn)
            foldId = double(data.(FOLD_COLUMN));
            fprintf('沿用 Excel 的「%s」欄分摺(與 ANN 版共用同一組切分)。\n', FOLD_COLUMN);
        else
            K = 5;
            [~, ord] = sort(Y);
            foldId = zeros(n, 1);
            foldId(ord) = mod(0:n-1, K) + 1;
            fprintf('⚠️ Excel 沒有「%s」欄，改用依實際長度分層的 %d 摺。\n', FOLD_COLUMN, K);
            fprintf('   注意：這樣跟 ANN 版的切分不保證一致，對照會混入切分差異。\n');
        end
        K = max(foldId);
        fprintf('共 %d 摺，每摺樣本數：', K);
        fprintf('%d ', accumarray(foldId, 1)); fprintf('\n');
    else
        if ~ismember(SPLIT_COLUMN, vn)
            error('USE_KFOLD=false 需要「%s」欄。', SPLIT_COLUMN);
        end
        isTest = string(data.(SPLIT_COLUMN)) == "test";
        foldId = ones(n, 1);
        foldId(~isTest) = 0;
        K = 1;
        fprintf('單次切分模式：train %d 筆 / test %d 筆\n', sum(~isTest), sum(isTest));
    end

    %% --- 3. 交叉驗證主迴圈(只把 ANN 換成 GPR) ---
    gprPred = nan(n, N_REPEATS);
    gprSd   = nan(n, N_REPEATS);
    linPred = nan(n, N_REPEATS);
    foldMAE = nan(K, N_REPEATS);

    hpRows = cell(K, 1);   % 每摺 [核參數..., σn]，只記第 1 次重複

    fprintf('\n訓練中(%d 摺 × %d 次重複)...\n', K, N_REPEATS);
    for r = 1:N_REPEATS
        rng(BASE_SEED + r);
        for k = 1:K
            teIdx = (foldId == k);
            trIdx = ~teIdx & (foldId > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end

            Xtr = Xmat(trIdx, :); Ytr = Y(trIdx);
            Xte = Xmat(teIdx, :);

            mu = mean(Xtr, 1);
            sg = std(Xtr, 0, 1);
            sg(sg == 0) = 1;
            XtrN = (Xtr - mu) ./ sg;
            XteN = (Xte - mu) ./ sg;

            gpr = fitrgp(XtrN, Ytr, ...
                'KernelFunction', GPR_KERNEL, ...
                'BasisFunction',  GPR_BASIS, ...
                'Standardize',    false);          % X 已手動標準化(同 v6)
            [mu_k, sd_k] = predict(gpr, XteN);     % sd = 預測區間標準差(含雜訊)
            gprPred(teIdx, r) = mu_k;
            gprSd(teIdx, r)   = sd_k;

            if r == 1
                hpRows{k} = [gpr.KernelInformation.KernelParameters(:)' gpr.Sigma];
            end

            if RUN_LINEAR_BASELINE
                b = [ones(size(Xtr,1),1) Xtr] \ Ytr;
                linPred(teIdx, r) = [ones(size(Xte,1),1) Xte] * b;
            end

            foldMAE(k, r) = mean(abs(gprPred(teIdx, r) - Y(teIdx)));
        end
        fprintf('  重複 %d/%d 完成\n', r, N_REPEATS);
    end

    %% --- 4. 逐次重複算指標，再取平均與標準差 ---
    M = emptyMetrics(N_REPEATS);
    for r = 1:N_REPEATS
        M = accumMetrics(M, r, gprPred(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
    end
    L = emptyMetrics(N_REPEATS);
    if RUN_LINEAR_BASELINE
        for r = 1:N_REPEATS
            L = accumMetrics(L, r, linPred(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
        end
    end
    foldSpread = mean(std(foldMAE, 0, 1, 'omitnan'), 'omitnan');

    %% --- 5. 代表性重複 ---
    [~, repIdx] = min(abs(M.mae - median(M.mae)));
    Y_pred = gprPred(:, repIdx);
    Y_sd   = gprSd(:, repIdx);
    if N_REPEATS > 1
        fprintf('\n逐筆結果採用第 %d 次重複(MAE=%.3f，最接近中位數)。\n', repIdx, M.mae(repIdx));
    end

    Y_pred_offset  = Y_pred - OFFSET_MM;
    residuals      = Y_pred - Y;
    abs_error      = abs(residuals);
    off_residuals  = Y_pred_offset - Y;
    off_abs_error  = abs(off_residuals);

    %% --- 6. 組輸出表 ---
    split_label = repmat("test", n, 1);

    ResultsTable = table(ids, split_label, X, Y, ...
        Y_pred, residuals, abs_error, ...
        Y_pred_offset, off_residuals, off_abs_error, foldId, round(Y_sd, 3), ...
        'VariableNames', {'檔名', 'Train_or_Test', '像素長度_px', '實際長度_mm', ...
        '模型預測_mm', '誤差Bias_mm', '絕對誤差_mm', ...
        '校正後預測_mm', '校正後誤差Bias_mm', '校正後絕對誤差_mm', '摺別', '預測標準差σ_mm'});
    if hasDemo
        sexLbl = repmat("F", n, 1);
        sexLbl(sex == 1) = "M";
        ResultsTable.('年齡') = age;
        ResultsTable.('性別') = sexLbl;
    end

    % 'A_Test set表現' 與 'C_臨床安全性評估_Test set校正後' 是 C2 的抓列關鍵字，勿改
    MetricNames = {
        sprintf('A_Test set表現 (out-of-fold, n=%d, GPR, %d次重複平均, 輸入=%s)', n, N_REPEATS, featDesc);
        sprintf('B_重複間標準差 (同一份資料重跑%d次的波動)', N_REPEATS);
        '--- 分界線 ---';
        sprintf('C_臨床安全性評估_Test set校正後 (offset=-%.2fmm)', OFFSET_MM);
        sprintf('D_線性迴歸對照組_未校正 (同一組摺, n=%d)', n);
        sprintf('E_線性迴歸對照組_校正後 (offset=-%.2fmm)', OFFSET_MM)
    };

    nanpad = NaN;
    hasLin = RUN_LINEAR_BASELINE;
    sdOr = @(x) sdIfRepeats(x, N_REPEATS);   % 只跑 1 次 → NaN，C2 不顯示 ±

    MAE_val = round([mean(M.mae); sdOr(M.mae); nanpad; mean(M.offMae); ...
                     meanOr(L.mae, hasLin); meanOr(L.offMae, hasLin)], 3);
    RMSE_val = round([mean(M.rmse); sdOr(M.rmse); nanpad; mean(M.offRmse); ...
                      meanOr(L.rmse, hasLin); meanOr(L.offRmse, hasLin)], 3);
    Bias_val = round([mean(M.bias); sdOr(M.bias); nanpad; mean(M.offBias); ...
                      meanOr(L.bias, hasLin); meanOr(L.offBias, hasLin)], 3);
    R_val  = round([mean(M.r); sdOr(M.r); nanpad; mean(M.r); ...
                    meanOr(L.r, hasLin); meanOr(L.r, hasLin)], 3);
    R2_val = round([mean(M.r2); sdOr(M.r2); nanpad; mean(M.r2); ...
                    meanOr(L.r2, hasLin); meanOr(L.r2, hasLin)], 3);
    IdealRate_pct = round([nanpad; sdOr(M.ideal); nanpad; mean(M.ideal); ...
                           nanpad; meanOr(L.ideal, hasLin)], 1);

    MetricsTable = table(MetricNames, MAE_val, RMSE_val, R_val, R2_val, Bias_val, ...
        IdealRate_pct, ...
        'VariableNames', {'評估範圍_與_嚴格程度', 'MAE_mm', 'RMSE_mm', 'R_Pearson', ...
        'R_Square', 'Mean_Bias_mm', '理想比率_pct'});

    SettingsTable = table( ...
        ["量測方法"; "輸入特徵"; "特徵標籤"; "n"; "模型"; "GPR kernel"; "GPR basis"; "腳本版本"], ...
        [string(METHOD); string(featDesc); string(FEAT_TAG); string(n); "GPR"; ...
         string(GPR_KERNEL); string(GPR_BASIS); "C1_GPR_v1"], ...
        'VariableNames', {'項目', '值'});

    % --- 每摺超參數 ---
    nHP = max(cellfun(@numel, hpRows));
    hp = nan(K, nHP);
    for k = 1:K
        if ~isempty(hpRows{k}), hp(k, :) = hpRows{k}; end
    end
    if startsWith(GPR_KERNEL, 'ard') && nHP == p + 2
        hpNames = [strcat("σL_", string(featNames)), "σf", "σn"];
    elseif nHP == 3
        hpNames = ["σL", "σf", "σn"];
    else                                    % 例：rationalquadratic = σL, α, σf
        hpNames = [compose("θ%d", 1:nHP-1), "σn"];
    end
    HpTable = array2table(round(hp, 4), 'VariableNames', cellstr(hpNames));
    HpTable = addvars(HpTable, (1:K)', 'Before', 1, 'NewVariableNames', '摺');

    % --- 全資料線性迴歸：只為解讀效應大小，不是效能(同 v6) ---
    CoefTable = [];
    if hasDemo
        tbl = table(X, age, sex, Y, 'VariableNames', {'X', 'age', 'sexM', 'Y'});
        predictors = {'X'};
        if USE_AGE, predictors{end+1} = 'age'; end
        if USE_SEX, predictors{end+1} = 'sexM'; end
        mdl = fitlm(tbl, 'ResponseVar', 'Y', 'PredictorVars', predictors);
        c = mdl.Coefficients;
        CoefTable = table(string(c.Properties.RowNames), round(c.Estimate, 4), ...
            round(c.SE, 4), round(c.tStat, 3), round(c.pValue, 4), ...
            'VariableNames', {'項', '係數', 'SE', 't', 'p值'});
    end

    %% --- 7. 匯出 ---
    output_filename = sprintf('預測結果與評估指標_kfold_GPR_%s%s.xlsx', METHOD, FEAT_TAG);
    if isfile(output_filename), delete(output_filename); end
    writetable(ResultsTable, output_filename, 'Sheet', '所有牙齒預測結果');
    writetable(MetricsTable, output_filename, 'Sheet', '模型評估指標');
    writetable(SettingsTable, output_filename, 'Sheet', '模型設定');
    writetable(HpTable,      output_filename, 'Sheet', 'GPR每摺超參數');
    if ~isempty(CoefTable)
        writetable(CoefTable, output_filename, 'Sheet', '線性係數_全資料');
    end

    fprintf('\n✅ 已匯出：%s\n\n', output_filename);
    disp('=== 評估指標 ===');
    disp(MetricsTable);
    disp('=== GPR 每摺超參數(輸入已標準化) ===');
    disp(HpTable);

    %% --- 8. 報告用區塊(校正後) ---
    fprintf('\n=== 臨床評估(校正後，報告用) ===\n');
    fprintf('量測方法：%s | 輸入：%s | offset = -%.2f mm | 理想容忍 = ±%.2f mm\n', ...
        METHOD, featDesc, OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('[Out-of-fold, n=%d, GPR %s / basis %s]\n', n, GPR_KERNEL, GPR_BASIS);
    fprintf('  校正後 MAE  : %.3f%s mm\n', mean(M.offMae),  pm(M.offMae,  N_REPEATS));
    fprintf('  校正後 RMSE : %.3f%s mm\n', mean(M.offRmse), pm(M.offRmse, N_REPEATS));
    fprintf('  校正後 Bias : %.3f%s mm   (正 = 整體偏長)\n', mean(M.offBias), pm(M.offBias, N_REPEATS));
    fprintf('  R (Pearson) : %.3f%s      (offset 不影響，與未校正同值)\n', mean(M.r), pm(M.r, N_REPEATS));
    fprintf('  R^2         : %.3f%s\n', mean(M.r2), pm(M.r2, N_REPEATS));
    fprintf('  理想比率    : %.1f%s %%\n', mean(M.ideal), pm(M.ideal, N_REPEATS));
    fprintf('  摺間 MAE 標準差 : %.3f mm\n', foldSpread);
    fprintf('  平均預測 σ      : %.3f mm\n', mean(Y_sd, 'omitnan'));

    if RUN_LINEAR_BASELINE
        fprintf('\n[線性迴歸對照組，校正後] MAE %.3f mm vs GPR %.3f mm  →  ', ...
            mean(L.offMae), mean(M.offMae));
        if mean(L.offMae) <= mean(M.offMae)
            fprintf('線性不輸 GPR\n');
        else
            fprintf('GPR 勝出 %.3f mm\n', mean(L.offMae) - mean(M.offMae));
        end
    end
    if median(hp(:, 1), 'omitnan') > 10
        fprintf('ℹ️ σL 中位數 %.1f > 10：GP 幾乎退化成線性(看「GPR每摺超參數」)。\n', ...
            median(hp(:, 1), 'omitnan'));
    end

    if ~isempty(CoefTable)
        fprintf('\n=== 全資料線性係數(解讀用，非效能) ===\n');
        disp(CoefTable);
        fprintf('  p 值只是參考：n=%d、未校正多重比較。\n', n);
    end

    fprintf('\n👉 要用 C2 畫圖：把 C2 的 filename 那行改成\n');
    fprintf('   filename = sprintf(''預測結果與評估指標_kfold_GPR_%%s%%s.xlsx'', METHOD, featTag(USE_AGE, USE_SEX));\n');
    fprintf('👉 跟 ANN 版對照時，METHOD / USE_AGE / USE_SEX / 輸入檔要一樣，才是只換模型。\n');
end


%% ============================================================
%  子函式(emptyMetrics / accumMetrics / meanOr / featTag / toNum / parseSex 與 v6 相同)
%  ============================================================

function S = emptyMetrics(N)
    S = struct('mae', nan(N,1), 'rmse', nan(N,1), 'r', nan(N,1), 'r2', nan(N,1), ...
               'bias', nan(N,1), 'offMae', nan(N,1), 'offRmse', nan(N,1), ...
               'offBias', nan(N,1), 'ideal', nan(N,1));
end


function S = accumMetrics(S, r, Yp, Y, offset, tol)
    ok = ~isnan(Yp) & ~isnan(Y);
    Yp = Yp(ok); Y = Y(ok);

    res  = Yp - Y;
    S.mae(r)  = mean(abs(res));
    S.rmse(r) = sqrt(mean(res.^2));
    S.bias(r) = mean(res);
    if numel(Y) >= 2
        S.r(r)  = corr(Y, Yp);
        S.r2(r) = S.r(r)^2;
    end

    offRes = (Yp - offset) - Y;
    S.offMae(r)  = mean(abs(offRes));
    S.offRmse(r) = sqrt(mean(offRes.^2));
    S.offBias(r) = mean(offRes);
    S.ideal(r)   = mean(abs(offRes) <= tol) * 100;
end


function v = meanOr(x, flag)
    if flag
        v = mean(x, 'omitnan');
    else
        v = NaN;
    end
end


function v = sdIfRepeats(x, nRep)
% 只跑 1 次時標準差沒有意義 → NaN(C2 讀到 NaN 就不顯示 ±)
    if nRep > 1
        v = std(x, 'omitnan');
    else
        v = NaN;
    end
end


function s = pm(x, nRep)
    if nRep > 1
        s = sprintf(' ± %.3f', std(x, 'omitnan'));
    else
        s = '';
    end
end


function tag = featTag(useAge, useSex)
    parts = {};
    if useAge, parts{end+1} = '年齡'; end
    if useSex, parts{end+1} = '性別'; end
    if isempty(parts)
        tag = '';
    else
        tag = ['_' strjoin(parts, '')];
    end
end


function v = toNum(x)
    if iscell(x) || isstring(x)
        v = str2double(string(x));
    else
        v = double(x);
    end
end


function s = parseSex(v)
    t = upper(strtrim(string(v)));
    s = nan(numel(t), 1);
    s(ismember(t, ["M", "MALE", "男"])) = 1;
    s(ismember(t, ["F", "FEMALE", "女"])) = 0;
end
