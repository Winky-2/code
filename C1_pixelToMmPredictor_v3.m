function C1v2_pixelToMmPredictor()
%% C1v2_pixelToMmPredictor.m
% ============================================================
% 對應流程圖 C 區塊：「轉換實際長度(-0.5mm)」+「臨床與安全性評估
% (MAE/理想比率/過長率)」。
%
% *** 2026-09-22：新增年齡 / 性別 ***
% 醫師表新增「年齡」「性別」兩欄，經 B4 帶進輸入檔(B4 的 MM_EXTRA_COLS
% 要加上這兩欄)。本版可以把它們當成額外的輸入特徵：
%
%   USE_AGE = true   → 年齡(連續值)進模型
%   USE_SEX = true   → 性別編碼成 M=1 / F=0 進模型
%
% 兩個開關可以獨立開關，方便一次只改一個變因做對照：
%   X only → X+年齡 → X+性別 → X+年齡+性別
% 輸出檔名會帶特徵標籤(例：_年齡性別)，不同組合不會互相覆蓋。
% 全部關掉時檔名與舊版相同(向下相容)。
%
% 不管開關怎麼設，只要輸入檔有這兩欄，年齡或性別缺值的列一律剔除，
% 這樣四種特徵組合的 n 與樣本完全相同，比出來的差異只來自特徵本身。
%
% 牙位基準 + 年齡 + 性別 = 「完全不看影像」的最強 baseline，
% 影像方法要贏過這個組合，才能說影像有貢獻。
%
% *** 參數量提醒 ***
% fitnet(H) 輸入 p 維的參數數 = H*p + H + H + 1。H=5 時：
%   p=1 → 16、p=2 → 21、p=3 → 26。n≈54 時 p=3 約 2 筆養一個參數，
% 靠 trainbr 的正則化撐著；線性對照組更值得看。
%
% *** 沿用的規矩(同前版) ***
% 1. 標準化參數只用 train 摺算(每個特徵各自算 mean/std)
% 2. 摺的分派沿用 Excel 的「折數fold」欄
% 3. 重複 N_REPEATS 次取平均與標準差
% 4. 輸出工作表/欄位/MetricNames 關鍵字不變，C2 讀得到
%
% *** 新增輸出 ***
%   逐筆結果多「年齡」「性別」兩欄(C2 用來做誤差分析)
%   「模型設定」sheet：量測法、輸入特徵、n、參數數
%   「線性係數_全資料」sheet：全資料線性迴歸(fitlm)的係數與 p 值，
%     只用來解讀「年齡/性別效應大小」，不是效能指標(效能看 OOF)。
% ============================================================

    %% ---------------- 輸入設定區 ----------------
    METHOD = '冠寬比例尺' ;   % 'mask幾何' | '冠寬比例尺' | '牙位基準'
    filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];

    FOLD_COLUMN  = '折數fold';
    SPLIT_COLUMN = '資料夾來源';   % 只在 USE_KFOLD=false 時才用

    %% ---------------- 特徵設定區(新增) ----------------
    USE_AGE = true;      % 年齡進模型
    USE_SEX = true;      % 性別進模型(M=1, F=0)
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    %% ---------------- 評估設定區 ----------------
    USE_KFOLD  = true;
    N_REPEATS  = 10;
    BASE_SEED  = 42;

    HIDDEN_SIZE = 5;
    TRAIN_FCN   = 'trainbr';
    MAX_EPOCHS  = 200;

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
    % 性別讀成文字，避免被猜成數值而全變 NaN
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
        sex = parseSex(data.(SEX_COLUMN));          % M=1, F=0, 其他 NaN
        if all(isnan(sex))
            error('「%s」欄全部無法解析(只認 M/F/男/女)，請檢查 Excel。', SEX_COLUMN);
        end
    else
        age = nan(height(data), 1);
        sex = nan(height(data), 1);
    end

    valid = ~isnan(X) & ~isnan(Y);
    if hasDemo
        % 不管開關怎麼設都套用，讓各特徵組合的樣本完全一致
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
    nParam = HIDDEN_SIZE*p + HIDDEN_SIZE + HIDDEN_SIZE + 1;

    fprintf('=== C1v2：%s ===\n', METHOD);
    fprintf('載入 %d 筆樣本。輸入特徵：%s\n', n, featDesc);
    fprintf('ANN 參數數 %d，約 %.1f 筆 / 參數\n', nParam, n / nParam);
    if hasDemo
        fprintf('性別：M %d / F %d；年齡 %g–%g 歲(中位數 %g)\n', ...
            sum(sex == 1), sum(sex == 0), min(age), max(age), median(age));
    end

    %% --- 2. 決定每一筆屬於哪一摺 ---
    if USE_KFOLD
        if ismember(FOLD_COLUMN, vn)
            foldId = double(data.(FOLD_COLUMN));
            fprintf('沿用 Excel 的「%s」欄分摺(三種方法共用同一組切分)。\n', FOLD_COLUMN);
        else
            K = 5;
            [~, ord] = sort(Y);
            foldId = zeros(n, 1);
            foldId(ord) = mod(0:n-1, K) + 1;
            fprintf('⚠️ Excel 沒有「%s」欄，改用依實際長度分層的 %d 摺。\n', FOLD_COLUMN, K);
            fprintf('   注意：這樣三種方法的切分不保證一致，對照結果會混入切分差異。\n');
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

    %% --- 3. 交叉驗證主迴圈 ---
    annPred = nan(n, N_REPEATS);
    linPred = nan(n, N_REPEATS);
    foldMAE = nan(K, N_REPEATS);

    fprintf('\n訓練中(%d 摺 × %d 次重複)...\n', K, N_REPEATS);
    for r = 1:N_REPEATS
        rng(BASE_SEED + r);
        for k = 1:K
            teIdx = (foldId == k);
            trIdx = ~teIdx & (foldId > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end

            Xtr = Xmat(trIdx, :); Ytr = Y(trIdx);
            Xte = Xmat(teIdx, :);

            % 標準化參數只用 train 摺算，每個特徵各自一組
            mu = mean(Xtr, 1);
            sg = std(Xtr, 0, 1);
            sg(sg == 0) = 1;
            XtrN = (Xtr - mu) ./ sg;
            XteN = (Xte - mu) ./ sg;

            net = fitnet(HIDDEN_SIZE, TRAIN_FCN);
            net.trainParam.showWindow = false;
            net.trainParam.epochs     = MAX_EPOCHS;
            net.divideFcn = 'dividetrain';

            net = train(net, XtrN', Ytr');
            annPred(teIdx, r) = net(XteN')';

            if RUN_LINEAR_BASELINE
                % 多元最小平方(p=1 時等同舊版 polyfit)
                b = [ones(size(Xtr,1),1) Xtr] \ Ytr;
                linPred(teIdx, r) = [ones(size(Xte,1),1) Xte] * b;
            end

            foldMAE(k, r) = mean(abs(annPred(teIdx, r) - Y(teIdx)));
        end
        fprintf('  重複 %d/%d 完成\n', r, N_REPEATS);
    end

    %% --- 4. 逐次重複算指標，再取平均與標準差 ---
    M = emptyMetrics(N_REPEATS);
    for r = 1:N_REPEATS
        M = accumMetrics(M, r, annPred(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
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
    Y_pred = annPred(:, repIdx);
    fprintf('\n逐筆結果採用第 %d 次重複(MAE=%.3f，最接近中位數)。\n', repIdx, M.mae(repIdx));

    Y_pred_offset  = Y_pred - OFFSET_MM;
    residuals      = Y_pred - Y;
    abs_error      = abs(residuals);
    off_residuals  = Y_pred_offset - Y;
    off_abs_error  = abs(off_residuals);
    is_overest     = off_residuals > 0;

    %% --- 6. 組輸出表 ---
    split_label = repmat("test", n, 1);

    ResultsTable = table(ids, split_label, X, Y, ...
        Y_pred, residuals, abs_error, ...
        Y_pred_offset, off_residuals, off_abs_error, is_overest, foldId, ...
        'VariableNames', {'檔名', 'Train_or_Test', '像素長度_px', '實際長度_mm', ...
        '模型預測_mm', '誤差Bias_mm', '絕對誤差_mm', ...
        '校正後預測_mm', '校正後誤差Bias_mm', '校正後絕對誤差_mm', '是否過長', '摺別'});
    if hasDemo
        sexLbl = repmat("F", n, 1);
        sexLbl(sex == 1) = "M";
        ResultsTable.('年齡') = age;
        ResultsTable.('性別') = sexLbl;
    end

    % 'A_Test set表現' 與 'C_臨床安全性評估_Test set校正後' 是 C2 的抓列關鍵字，勿改
    MetricNames = {
        sprintf('A_Test set表現 (out-of-fold, n=%d, ANN, %d次重複平均, 輸入=%s)', n, N_REPEATS, featDesc);
        sprintf('B_重複間標準差 (同一份資料重跑%d次的波動)', N_REPEATS);
        '--- 分界線 ---';
        sprintf('C_臨床安全性評估_Test set校正後 (offset=-%.2fmm)', OFFSET_MM);
        sprintf('D_線性迴歸對照組_未校正 (同一組摺, n=%d)', n);
        sprintf('E_線性迴歸對照組_校正後 (offset=-%.2fmm)', OFFSET_MM)
    };

    nanpad = NaN;
    MAE_val = round([mean(M.mae); std(M.mae); nanpad; mean(M.offMae); ...
                     meanOr(L.mae, RUN_LINEAR_BASELINE); ...
                     meanOr(L.offMae, RUN_LINEAR_BASELINE)], 3);
    RMSE_val = round([mean(M.rmse); std(M.rmse); nanpad; nanpad; ...
                      meanOr(L.rmse, RUN_LINEAR_BASELINE); nanpad], 3);
    R2_val = round([mean(M.r2); std(M.r2); nanpad; nanpad; ...
                    meanOr(L.r2, RUN_LINEAR_BASELINE); nanpad], 3);
    Bias_val = round([mean(M.bias); std(M.bias); nanpad; mean(M.offBias); ...
                      meanOr(L.bias, RUN_LINEAR_BASELINE); ...
                      meanOr(L.offBias, RUN_LINEAR_BASELINE)], 3);
    IdealRate_pct = round([nanpad; std(M.ideal); nanpad; mean(M.ideal); ...
                           nanpad; meanOr(L.ideal, RUN_LINEAR_BASELINE)], 1);
    OverestRate_pct = round([nanpad; std(M.over); nanpad; mean(M.over); ...
                             nanpad; meanOr(L.over, RUN_LINEAR_BASELINE)], 1);

    MetricsTable = table(MetricNames, MAE_val, RMSE_val, R2_val, Bias_val, ...
        IdealRate_pct, OverestRate_pct, ...
        'VariableNames', {'評估範圍_與_嚴格程度', 'MAE_mm', 'RMSE_mm', 'R_Square', ...
        'Mean_Bias_mm', '理想比率_pct', '過長率_pct'});

    SettingsTable = table( ...
        ["量測方法"; "輸入特徵"; "特徵標籤"; "n"; "ANN參數數"; "樣本每參數"], ...
        [string(METHOD); string(featDesc); string(FEAT_TAG); string(n); ...
         string(nParam); string(sprintf('%.2f', n/nParam))], ...
        'VariableNames', {'項目', '值'});

    % --- 全資料線性迴歸：只為解讀效應大小，不是效能 ---
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
    output_filename = sprintf('預測結果與評估指標_kfold_%s%s.xlsx', METHOD, FEAT_TAG);
    if isfile(output_filename), delete(output_filename); end
    writetable(ResultsTable, output_filename, 'Sheet', '所有牙齒預測結果');
    writetable(MetricsTable, output_filename, 'Sheet', '模型評估指標');
    writetable(SettingsTable, output_filename, 'Sheet', '模型設定');
    if ~isempty(CoefTable)
        writetable(CoefTable, output_filename, 'Sheet', '線性係數_全資料');
    end

    fprintf('\n✅ 已匯出：%s\n\n', output_filename);
    disp('=== 評估指標 ===');
    disp(MetricsTable);

    fprintf('\n=== 臨床與安全性評估 ===\n');
    fprintf('量測方法：%s | 輸入：%s | offset = -%.2f mm | 理想容忍 = ±%.2f mm\n', ...
        METHOD, featDesc, OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('[Out-of-fold, n=%d, %d次重複]\n', n, N_REPEATS);
    fprintf('  校正後 MAE  : %.3f ± %.3f mm\n', mean(M.offMae), std(M.offMae));
    fprintf('  理想比率    : %.1f ± %.1f %%\n', mean(M.ideal), std(M.ideal));
    fprintf('  過長率      : %.1f ± %.1f %%  ← 臨床風險較高的指標\n', ...
        mean(M.over), std(M.over));
    fprintf('  摺間 MAE 標準差 : %.3f mm\n', foldSpread);

    if RUN_LINEAR_BASELINE
        fprintf('\n[線性迴歸對照組] MAE %.3f mm vs ANN %.3f mm  →  ', ...
            mean(L.mae), mean(M.mae));
        if mean(L.mae) <= mean(M.mae)
            fprintf('線性不輸 ANN\n');
        else
            fprintf('ANN 勝出 %.3f mm\n', mean(L.mae) - mean(M.mae));
        end
    end

    if ~isempty(CoefTable)
        fprintf('\n=== 全資料線性係數(解讀用，非效能) ===\n');
        disp(CoefTable);
        fprintf('  age 係數 = 每多 1 歲長度變化(mm)；sexM 係數 = 男比女多幾 mm(控制其他變數後)。\n');
        fprintf('  p 值只是參考：n=%d、未校正多重比較。\n', n);
    end

    fprintf('\n👉 對照建議：同一個 METHOD 依序跑\n');
    fprintf('   (USE_AGE,USE_SEX) = (F,F) → (T,F) → (F,T) → (T,T)，比 OOF MAE。\n');
    fprintf('👉 C2 的 METHOD / USE_AGE / USE_SEX 要設成跟這次一樣。\n');
end


%% ============================================================
%  子函式
%  ============================================================

function S = emptyMetrics(N)
    S = struct('mae', nan(N,1), 'rmse', nan(N,1), 'r2', nan(N,1), ...
               'bias', nan(N,1), 'offMae', nan(N,1), 'offBias', nan(N,1), ...
               'ideal', nan(N,1), 'over', nan(N,1));
end


function S = accumMetrics(S, r, Yp, Y, offset, tol)
    res  = Yp - Y;
    S.mae(r)  = mean(abs(res));
    S.rmse(r) = sqrt(mean(res.^2));
    S.bias(r) = mean(res);
    if numel(Y) >= 2
        S.r2(r) = corr(Y, Yp)^2;
    end
    offRes = (Yp - offset) - Y;
    S.offMae(r)  = mean(abs(offRes));
    S.offBias(r) = mean(offRes);
    S.ideal(r)   = mean(abs(offRes) <= tol) * 100;
    S.over(r)    = mean(offRes > 0) * 100;
end


function v = meanOr(x, flag)
    if flag
        v = mean(x, 'omitnan');
    else
        v = NaN;
    end
end


function tag = featTag(useAge, useSex)
% 輸出檔名的特徵後綴；C2 用同一套規則組檔名。都關 = '' (跟舊版同名)
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
% M/男/Male → 1，F/女/Female → 0，其他 → NaN
    t = upper(strtrim(string(v)));
    s = nan(numel(t), 1);
    s(ismember(t, ["M", "MALE", "男"])) = 1;
    s(ismember(t, ["F", "FEMALE", "女"])) = 0;
end
