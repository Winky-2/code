function C1x_modelCompare_v1()
%% C1x_modelCompare_v1.m
% ============================================================
% 以 C1_pixelToMmPredictor_v6 為底，把「模型」變成一個參數。
% 一次 run 會用「完全同一組摺、同一組種子、同一組標準化」
% 把 MODEL_LIST 裡的每個模型都跑過，輸出一張對照總表。
%
% 設計重點（為了讓對照只差模型這一個變因）：
%   1. 分摺沿用 Excel 的「折數fold」欄 → 與 C1/C1g 的切分一致
%   2. mu/sg 在每一摺只算一次，所有模型共用
%   3. 每個模型在每個 (重複, 摺) 都 rng(同一顆種子)，
%      所以在 MODEL_LIST 裡增刪模型，不會改到別的模型的結果
%   4. offset 與理想容忍對所有模型一視同仁
%
% 搭配輸出：模型對照_kfold_<METHOD><FEAT_TAG>.xlsx
% ============================================================

    %% ---------------- 輸入設定區（與 C1 v6 相同）----------------
    METHOD   = 'mask幾何_長邊正規化';   % 'mask幾何_長邊正規化' | '冠寬比例尺'
    filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];

    FOLD_COLUMN  = '折數fold';

    %% ---------------- 特徵設定區 ----------------
    USE_AGE = true;
    USE_SEX = false;
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    %% ---------------- 模型清單 ----------------
    % 可自由增刪、調順序；順序不影響各模型的結果
    MODEL_LIST     = {'constmean','ratio','linear','robust','ann','gpr','svr','lsboost','rf'};
    BASELINE_MODEL = 'constmean';   % ΔMAE 的比較基準

    %% ---------------- 評估設定區 ----------------
    N_REPEATS = 10;
    BASE_SEED = 42;

    HIDDEN_SIZE = 5;          % ANN
    TRAIN_FCN   = 'trainbr';
    MAX_EPOCHS  = 200;

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
        error('輸入檔沒有「%s」「%s」欄，請重跑 B4。', AGE_COLUMN, SEX_COLUMN);
    end
    if hasDemo
        age = toNum(data.(AGE_COLUMN));
        sex = parseSex(data.(SEX_COLUMN));
    else
        age = nan(height(data),1); sex = nan(height(data),1);
    end

    valid = ~isnan(X) & ~isnan(Y);
    if hasDemo, valid = valid & ~isnan(age) & ~isnan(sex); end
    if any(~valid)
        fprintf('有 %d 筆含缺值，已剔除。\n', sum(~valid));
        ids = ids(valid); Y = Y(valid); X = X(valid);
        age = age(valid); sex = sex(valid); data = data(valid,:);
    end
    n = numel(Y);

    % --- 特徵矩陣（第 1 欄固定是像素長度）---
    Xmat = X; featNames = {'像素長度'};
    if USE_AGE, Xmat = [Xmat age]; featNames{end+1} = '年齡'; end
    if USE_SEX, Xmat = [Xmat sex]; featNames{end+1} = '性別(M=1)'; end
    p = size(Xmat,2);
    featDesc = strjoin(featNames, ' + ');
    FEAT_TAG = featTag(USE_AGE, USE_SEX);

    fprintf('=== C1x 模型對照：%s ===\n', METHOD);
    fprintf('n=%d，輸入特徵：%s\n', n, featDesc);
    fprintf('模型：%s\n', strjoin(MODEL_LIST, ', '));

    %% --- 2. 分摺（沿用 Excel，確保與 C1/C1g 同一組切分）---
    if ismember(FOLD_COLUMN, vn)
        foldId = double(data.(FOLD_COLUMN));
        fprintf('沿用 Excel 的「%s」欄分摺。\n', FOLD_COLUMN);
    else
        K = 5; [~,ord] = sort(Y); foldId = zeros(n,1);
        foldId(ord) = mod(0:n-1, K) + 1;
        fprintf('Excel 沒有「%s」欄，改用依實際長度分層的 %d 摺（與 C1 不保證一致）。\n', FOLD_COLUMN, K);
    end
    K = max(foldId);
    fprintf('共 %d 摺，每摺樣本數：', K);
    fprintf('%d ', accumarray(foldId,1)); fprintf('\n');

    %% --- 3. 主迴圈：摺內一次切分，所有模型共用 ---
    nM = numel(MODEL_LIST);
    P  = struct();                      % 每個模型的 out-of-fold 預測
    for i = 1:nM, P.(MODEL_LIST{i}) = nan(n, N_REPEATS); end
    failMsg = struct();

    fprintf('\n訓練中（%d 摺 × %d 重複 × %d 模型 = %d 次擬合）...\n', ...
        K, N_REPEATS, nM, K*N_REPEATS*nM);
    t0 = tic;
    for r = 1:N_REPEATS
        for k = 1:K
            teIdx = (foldId == k);
            trIdx = ~teIdx & (foldId > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end

            Xtr = Xmat(trIdx,:); Ytr = Y(trIdx); Xte = Xmat(teIdx,:);
            mu = mean(Xtr,1); sg = std(Xtr,0,1); sg(sg==0) = 1;
            XtrN = (Xtr-mu)./sg; XteN = (Xte-mu)./sg;

            for i = 1:nM
                m = MODEL_LIST{i};
                % 同一顆種子給每個模型 → 與清單順序無關
                rng(BASE_SEED + 1000*r + 10*k);
                try
                    yhat = fitPredict(m, Xtr, Ytr, Xte, XtrN, XteN, ...
                                      HIDDEN_SIZE, TRAIN_FCN, MAX_EPOCHS);
                catch ME
                    yhat = nan(sum(teIdx),1);
                    if ~isfield(failMsg, m), failMsg.(m) = ME.message; end
                end
                P.(m)(teIdx, r) = yhat;
            end
        end
        fprintf('  重複 %d/%d 完成（累計 %.0f 秒）\n', r, N_REPEATS, toc(t0));
    end

    fn = fieldnames(failMsg);
    for i = 1:numel(fn)
        fprintf('[略過] %s 擬合失敗：%s\n', fn{i}, failMsg.(fn{i}));
    end

    %% --- 4. 逐模型算指標（每次重複算一次，再取 mean±std）---
    Mall = struct();
    for i = 1:nM
        m = MODEL_LIST{i};
        S = emptyMetrics(N_REPEATS);
        for r = 1:N_REPEATS
            S = accumMetrics(S, r, P.(m)(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
        end
        % 摺間 MAE 標準差（校正後）
        fm = nan(K, N_REPEATS);
        for r = 1:N_REPEATS
            for k = 1:K
                idx = (foldId == k) & ~isnan(P.(m)(:,r));
                if any(idx)
                    fm(k,r) = mean(abs((P.(m)(idx,r) - OFFSET_MM) - Y(idx)));
                end
            end
        end
        S.foldSpread = mean(std(fm, 0, 1, 'omitnan'), 'omitnan');
        Mall.(m) = S;
    end

    %% --- 5. 對照總表 ---
    base = Mall.(BASELINE_MODEL).offMae;      % 逐重複配對比較
    rows = cell(nM,1);
    offMae_m = nan(nM,1); offMae_s = nan(nM,1);
    mae_m = nan(nM,1); rmse_m = nan(nM,1); bias_m = nan(nM,1);
    r_m = nan(nM,1); r2_m = nan(nM,1); ideal_m = nan(nM,1);
    dmae_m = nan(nM,1); dmae_s = nan(nM,1); spread = nan(nM,1);
    usedFeat = cell(nM,1);
    for i = 1:nM
        m = MODEL_LIST{i}; S = Mall.(m);
        rows{i}     = modelLabel(m);
        offMae_m(i) = mean(S.offMae,'omitnan');  offMae_s(i) = std(S.offMae,'omitnan');
        mae_m(i)    = mean(S.mae,'omitnan');     rmse_m(i)   = mean(S.rmse,'omitnan');
        bias_m(i)   = mean(S.offBias,'omitnan');
        r_m(i)      = mean(S.r,'omitnan');       r2_m(i)     = mean(S.r2,'omitnan');
        ideal_m(i)  = mean(S.ideal,'omitnan');
        d = S.offMae - base;
        dmae_m(i)   = mean(d,'omitnan');         dmae_s(i)   = std(d,'omitnan');
        spread(i)   = S.foldSpread;
        usedFeat{i} = usesFeatures(m, featDesc);
    end

    CompareTable = table(string(MODEL_LIST(:)), string(rows), string(usedFeat), ...
        round(offMae_m,3), round(offMae_s,3), round(mae_m,3), round(rmse_m,3), ...
        round(bias_m,3), round(r_m,3), round(r2_m,3), round(ideal_m,1), ...
        round(dmae_m,3), round(dmae_s,3), round(spread,3), ...
        'VariableNames', {'模型','說明','用到的特徵', ...
        '校正後MAE_mm','校正後MAE_SD','未校正MAE_mm','未校正RMSE_mm', ...
        '校正後Bias_mm','R_Pearson','R_Square','理想比率_pct', ...
        ['ΔMAE_vs_' BASELINE_MODEL], 'ΔMAE_SD', '摺間MAE_SD'});
    CompareTable = sortrows(CompareTable, '校正後MAE_mm');

    %% --- 6. 逐筆預測（每個模型取各自的代表性重複）---
    Per = table(ids, X, Y, foldId, 'VariableNames', ...
        {'檔名','像素長度_px','實際長度_mm','摺別'});
    for i = 1:nM
        m = MODEL_LIST{i}; S = Mall.(m);
        if all(isnan(S.offMae)), continue; end
        [~, ri] = min(abs(S.offMae - median(S.offMae,'omitnan')));
        yp = P.(m)(:,ri) - OFFSET_MM;
        Per.(['預測_' m])     = round(yp, 3);
        Per.(['絕對誤差_' m]) = round(abs(yp - Y), 3);
    end

    SettingsTable = table( ...
        ["量測方法";"輸入特徵";"n";"摺數";"重複次數";"offset_mm";"理想容忍_mm";"基準模型";"腳本"], ...
        [string(METHOD); string(featDesc); string(n); string(K); string(N_REPEATS); ...
         string(OFFSET_MM); string(IDEAL_TOLERANCE_MM); string(BASELINE_MODEL); "C1x_v1"], ...
        'VariableNames', {'項目','值'});

    %% --- 7. 匯出 ---
    outName = sprintf('模型對照_kfold_%s%s.xlsx', METHOD, FEAT_TAG);
    if isfile(outName), delete(outName); end
    writetable(CompareTable,  outName, 'Sheet', '模型對照總表');
    writetable(Per,           outName, 'Sheet', '逐筆預測');
    writetable(SettingsTable, outName, 'Sheet', '設定');

    fprintf('\n已匯出：%s\n\n', outName);
    disp('=== 模型對照（依校正後 MAE 排序）===');
    disp(CompareTable(:, {'模型','校正後MAE_mm','校正後MAE_SD', ...
        ['ΔMAE_vs_' BASELINE_MODEL],'ΔMAE_SD','R_Square','理想比率_pct'}));
    fprintf(['判讀：ΔMAE 為負才代表贏過 %s；若 |ΔMAE| 小於 ΔMAE_SD，\n' ...
             '      這個差距就落在重複間的雜訊裡，不要當成實質改善。\n'], BASELINE_MODEL);
end


%% ============================================================
%  模型擬合＋預測：要加模型就在這裡加一個 case
%  ============================================================
function yhat = fitPredict(m, Xtr, Ytr, Xte, XtrN, XteN, H, TRAINFCN, EPOCHS)
    switch lower(m)
        case 'constmean'      % 天真基準：只用訓練摺的平均
            yhat = repmat(mean(Ytr), size(Xte,1), 1);

        case 'ratio'          % 解析比例尺 mm = k * px（1 個參數，不用學習）
            kk   = median(Ytr ./ Xtr(:,1));
            yhat = Xte(:,1) * kk;

        case 'linear'
            b    = [ones(size(Xtr,1),1) Xtr] \ Ytr;
            yhat = [ones(size(Xte,1),1) Xte] * b;

        case 'robust'         % 抗離群線性（Stats Toolbox）
            b    = robustfit(Xtr, Ytr);
            yhat = [ones(size(Xte,1),1) Xte] * b;

        case 'ann'            % 與 C1 v6 設定一致
            net = fitnet(H, TRAINFCN);
            net.trainParam.showWindow = false;
            net.trainParam.epochs     = EPOCHS;
            net.divideFcn = 'dividetrain';
            net  = train(net, XtrN', Ytr');
            yhat = net(XteN')';

        case 'gpr'            % 注意 BasisFunction='linear'，見下方說明
            mdl  = fitrgp(XtrN, Ytr, 'KernelFunction','squaredexponential', ...
                          'BasisFunction','linear', 'Standardize', false);
            yhat = predict(mdl, XteN);

        case 'svr'
            mdl  = fitrsvm(XtrN, Ytr, 'KernelFunction','rbf', ...
                           'KernelScale','auto', 'Standardize', false);
            yhat = predict(mdl, XteN);

        case 'lsboost'
            mdl  = fitrensemble(XtrN, Ytr, 'Method','LSBoost', ...
                                'NumLearningCycles',100, 'LearnRate',0.05);
            yhat = predict(mdl, XteN);

        case 'rf'
            mdl  = TreeBagger(200, XtrN, Ytr, 'Method','regression', ...
                              'MinLeafSize', 5);
            yhat = predict(mdl, XteN);

        case 'spline'         % 只支援單特徵
            if size(Xtr,2) > 1
                error('spline 只支援單特徵，請把 USE_AGE/USE_SEX 關掉。');
            end
            f    = fit(Xtr(:,1), Ytr, 'smoothingspline', 'SmoothingParam', 0.9);
            yhat = feval(f, Xte(:,1));

        otherwise
            error('未知的模型代號：%s', m);
    end
    yhat = yhat(:);
end


function s = modelLabel(m)
    switch lower(m)
        case 'constmean', s = '常數均值（天真基準，不用影像）';
        case 'ratio',     s = '解析比例尺 mm=k*px（1 參數）';
        case 'linear',    s = '線性迴歸';
        case 'robust',    s = '穩健線性迴歸';
        case 'ann',       s = 'ANN fitnet(5, trainbr)';
        case 'gpr',       s = 'GPR SE 核 + 線性基底';
        case 'svr',       s = 'SVR RBF 核';
        case 'lsboost',   s = 'Boosted trees (LSBoost)';
        case 'rf',        s = '隨機森林 (TreeBagger 200)';
        case 'spline',    s = '平滑樣條';
        otherwise,        s = m;
    end
end


function s = usesFeatures(m, featDesc)
% 標出哪些模型其實沒吃到 年齡/性別，避免誤讀對照表
    switch lower(m)
        case 'constmean', s = '無（只用訓練摺平均）';
        case {'ratio','spline'}, s = '僅像素長度';
        otherwise, s = featDesc;
    end
end


%% ---------------- 指標子函式（與 C1 v6 同定義）----------------
function S = emptyMetrics(N)
    S = struct('mae', nan(N,1), 'rmse', nan(N,1), 'r', nan(N,1), 'r2', nan(N,1), ...
               'bias', nan(N,1), 'offMae', nan(N,1), 'offRmse', nan(N,1), ...
               'offBias', nan(N,1), 'ideal', nan(N,1), 'foldSpread', NaN);
end

function S = accumMetrics(S, r, Yp, Y, offset, tol)
    ok = ~isnan(Yp) & ~isnan(Y);
    if ~any(ok), return; end
    Yp = Yp(ok); Yv = Y(ok);

    res = Yp - Yv;
    S.mae(r)  = mean(abs(res));
    S.rmse(r) = sqrt(mean(res.^2));
    S.bias(r) = mean(res);
    if numel(Yv) >= 2 && std(Yp) > 0
        S.r(r)  = corr(Yv, Yp);
        S.r2(r) = S.r(r)^2;
    else
        S.r(r) = NaN; S.r2(r) = NaN;   % 常數均值預測沒有相關係數
    end

    offRes = (Yp - offset) - Yv;
    S.offMae(r)  = mean(abs(offRes));
    S.offRmse(r) = sqrt(mean(offRes.^2));
    S.offBias(r) = mean(offRes);
    S.ideal(r)   = mean(abs(offRes) <= tol) * 100;
end

function tag = featTag(useAge, useSex)
    parts = {};
    if useAge, parts{end+1} = '年齡'; end
    if useSex, parts{end+1} = '性別'; end
    if isempty(parts), tag = ''; else, tag = ['_' strjoin(parts,'')]; end
end

function v = toNum(x)
    if iscell(x) || isstring(x), v = str2double(string(x)); else, v = double(x); end
end

function s = parseSex(v)
    t = upper(strtrim(string(v)));
    s = nan(numel(t),1);
    s(ismember(t, ["M","MALE","男"]))   = 1;
    s(ismember(t, ["F","FEMALE","女"])) = 0;
end
