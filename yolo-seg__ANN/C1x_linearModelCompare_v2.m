function C1x_modelCompare_v2()
%% C1x_modelCompare_v2.m
% ============================================================
% v1 的九個模型 + 四個新的迴歸模型，同一組摺、同一組種子、同一份標準化。
%
% 新增（相對 v1）：
%   deming    總最小平方 / Deming：承認像素長度 X 本身有量測誤差
%   l1lin     L1（中位數）迴歸：損失改成 MAE，與主要指標對齊
%   powerlaw  冪律 mm = a*px^b（log-log 擬合 + Duan smearing 修正）
%   lme       混合效應：感測器當隨機效應（需要感測器欄位，見 SENSOR_COLUMN）
%
% 另外獨立一區（不進對照總表）：
%   序位邏輯迴歸 + 各迴歸模型轉三分類的準確率對照
%   需要 CLASS_EDGES，預設關閉，見下方說明。
%
% 設計與 v1 相同：摺內只切一次、mu/sg 只算一次、每個模型拿同一顆種子，
% 所以在 MODEL_LIST 裡增刪模型不會動到其他模型的結果。
% ============================================================

    %% ---------------- 輸入設定區 ----------------
    METHOD   = 'mask幾何_長邊正規化';   % 'mask幾何_長邊正規化' | '冠寬比例尺'
    filename = ['根管填充物像素長度_已配對_11n_' METHOD '.xlsx'];

    FOLD_COLUMN = '折數fold';

    %% ---------------- 特徵設定區 ----------------
    USE_AGE = true;
    USE_SEX = false;
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    % 感測器欄位（lme 用）。留空會自動從下列候選名找；都找不到就自動略過 lme。
    SENSOR_COLUMN = '';
    SENSOR_CANDIDATES = {'感測器','感測器群組','sensor','機台','影像來源','資料來源','來源'};

    %% ---------------- 模型清單 ----------------
    MODEL_LIST     = {'constmean','ratio','linear','robust','deming','l1lin', ...
                      'powerlaw','lme','ann','gpr','svr','lsboost','rf'};
    BASELINE_MODEL = 'constmean';

    %% ---------------- 模型參數 ----------------
    DEMING_LAMBDA   = 1;      % var(ε_y)/var(ε_x)；=1 即正交迴歸。未知，預設 1
    POWERLAW_SMEAR  = true;   % log 空間擬合的反轉換偏誤修正（Duan smearing）
    L1_ITERS        = 60;     % L1 用 IRLS 逼近的迭代次數

    HIDDEN_SIZE = 5;          % ANN
    TRAIN_FCN   = 'trainbr';
    MAX_EPOCHS  = 200;

    %% ---------------- 評估設定區 ----------------
    N_REPEATS = 10;
    BASE_SEED = 42;

    OFFSET_MM          = 0.5;
    IDEAL_TOLERANCE_MM = 1.0;

    %% ---------------- 三分類設定（序位邏輯迴歸）----------------
    % 以 mm 切成三類的邊界，例如 [20 22] = (<20 | 20~22 | >22)。
    % ⚠️ 這兩個數字必須由臨床定義，不要用資料分位數硬切；留空 [] 則整區略過。
    CLASS_EDGES  = [];
    CLASS_LABELS = {'underfill','adequate','overfill'};

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

    % --- 感測器欄位（lme 用）---
    sensorCol = SENSOR_COLUMN;
    if isempty(sensorCol)
        for i = 1:numel(SENSOR_CANDIDATES)
            if ismember(SENSOR_CANDIDATES{i}, vn)
                sensorCol = SENSOR_CANDIDATES{i}; break;
            end
        end
    end
    if ~isempty(sensorCol) && ismember(sensorCol, vn)
        sensor = categorical(string(data.(sensorCol)));
    else
        sensor = categorical(repmat("NA", height(data), 1));
    end
    hasSensor = numel(categories(removecats(sensor))) >= 2;

    valid = ~isnan(X) & ~isnan(Y);
    if hasDemo, valid = valid & ~isnan(age) & ~isnan(sex); end
    if any(~valid)
        fprintf('有 %d 筆含缺值，已剔除。\n', sum(~valid));
        ids = ids(valid); Y = Y(valid); X = X(valid);
        age = age(valid); sex = sex(valid); sensor = sensor(valid);
        data = data(valid,:);
    end
    n = numel(Y);

    % --- 特徵矩陣（第 1 欄固定是像素長度）---
    Xmat = X; featNames = {'像素長度'};
    if USE_AGE, Xmat = [Xmat age]; featNames{end+1} = '年齡'; end
    if USE_SEX, Xmat = [Xmat sex]; featNames{end+1} = '性別(M=1)'; end
    featDesc = strjoin(featNames, ' + ');
    FEAT_TAG = featTag(USE_AGE, USE_SEX);

    % --- lme 沒有感測器欄位就移出清單 ---
    if ismember('lme', MODEL_LIST) && ~hasSensor
        MODEL_LIST = MODEL_LIST(~strcmp(MODEL_LIST, 'lme'));
        if isempty(sensorCol)
            fprintf(['[略過 lme] 輸入檔裡找不到感測器欄位。\n' ...
                     '   候選欄名：%s\n' ...
                     '   請在 B4 的輸出加上感測器分組欄（B6_check_sensor_groups.py 可以產生），\n' ...
                     '   或把 SENSOR_COLUMN 設成實際欄名。\n'], strjoin(SENSOR_CANDIDATES, ', '));
        else
            fprintf('[略過 lme] 「%s」欄只有一個群組，隨機效應沒有意義。\n', sensorCol);
        end
    elseif hasSensor
        fprintf('感測器欄位：「%s」，共 %d 組。\n', sensorCol, numel(categories(removecats(sensor))));
    end

    fprintf('=== C1x v2 模型對照：%s ===\n', METHOD);
    fprintf('n=%d，輸入特徵：%s\n', n, featDesc);
    fprintf('模型：%s\n', strjoin(MODEL_LIST, ', '));

    %% --- 2. 分摺 ---
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

    %% --- 3. 主迴圈 ---
    nM = numel(MODEL_LIST);
    P  = struct();
    for i = 1:nM, P.(MODEL_LIST{i}) = nan(n, N_REPEATS); end
    failMsg = struct();

    cfg = struct('H', HIDDEN_SIZE, 'trainFcn', TRAIN_FCN, 'epochs', MAX_EPOCHS, ...
                 'demingLambda', DEMING_LAMBDA, 'powerSmear', POWERLAW_SMEAR, ...
                 'l1Iters', L1_ITERS);

    fprintf('\n訓練中（%d 摺 × %d 重複 × %d 模型 = %d 次擬合）...\n', ...
        K, N_REPEATS, nM, K*N_REPEATS*nM);
    t0 = tic;
    for r = 1:N_REPEATS
        for k = 1:K
            teIdx = (foldId == k);
            trIdx = ~teIdx & (foldId > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end

            S = struct();
            S.Xtr = Xmat(trIdx,:);  S.Ytr = Y(trIdx);  S.Xte = Xmat(teIdx,:);
            mu = mean(S.Xtr,1); sg = std(S.Xtr,0,1); sg(sg==0) = 1;
            S.XtrN = (S.Xtr-mu)./sg;  S.XteN = (S.Xte-mu)./sg;
            S.sensorTr = sensor(trIdx);  S.sensorTe = sensor(teIdx);
            S.cfg = cfg;

            for i = 1:nM
                m = MODEL_LIST{i};
                rng(BASE_SEED + 1000*r + 10*k);     % 與清單順序無關
                try
                    yhat = fitPredict(m, S);
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

    %% --- 4. 逐模型算指標 ---
    Mall = struct();
    for i = 1:nM
        m = MODEL_LIST{i};
        Sm = emptyMetrics(N_REPEATS);
        for r = 1:N_REPEATS
            Sm = accumMetrics(Sm, r, P.(m)(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
        end
        fm = nan(K, N_REPEATS);
        for r = 1:N_REPEATS
            for k = 1:K
                idx = (foldId == k) & ~isnan(P.(m)(:,r));
                if any(idx)
                    fm(k,r) = mean(abs((P.(m)(idx,r) - OFFSET_MM) - Y(idx)));
                end
            end
        end
        Sm.foldSpread = mean(std(fm, 0, 1, 'omitnan'), 'omitnan');
        Mall.(m) = Sm;
    end

    %% --- 5. 對照總表 ---
    base = Mall.(BASELINE_MODEL).offMae;
    rows = cell(nM,1); usedFeat = cell(nM,1);
    offMae_m = nan(nM,1); offMae_s = nan(nM,1);
    mae_m = nan(nM,1); rmse_m = nan(nM,1); bias_m = nan(nM,1);
    rmseOff_m = nan(nM,1);
    r_m = nan(nM,1); r2_m = nan(nM,1); ideal_m = nan(nM,1);
    dmae_m = nan(nM,1); dmae_s = nan(nM,1); spread = nan(nM,1);
    for i = 1:nM
        m = MODEL_LIST{i}; Sm = Mall.(m);
        rows{i}     = modelLabel(m);
        offMae_m(i) = mean(Sm.offMae,'omitnan');  offMae_s(i) = std(Sm.offMae,'omitnan');
        mae_m(i)    = mean(Sm.mae,'omitnan');     rmse_m(i)   = mean(Sm.rmse,'omitnan');
        rmseOff_m(i) = mean(Sm.offRmse,'omitnan');
        bias_m(i)   = mean(Sm.offBias,'omitnan');
        r_m(i)      = mean(Sm.r,'omitnan');       r2_m(i)     = mean(Sm.r2,'omitnan');
        ideal_m(i)  = mean(Sm.ideal,'omitnan');
        d = Sm.offMae - base;
        dmae_m(i)   = mean(d,'omitnan');          dmae_s(i)   = std(d,'omitnan');
        spread(i)   = Sm.foldSpread;
        usedFeat{i} = usesFeatures(m, featDesc, sensorCol);
    end

    % 報告用指標。前三欄 MAE_mm / RMSE_mm / Mean_Bias_mm 都是「校正後」的值，
    % 與 C2s 圖上文字框定義一致；另附未校正的 MAE 與 RMSE 供對照。
    % R / R^2 不受 offset 影響，校正前後同值，所以只有一欄。
    CompareTable = table(string(MODEL_LIST(:)), ...
        round(offMae_m,3), round(rmseOff_m,3), round(bias_m,3), ...
        round(mae_m,3), round(rmse_m,3), ...
        round(r_m,3), round(r2_m,3), round(ideal_m,1), ...
        'VariableNames', {'模型','MAE_mm','RMSE_mm','Mean_Bias_mm', ...
        '未校正MAE_mm','未校正RMSE_mm','R_Pearson','R_Square','理想比率_pct'});
    CompareTable = sortrows(CompareTable, 'MAE_mm');

    %% --- 6. 逐筆預測 ---
    Per = table(ids, X, Y, foldId, 'VariableNames', ...
        {'檔名','像素長度_px','實際長度_mm','摺別'});
    repIdxOf = struct();
    for i = 1:nM
        m = MODEL_LIST{i}; Sm = Mall.(m);
        if all(isnan(Sm.offMae)), continue; end
        [~, ri] = min(abs(Sm.offMae - median(Sm.offMae,'omitnan')));
        repIdxOf.(m) = ri;
        yp = P.(m)(:,ri) - OFFSET_MM;
        Per.(['預測_' m])     = round(yp, 3);
        Per.(['絕對誤差_' m]) = round(abs(yp - Y), 3);
    end

    SettingsTable = table( ...
        ["量測方法";"輸入特徵";"n";"摺數";"重複次數";"offset_mm";"理想容忍_mm"; ...
         "基準模型";"Deming_lambda";"感測器欄位";"腳本"], ...
        [string(METHOD); string(featDesc); string(n); string(K); string(N_REPEATS); ...
         string(OFFSET_MM); string(IDEAL_TOLERANCE_MM); string(BASELINE_MODEL); ...
         string(DEMING_LAMBDA); string(ternary(isempty(sensorCol),"（無）",string(sensorCol))); ...
         "C1x_v2"], ...
        'VariableNames', {'項目','值'});

    %% --- 7. 三分類（序位邏輯迴歸 + 各迴歸模型轉類別）---
    ClassTable = [];
    if ~isempty(CLASS_EDGES)
        ClassTable = runOrdinalBlock(Xmat, Y, foldId, K, N_REPEATS, BASE_SEED, ...
                        P, MODEL_LIST, repIdxOf, OFFSET_MM, CLASS_EDGES, CLASS_LABELS);
    else
        fprintf(['\n[略過三分類] CLASS_EDGES 是空的。\n' ...
                 '   要跑序位邏輯迴歸，把 CLASS_EDGES 設成臨床定義的 mm 切點，例如 [20 22]。\n' ...
                 '   ⚠️ 這兩個數字請跟指導醫師確認，不要用資料分位數硬切。\n']);
    end

    %% --- 8. 匯出 ---
    outName = sprintf('模型對照v2_kfold_%s%s.xlsx', METHOD, FEAT_TAG);
    if isfile(outName), delete(outName); end
    writetable(CompareTable,  outName, 'Sheet', '模型對照總表');
    writetable(Per,           outName, 'Sheet', '逐筆預測');
    writetable(SettingsTable, outName, 'Sheet', '設定');
    if ~isempty(ClassTable)
        writetable(ClassTable, outName, 'Sheet', '三分類對照');
    end

    fprintf('\n已匯出：%s\n\n', outName);
    disp('=== 模型對照（校正後指標，依 MAE 排序）===');
    disp(CompareTable);
    fprintf('全部為 out-of-fold，%d 次重複平均；offset = -%.2f mm，理想容忍 ±%.2f mm。\n', ...
            N_REPEATS, OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('MAE_mm / RMSE_mm / Mean_Bias_mm 為校正後；另附未校正 MAE 與 RMSE 對照。\n');
    fprintf('R / R^2 不受 offset 影響，校正前後同值。\n');
    fprintf('要宣稱 A 模型贏 B 模型，請用 C2x 的逐樣本配對檢定，不要只看 MAE 差。\n');
end


%% ============================================================
%  模型擬合＋預測：要加模型就在這裡加一個 case
%  ============================================================
function yhat = fitPredict(m, S)
    Xtr = S.Xtr; Ytr = S.Ytr; Xte = S.Xte;
    XtrN = S.XtrN; XteN = S.XteN; c = S.cfg;

    switch lower(m)
        case 'constmean'      % 天真基準：只用訓練摺的平均
            yhat = repmat(mean(Ytr), size(Xte,1), 1);

        case 'ratio'          % 解析比例尺 mm = k*px（1 個參數）
            kk   = median(Ytr ./ Xtr(:,1));
            yhat = Xte(:,1) * kk;

        case 'linear'
            b    = [ones(size(Xtr,1),1) Xtr] \ Ytr;
            yhat = [ones(size(Xte,1),1) Xte] * b;

        case 'robust'         % 抗離群線性（Stats Toolbox）
            b    = robustfit(Xtr, Ytr);
            yhat = [ones(size(Xte,1),1) Xte] * b;

        case 'deming'
            % 總最小平方 / Deming：X 也有量測誤差時，OLS 斜率會被低估
            % （迴歸稀釋），預測會往均值壓縮。lambda = var(ε_y)/var(ε_x)。
            % 只用像素長度這一個特徵。
            lam = c.demingLambda;
            xx  = Xtr(:,1); yy = Ytr;
            sxx = var(xx); syy = var(yy); cc = cov(xx, yy); sxy = cc(1,2);
            if abs(sxy) < eps
                b1 = 0;
            else
                b1 = (syy - lam*sxx + sqrt((syy - lam*sxx)^2 + 4*lam*sxy^2)) / (2*sxy);
            end
            b0   = mean(yy) - b1*mean(xx);
            yhat = b0 + b1*Xte(:,1);

        case 'l1lin'
            % L1（中位數）迴歸，以 IRLS 逼近：w_i = 1/|r_i|，重複加權最小平方。
            % 比 fminsearch 穩定，且多特徵時收斂可靠。
            A = [ones(size(Xtr,1),1) Xtr];
            b = A \ Ytr;
            for it = 1:c.l1Iters
                r  = Ytr - A*b;
                w  = 1 ./ max(abs(r), 1e-4);
                sw = sqrt(w);
                b  = (sw .* A) \ (sw .* Ytr);
            end
            yhat = [ones(size(Xte,1),1) Xte] * b;

        case 'powerlaw'
            % mm = a * px^b，在 log-log 空間用最小平方擬合。
            % 反轉換會系統性低估，所以用 Duan smearing 修正。只用像素長度。
            xx = Xtr(:,1); 
            if any(xx <= 0) || any(Ytr <= 0)
                error('powerlaw 需要像素長度與長度皆為正值。');
            end
            A    = [ones(numel(xx),1) log(xx)];
            b    = A \ log(Ytr);
            resid = log(Ytr) - A*b;
            smear = 1;
            if c.powerSmear, smear = mean(exp(resid)); end
            yhat = smear * exp(b(1) + b(2)*log(Xte(:,1)));

        case 'lme'
            % 混合效應：像素長度（與其他特徵）為固定效應，感測器為隨機截距。
            % 測試摺的感測器若沒在訓練摺出現過，退回只用固定效應。
            tv = array2table(Xtr, 'VariableNames', featVarNames(size(Xtr,2)));
            tv.mm = Ytr;  tv.sensor = S.sensorTr;
            formula = sprintf('mm ~ %s + (1|sensor)', ...
                strjoin(featVarNames(size(Xtr,2)), ' + '));
            mdl = fitlme(tv, formula);
            te = array2table(Xte, 'VariableNames', featVarNames(size(Xte,2)));
            te.sensor = S.sensorTe;
            te.mm = zeros(size(Xte,1),1);          % predict 需要響應欄佔位
            try
                yhat = predict(mdl, te);
            catch
                yhat = predict(mdl, te, 'Conditional', false);
            end

        case 'ann'
            net = fitnet(c.H, c.trainFcn);
            net.trainParam.showWindow = false;
            net.trainParam.epochs     = c.epochs;
            net.divideFcn = 'dividetrain';
            net  = train(net, XtrN', Ytr');
            yhat = net(XteN')';

        case 'gpr'
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

        otherwise
            error('未知的模型代號：%s', m);
    end
    yhat = yhat(:);
end


%% ============================================================
%  三分類區塊：序位邏輯迴歸 + 各迴歸模型的 mm 預測轉類別
%  ============================================================
function T = runOrdinalBlock(Xmat, Y, foldId, K, N_REPEATS, BASE_SEED, ...
                             P, MODEL_LIST, repIdxOf, OFFSET_MM, edges, labels)
    n = numel(Y);
    cls = mm2class(Y, edges);
    fprintf('\n=== 三分類（切點 %s）===\n', mat2str(edges));
    for g = 1:numel(labels)
        fprintf('  %-10s n=%d\n', labels{g}, sum(cls == g));
    end
    if numel(unique(cls)) < 2
        fprintf('  只有一個類別，略過序位模型。\n'); T = []; return;
    end

    % --- 序位邏輯迴歸（同一組摺）---
    ordPred = nan(n, N_REPEATS);
    ordFail = '';
    for r = 1:N_REPEATS
        for k = 1:K
            teIdx = (foldId == k); trIdx = ~teIdx & (foldId > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end
            Xtr = Xmat(trIdx,:); Xte = Xmat(teIdx,:);
            mu = mean(Xtr,1); sg = std(Xtr,0,1); sg(sg==0)=1;
            XtrN = (Xtr-mu)./sg; XteN = (Xte-mu)./sg;
            ctr = cls(trIdx);
            rng(BASE_SEED + 1000*r + 10*k);
            try
                B  = mnrfit(XtrN, ctr, 'Model', 'ordinal');
                pr = mnrval(B, XteN, 'Model', 'ordinal');
                [~, ch] = max(pr, [], 2);
                ordPred(teIdx, r) = ch;
            catch ME
                if isempty(ordFail), ordFail = ME.message; end
            end
        end
    end
    if ~isempty(ordFail)
        fprintf('  [序位模型失敗] %s\n', ordFail);
        fprintf('  如果是 mnrfit 不存在，改用 fitmnr（新版 MATLAB）。\n');
    end

    % --- 對照：各迴歸模型的 mm 預測轉三分類 ---
    names = strings(0); acc = []; note = strings(0);
    if ~all(isnan(ordPred(:)))
        a = nan(N_REPEATS,1);
        for r = 1:N_REPEATS
            ok = ~isnan(ordPred(:,r));
            if any(ok), a(r) = mean(ordPred(ok,r) == cls(ok)) * 100; end
        end
        names(end+1,1) = "ordinal_logit";
        acc(end+1,1)   = mean(a, 'omitnan');
        note(end+1,1)  = "直接做三分類";
    end
    for i = 1:numel(MODEL_LIST)
        m = MODEL_LIST{i};
        if ~isfield(repIdxOf, m), continue; end
        a = nan(N_REPEATS,1);
        for r = 1:N_REPEATS
            yp = P.(m)(:,r) - OFFSET_MM;
            ok = ~isnan(yp);
            if any(ok)
                a(r) = mean(mm2class(yp(ok), edges) == cls(ok)) * 100;
            end
        end
        names(end+1,1) = string(m);                 %#ok<AGROW>
        acc(end+1,1)   = mean(a, 'omitnan');        %#ok<AGROW>
        note(end+1,1)  = "mm 預測後切類";            %#ok<AGROW>
    end

    % --- 多數類基準 ---
    names(end+1,1) = "majority_class";
    acc(end+1,1)   = max(histcounts(cls, 0.5:1:(numel(labels)+0.5))) / n * 100;
    note(end+1,1)  = "全猜最多的那一類";

    T = sortrows(table(names, round(acc,1), note, ...
        'VariableNames', {'模型','三分類準確率_pct','作法'}), ...
        '三分類準確率_pct', 'descend');
    disp(T);
    fprintf('判讀：贏不過 majority_class 就等於沒有分類能力。\n');
    fprintf('      準確率沒有考慮類別不平衡，報告時建議一併附混淆矩陣。\n');
end


function c = mm2class(v, edges)
    c = ones(numel(v), 1);
    for e = 1:numel(edges)
        c = c + (v(:) > edges(e));
    end
end


%% ---------------- 標籤 ----------------
function s = modelLabel(m)
    switch lower(m)
        case 'constmean', s = '常數均值（天真基準，不用影像）';
        case 'ratio',     s = '解析比例尺 mm=k*px（1 參數）';
        case 'linear',    s = '線性迴歸（OLS）';
        case 'robust',    s = '穩健線性迴歸（IRLS）';
        case 'deming',    s = 'Deming / 總最小平方（X 含誤差）';
        case 'l1lin',     s = 'L1 中位數迴歸（損失對齊 MAE）';
        case 'powerlaw',  s = '冪律 mm=a*px^b（log-log + smearing）';
        case 'lme',       s = '混合效應（感測器為隨機截距）';
        case 'ann',       s = 'ANN fitnet(5, trainbr)';
        case 'gpr',       s = 'GPR SE 核 + 線性基底';
        case 'svr',       s = 'SVR RBF 核';
        case 'lsboost',   s = 'Boosted trees (LSBoost)';
        case 'rf',        s = '隨機森林 (TreeBagger 200)';
        otherwise,        s = m;
    end
end

function s = usesFeatures(m, featDesc, sensorCol)
    switch lower(m)
        case 'constmean', s = '無（只用訓練摺平均）';
        case {'ratio','deming','powerlaw'}, s = '僅像素長度';
        case 'lme', s = sprintf('%s + 隨機截距(%s)', featDesc, sensorCol);
        otherwise, s = featDesc;
    end
end

function nm = featVarNames(p)
    nm = cell(1,p);
    for i = 1:p, nm{i} = sprintf('x%d', i); end
end


%% ---------------- 指標 ----------------
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
        S.r(r) = NaN; S.r2(r) = NaN;
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

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
