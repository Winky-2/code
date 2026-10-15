function L = C1g_lib()
%% C1g_lib.m
% ============================================================
% C1g_pixelToMmPredictor_GPR.m 與 C1g2_ablation_frontend_backend.m 共用。
%   用法：L = C1g_lib();  D = L.loadC1Input(file, cfg); ...
%   三支檔案放同一個資料夾即可。
%
% 為什麼抽出來：兩支腳本的讀檔、分摺、指標定義必須完全相同，數字才能
% 放進同一張表；各寫一份，遲早會改了一邊忘了另一邊。
%
% 讀檔規則與 ANN 迴圈逐行沿用 C1_pixelToMmPredictor_v6：
%   同一份輸入檔、同一組設定下，cvANN 的結果應與 C1 v6 的 ANN 逐位相同。
% ============================================================
    L.loadC1Input        = @loadC1Input;
    L.alignByName        = @alignByName;
    L.buildFeatures      = @buildFeatures;
    L.featTag            = @featTag;
    L.cvANN              = @cvANN;
    L.cvGPR              = @cvGPR;
    L.cvLinear           = @cvLinear;
    L.cvMean             = @cvMean;
    L.cvArchMean         = @cvArchMean;
    L.pointMetrics       = @pointMetrics;
    L.repMetrics         = @repMetrics;
    L.uncertaintyMetrics = @uncertaintyMetrics;
    L.riskCoverage       = @riskCoverage;
    L.minReferForTarget  = @minReferForTarget;
    L.runTag             = @runTag;
end


%% ============================================================
%  讀檔
%  ============================================================

function D = loadC1Input(filename, cfg)
% cfg 欄位：AGE_COLUMN, SEX_COLUMN, FOLD_COLUMN, POS_COLUMN, USE_AGE, USE_SEX,
%           SCORE_COLUMN(可留空)
    if ~isfile(filename)
        error(['找不到 %s\n' ...
               '   這份檔案要跟 C1 輸入同格式：圖片檔名 / 填充物長度(mm) / 像素長度 / 折數fold\n' ...
               '   (+ 年齡 / 性別 / 牙位)。seg 版由 B4 產生；B4 目前是 seg-only，pose 版要另外產生。'], ...
               filename);
    end
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    if ismember(cfg.SEX_COLUMN, opts.VariableNames)
        opts = setvartype(opts, cfg.SEX_COLUMN, 'string');
    end
    data = readtable(filename, opts);
    vn = data.Properties.VariableNames;

    need = {'圖片檔名', '填充物長度(mm)', '像素長度', cfg.FOLD_COLUMN};
    miss = need(~ismember(need, vn));
    if ~isempty(miss)
        error('%s 缺少欄位：%s', filename, strjoin(miss, '、'));
    end

    ids  = string(data.('圖片檔名'));
    Y    = data.('填充物長度(mm)');
    X    = data.('像素長度');
    if ~isnumeric(Y) || ~isnumeric(X)
        error('%s 的實際長度或像素長度欄含非數值資料，請檢查 Excel。', filename);
    end
    fold = double(data.(cfg.FOLD_COLUMN));

    hasDemo = ismember(cfg.AGE_COLUMN, vn) && ismember(cfg.SEX_COLUMN, vn);
    if (cfg.USE_AGE || cfg.USE_SEX) && ~hasDemo
        error('%s 沒有「%s」「%s」欄(B4 的 MM_EXTRA_COLS 要帶這兩欄)。', ...
              filename, cfg.AGE_COLUMN, cfg.SEX_COLUMN);
    end
    if hasDemo
        age = toNum(data.(cfg.AGE_COLUMN));
        sex = parseSex(data.(cfg.SEX_COLUMN));           % M=1, F=0, 其他 NaN
        if all(isnan(sex))
            error('%s 的「%s」欄全部無法解析(只認 M/F/男/女)。', filename, cfg.SEX_COLUMN);
        end
    else
        age = nan(height(data), 1);
        sex = nan(height(data), 1);
    end

    if ismember(cfg.POS_COLUMN, vn)
        pos = toNum(data.(cfg.POS_COLUMN));
    else
        pos = nan(height(data), 1);
    end
    q = floor(pos / 10);
    arch = nan(size(pos));
    arch(ismember(q, [1 2])) = 1;      % 上顎
    arch(ismember(q, [3 4])) = 0;      % 下顎

    score = nan(height(data), 1);
    if isfield(cfg, 'SCORE_COLUMN') && ~isempty(cfg.SCORE_COLUMN)
        if ~ismember(cfg.SCORE_COLUMN, vn)
            error('%s 沒有品質分數欄「%s」。', filename, cfg.SCORE_COLUMN);
        end
        score = toNum(data.(cfg.SCORE_COLUMN));
    end

    % 剔除規則與 C1 v6 相同：有人口學欄時，年齡/性別缺值也剔除(即使沒用到)
    valid = ~isnan(X) & ~isnan(Y);
    if hasDemo
        valid = valid & ~isnan(age) & ~isnan(sex);
    end

    D = struct('file', string(filename), 'ids', ids, 'Y', Y, 'X', X, ...
               'age', age, 'sex', sex, 'pos', pos, 'arch', arch, ...
               'fold', fold, 'score', score);
    D = subsetD(D, valid);
    D.hasDemo  = hasDemo;
    D.nDropped = sum(~valid);
end


function D = subsetD(D, idx)
    f = {'ids', 'Y', 'X', 'age', 'sex', 'pos', 'arch', 'fold', 'score'};
    for i = 1:numel(f)
        D.(f{i}) = D.(f{i})(idx);
    end
    D.n = numel(D.Y);
end


function [A, B, info] = alignByName(A, B)
% 依檔名開頭編號取交集。保留 A 的列順序，B 重排成 A 的順序，摺一律用 A 的。
    kA = nameKey(A.ids);
    kB = nameKey(B.ids);
    if numel(unique(kA)) < numel(kA) || numel(unique(kB)) < numel(kB)
        error('檔名編號有重複，無法一對一對齊(%s / %s)。', A.file, B.file);
    end
    [~, iA, iB] = intersect(kA, kB, 'stable');

    info.nA      = A.n;
    info.nB      = B.n;
    info.nCommon = numel(iA);
    info.onlyA   = kA(setdiff((1:A.n)', iA));
    info.onlyB   = kB(setdiff((1:B.n)', iB));

    A = subsetD(A, iA);
    B = subsetD(B, iB);
    info.foldAgree = isequal(A.fold, B.fold);

    dY = max(abs(A.Y - B.Y));
    if dY > 1e-6
        error('對齊後兩份檔的實際長度不一致(最大差 %.3f mm)，檔名對應可能錯了。', dY);
    end
    B.fold = A.fold;
end


function k = nameKey(ids)
% 與 B4 normalize_name 同規則：取檔名開頭的編號(060_xxx.jpg → "60")；
% 開頭不是數字才用小寫全名。
    ids = string(ids);
    k = lower(strtrim(ids));
    for i = 1:numel(ids)
        t = regexp(char(ids(i)), '^\s*0*(\d+)', 'tokens', 'once');
        if ~isempty(t)
            k(i) = string(str2double(t{1}));
        end
    end
end


function [Xmat, names] = buildFeatures(D, useAge, useSex)
    Xmat = D.X;
    names = {'像素長度'};
    if useAge, Xmat = [Xmat D.age]; names{end+1} = '年齡'; end
    if useSex, Xmat = [Xmat D.sex]; names{end+1} = '性別(M=1)'; end
end


function tag = featTag(useAge, useSex)
% 與 C1 v6 相同：都關 = ''
    parts = {};
    if useAge, parts{end+1} = '年齡'; end
    if useSex, parts{end+1} = '性別'; end
    if isempty(parts)
        tag = '';
    else
        tag = ['_' strjoin(parts, '')];
    end
end


function t = runTag()
    t = char(datetime('now', 'Format', 'yyyyMMdd_HHmm'));
end


%% ============================================================
%  模型(全部 out-of-fold)
%  ============================================================

function [XtrN, XteN] = zscoreTrain(Xtr, Xte)
% 只用訓練摺的平均/標準差(與 C1 v6 相同)
    mu = mean(Xtr, 1);
    sg = std(Xtr, 0, 1);
    sg(sg == 0) = 1;
    XtrN = (Xtr - mu) ./ sg;
    XteN = (Xte - mu) ./ sg;
end


function pred = cvANN(Xmat, Y, fold, P)
% 與 C1 v6 第 3 節逐行相同，亂數呼叫順序也相同(重複在外、摺在內)。
% P：N_REPEATS, BASE_SEED, HIDDEN_SIZE, TRAIN_FCN, MAX_EPOCHS
    n = numel(Y);
    K = max(fold);
    pred = nan(n, P.N_REPEATS);
    for r = 1:P.N_REPEATS
        rng(P.BASE_SEED + r);
        for k = 1:K
            teIdx = (fold == k);
            trIdx = ~teIdx & (fold > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end

            [XtrN, XteN] = zscoreTrain(Xmat(trIdx, :), Xmat(teIdx, :));
            Ytr = Y(trIdx);

            net = fitnet(P.HIDDEN_SIZE, P.TRAIN_FCN);
            net.trainParam.showWindow = false;
            net.trainParam.epochs     = P.MAX_EPOCHS;
            net.divideFcn = 'dividetrain';

            net = train(net, XtrN', Ytr');
            pred(teIdx, r) = net(XteN')';
        end
        fprintf('    ANN 重複 %d/%d 完成\n', r, P.N_REPEATS);
    end
end


function [mu, sd, fitInfo] = cvGPR(Xmat, Y, fold, P)
% P：GPR_KERNEL, GPR_BASIS
% GPR 沒有隨機初始化(超參數從資料推出的初值做梯度最佳化)，
% 摺固定時重跑結果相同 → 不需要重複，也沒有「重複間標準差」。
% sd = 新觀測值的預測標準差(含雜訊項)，用來算預測區間。
    n = numel(Y);
    K = max(fold);
    mu = nan(n, 1);
    sd = nan(n, 1);
    fk = []; ntr = []; sig = []; kps = strings(0, 1); ll = [];
    for k = 1:K
        teIdx = (fold == k);
        trIdx = ~teIdx & (fold > 0);
        if ~any(teIdx) || ~any(trIdx), continue; end

        [XtrN, XteN] = zscoreTrain(Xmat(trIdx, :), Xmat(teIdx, :));
        g = fitrgp(XtrN, Y(trIdx), ...
                   'KernelFunction', P.GPR_KERNEL, ...
                   'BasisFunction',  P.GPR_BASIS, ...
                   'Standardize',    false);
        [m, s] = predict(g, XteN);
        mu(teIdx) = m;
        sd(teIdx) = s;

        kp = g.KernelInformation.KernelParameters;
        fk(end+1, 1)   = k;                                     %#ok<AGROW>
        ntr(end+1, 1)  = sum(trIdx);                            %#ok<AGROW>
        sig(end+1, 1)  = round(g.Sigma, 4);                     %#ok<AGROW>
        kps(end+1, 1)  = strjoin(string(round(kp(:)', 4)), ', '); %#ok<AGROW>
        ll(end+1, 1)   = round(g.LogLikelihood, 3);             %#ok<AGROW>
    end
    fitInfo = table(fk, ntr, sig, kps, ll, 'VariableNames', ...
        {'摺', '訓練筆數', '雜訊Sigma_mm', '核參數', '對數概似'});
end


function pred = cvLinear(Xmat, Y, fold)
% 與 C1 v6 的線性對照組相同(未標準化，最小平方)
    n = numel(Y);
    K = max(fold);
    pred = nan(n, 1);
    for k = 1:K
        te = (fold == k);
        tr = ~te & (fold > 0);
        if ~any(te) || ~any(tr), continue; end
        b = [ones(sum(tr), 1) Xmat(tr, :)] \ Y(tr);
        pred(te) = [ones(sum(te), 1) Xmat(te, :)] * b;
    end
end


function pred = cvMean(Y, fold)
% 基準：不看影像，猜訓練摺的平均長度
    n = numel(Y);
    K = max(fold);
    pred = nan(n, 1);
    for k = 1:K
        te = (fold == k);
        tr = ~te & (fold > 0);
        if ~any(te) || ~any(tr), continue; end
        pred(te) = mean(Y(tr));
    end
end


function pred = cvArchMean(Y, fold, arch)
% 基準：不看影像，只知道上/下顎，猜訓練摺該顎的平均。
% 牙位缺漏或訓練摺沒有同顎的牙 → 退回整體平均。全部沒有牙位 → 全 NaN。
    n = numel(Y);
    pred = nan(n, 1);
    if all(isnan(arch)), return; end
    K = max(fold);
    for k = 1:K
        te = (fold == k);
        tr = ~te & (fold > 0);
        if ~any(te) || ~any(tr), continue; end
        mAll = mean(Y(tr));
        for i = find(te)'
            same = tr & (arch == arch(i));
            if ~isnan(arch(i)) && any(same)
                pred(i) = mean(Y(same));
            else
                pred(i) = mAll;
            end
        end
    end
end


%% ============================================================
%  指標
%  ============================================================

function S = pointMetrics(Yp, Y, offset, tol)
% 與 C1 v6 accumMetrics 同定義(單次)
    ok = ~isnan(Yp) & ~isnan(Y);
    Yp = Yp(ok); Y = Y(ok);

    res = Yp - Y;
    S.mae  = mean(abs(res));
    S.rmse = sqrt(mean(res.^2));
    S.bias = mean(res);
    S.r  = NaN;
    S.r2 = NaN;
    if numel(Y) >= 2 && std(Yp) > 0
        S.r  = corr(Y, Yp);
        S.r2 = S.r^2;
    end

    offRes = (Yp - offset) - Y;
    S.offMae  = mean(abs(offRes));
    S.offRmse = sqrt(mean(offRes.^2));
    S.offBias = mean(offRes);
    S.ideal   = mean(abs(offRes) <= tol) * 100;
end


function M = repMetrics(predMat, Y, offset, tol)
% 每一欄(一次重複)算一次 pointMetrics → 每個欄位是 R×1 向量。
% 決定性方法只有一欄 → std = 0。
    f = {'mae', 'rmse', 'bias', 'r', 'r2', 'offMae', 'offRmse', 'offBias', 'ideal'};
    R = size(predMat, 2);
    for i = 1:numel(f)
        M.(f{i}) = nan(R, 1);
    end
    for r = 1:R
        S = pointMetrics(predMat(:, r), Y, offset, tol);
        for i = 1:numel(f)
            M.(f{i})(r) = S.(f{i});
        end
    end
end


function [T, U] = uncertaintyMetrics(mu, sd, Y, levels)
% 用「未校正」的 GPR 輸出算：offset 是事後的臨床平移，不屬於模型的機率分布。
    ok = ~isnan(mu) & ~isnan(sd) & ~isnan(Y);
    mu = mu(ok); sd = sd(ok); Y = Y(ok);
    n = numel(Y);
    absErr = abs(Y - mu);

    item = strings(0, 1); val = zeros(0, 1); note = strings(0, 1);
    U.coverage = nan(numel(levels), 1);
    for j = 1:numel(levels)
        lv = levels(j);
        z = norminv(0.5 + lv / 2);
        c = mean(absErr <= z * sd) * 100;
        U.coverage(j) = c;
        se2 = 2 * sqrt(lv * (1 - lv) / n) * 100;     % 二項抽樣誤差的 2 倍
        if c < lv * 100 - se2
            nt = sprintf('偏低(理想 %g±%.0f)：區間太窄，過度自信', lv * 100, se2);
        elseif c > lv * 100 + se2
            nt = sprintf('偏高(理想 %g±%.0f)：區間太寬，偏保守', lv * 100, se2);
        else
            nt = sprintf('✅ 在理想 %g±%.0f 內', lv * 100, se2);
        end
        item(end+1, 1) = sprintf('%g%% 預測區間實際涵蓋率(%%)', lv * 100); %#ok<AGROW>
        val(end+1, 1)  = round(c, 1);                                     %#ok<AGROW>
        note(end+1, 1) = nt;                                              %#ok<AGROW>
    end

    z95 = norminv(0.975);
    item(end+1, 1) = "95% 預測區間平均寬度(mm)";
    val(end+1, 1)  = round(mean(2 * z95 * sd), 3);
    note(end+1, 1) = "涵蓋率達標的前提下越窄越有用";
    item(end+1, 1) = "預測標準差 σ：平均(mm)";
    val(end+1, 1)  = round(mean(sd), 3);
    note(end+1, 1) = "";
    item(end+1, 1) = "預測標準差 σ：最小(mm)";
    val(end+1, 1)  = round(min(sd), 3);
    note(end+1, 1) = "";
    item(end+1, 1) = "預測標準差 σ：最大(mm)";
    val(end+1, 1)  = round(max(sd), 3);
    note(end+1, 1) = "最大/最小差很少 → σ 幾乎是常數，排轉診順序沒有意義";

    [rho, p] = corr(absErr, sd, 'Type', 'Spearman');
    if rho > 0 && p < 0.05
        nt = "✅ σ 大的牙誤差確實較大 → σ 可用來排轉診";
    elseif rho > 0
        nt = "⚠️ 正相關但不顯著(n 小)，轉診故事證據弱";
    else
        nt = "❌ σ 跟誤差無正相關 → 用 σ 轉診不成立";
    end
    item(end+1, 1) = "|誤差| 與 σ 的 Spearman ρ";
    val(end+1, 1)  = round(rho, 3);
    note(end+1, 1) = nt;
    item(end+1, 1) = "  ρ 的 p 值(雙尾)";
    val(end+1, 1)  = round(p, 4);
    note(end+1, 1) = sprintf("n = %d", n);

    T = table(item, val, note, 'VariableNames', {'項目', '值', '判讀'});
    U.rho = rho;
    U.p   = p;
    U.levels = levels(:);
end


function ord = riskOrder(score, higherIsRiskier)
% 由「最該轉診」排到「最不該」。score 為 NaN 視為最該轉診。
    s = score(:);
    if ~higherIsRiskier, s = -s; end
    s(isnan(s)) = Inf;
    [~, ord] = sort(s, 'descend');
end


function [retMAE, nRef] = riskCoverage(absErr, score, grid, higherIsRiskier)
% 依 score 轉診 grid 比例的案例，回傳剩下案例的 MAE。
    n = numel(absErr);
    ord = riskOrder(score, higherIsRiskier);
    retMAE = nan(numel(grid), 1);
    nRef = zeros(numel(grid), 1);
    for g = 1:numel(grid)
        k = min(round(grid(g) * n), n - 1);
        nRef(g) = k;
        retMAE(g) = mean(absErr(ord(k+1:end)));
    end
end


function [frac, k] = minReferForTarget(absErr, score, target, higherIsRiskier)
% 讓剩下案例 MAE < target 所需的最少轉診比例；轉到只剩 1 筆都做不到 → NaN
    n = numel(absErr);
    ord = riskOrder(score, higherIsRiskier);
    frac = NaN; k = NaN;
    for kk = 0:n-1
        if mean(absErr(ord(kk+1:end))) < target
            frac = kk / n;
            k = kk;
            return;
        end
    end
end


%% ============================================================
%  小工具(與 C1 v6 相同)
%  ============================================================

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
