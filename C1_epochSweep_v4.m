function C1b_epochSweep()
%% C1b_epochSweep.m
% ============================================================
% 用途：epoch 對照實驗。同一份資料、同一組摺、同一組初始權重，
%       唯一變因 = epoch 數，一次跑出 EPOCH_LIST 每個 epoch 的
%       out-of-fold MAE(10 次重複的平均 ± SD)。
%
% 作法：每一摺先 configure 決定初始權重 net0，
%       對 EPOCH_LIST 的每個 E 都「從 net0 重新訓練 E 個 epoch」。
%       → 每個 E 的結果 = 把 C1 的 MAX_EPOCHS 設成 E 單獨跑的結果。
%       不採分段續訓：C1 用 trainbr，每次呼叫 train 會重新初始化
%       正則化參數(alpha/beta)，續訓與一次跑完不等價。
%
% 與 C1 的關係：資料讀取、剔除、分摺、標準化、指標算法全部照 C1_v3。
%       本檔不覆寫 C1/C2 的任何檔案，另存帶時間戳的 xlsx + png。
%
% 驗證：EPOCH_LIST 放入 C1 當時的 MAX_EPOCHS(200)，且 NO_EARLY_STOP
%       設成跟 C1 一致時，該列 MAE 應與 C1 輸出的 A 列完全相同。
% ============================================================

    %% ---------------- 輸入設定區(同 C1) ----------------
    METHOD   = '冠寬比例尺';   % 'mask幾何' | '冠寬比例尺' | '牙位基準'
    filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];
    FOLD_COLUMN = '折數fold';

    USE_AGE = false;
    USE_SEX = false;
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    %% ---------------- epoch 掃描設定 ----------------
    EPOCH_LIST    = [50:10:300];  % 任意間隔都可，例：10:10:300
    NO_EARLY_STOP = true;   % goal=0、min_grad=0；mu 達 mu_max 仍可能提早停(會記錄實際 epoch)

    %% ---------------- 評估設定區(同 C1) ----------------
    N_REPEATS   = 10;
    BASE_SEED   = 42;
    HIDDEN_SIZE = 5;
    TRAIN_FCN   = 'trainbr';
    RUN_LINEAR_BASELINE = true;

    OFFSET_MM          = 0.5;
    IDEAL_TOLERANCE_MM = 1.0;

    %% --- 1. 讀取資料(同 C1) ---
    if ~isfile(filename)
        error('找不到 %s，請先跑 B4_build_C1_input.py。', filename);
    end
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    if ismember(SEX_COLUMN, opts.VariableNames)
        opts = setvartype(opts, SEX_COLUMN, 'string');
    end
    data = readtable(filename, opts);

    Y = data.('填充物長度(mm)');
    X = data.('像素長度');
    vn = data.Properties.VariableNames;
    hasDemo = ismember(AGE_COLUMN, vn) && ismember(SEX_COLUMN, vn);
    if (USE_AGE || USE_SEX) && ~hasDemo
        error('輸入檔沒有「%s」「%s」欄，請重跑 B4。', AGE_COLUMN, SEX_COLUMN);
    end
    if hasDemo
        age = toNum(data.(AGE_COLUMN));
        sex = parseSex(data.(SEX_COLUMN));
    else
        age = nan(height(data), 1);
        sex = nan(height(data), 1);
    end

    valid = ~isnan(X) & ~isnan(Y);
    if hasDemo, valid = valid & ~isnan(age) & ~isnan(sex); end
    if any(~valid)
        fprintf('⚠️ 有 %d 筆含缺值，已剔除。\n', sum(~valid));
    end
    X = X(valid); Y = Y(valid); age = age(valid); sex = sex(valid);
    data = data(valid, :);
    n = numel(Y);

    Xmat = X; featNames = {'像素長度'};
    if USE_AGE, Xmat = [Xmat age]; featNames{end+1} = '年齡'; end
    if USE_SEX, Xmat = [Xmat sex]; featNames{end+1} = '性別(M=1)'; end
    featDesc = strjoin(featNames, ' + ');
    FEAT_TAG = featTag(USE_AGE, USE_SEX);

    if ~ismember(FOLD_COLUMN, vn)
        error('找不到「%s」欄。本腳本要求與 C1 共用同一組分摺。', FOLD_COLUMN);
    end
    foldId = double(data.(FOLD_COLUMN));
    K = max(foldId);

    EPOCH_LIST = unique(EPOCH_LIST(:)');   % 排序、去重
    nE = numel(EPOCH_LIST);

    fprintf('=== C1b epoch 掃描：%s | 輸入 %s | n=%d ===\n', METHOD, featDesc, n);
    fprintf('epoch：%s\n', mat2str(EPOCH_LIST));
    fprintf('每摺訓練總 epoch = %d，共 %d 摺 × %d 次重複\n', sum(EPOCH_LIST), K, N_REPEATS);

    %% --- 2. 交叉驗證主迴圈 ---
    annPred = nan(n, N_REPEATS, nE);
    linPred = nan(n, N_REPEATS);
    epUsed  = nan(K, N_REPEATS, nE);     % 實際訓練到的 epoch

    t0 = tic;
    for r = 1:N_REPEATS
        rng(BASE_SEED + r);
        for k = 1:K
            teIdx = (foldId == k);
            trIdx = ~teIdx & (foldId > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end

            Xtr = Xmat(trIdx, :); Ytr = Y(trIdx);
            Xte = Xmat(teIdx, :);
            mu = mean(Xtr, 1);
            sg = std(Xtr, 0, 1); sg(sg == 0) = 1;
            XtrN = (Xtr - mu) ./ sg;
            XteN = (Xte - mu) ./ sg;

            net0 = fitnet(HIDDEN_SIZE, TRAIN_FCN);
            net0.trainParam.showWindow = false;
            net0.divideFcn = 'dividetrain';
            if NO_EARLY_STOP
                net0.trainParam.goal     = 0;
                net0.trainParam.min_grad = 0;
            end
            net0 = configure(net0, XtrN', Ytr');   % 初始權重在此決定(每摺只抽一次亂數)

            for e = 1:nE
                net = net0;                         % 每個 E 都從同一組初始權重開始
                net.trainParam.epochs = EPOCH_LIST(e);
                [net, tr] = train(net, XtrN', Ytr');
                annPred(teIdx, r, e) = net(XteN')';
                epUsed(k, r, e) = tr.num_epochs;
            end

            if RUN_LINEAR_BASELINE
                b = [ones(size(Xtr,1),1) Xtr] \ Ytr;
                linPred(teIdx, r) = [ones(size(Xte,1),1) Xte] * b;
            end
        end
        fprintf('  重複 %d/%d 完成(預估剩餘 %.1f 分)\n', r, N_REPEATS, ...
            toc(t0) / r * (N_REPEATS - r) / 60);
    end

    %% --- 3. 指標：每個 epoch 各自算 10 次重複，再取平均 ± SD ---
    ANN = cell(nE, 1);
    for e = 1:nE
        M = emptyMetrics(N_REPEATS);
        for r = 1:N_REPEATS
            M = accumMetrics(M, r, annPred(:,r,e), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
        end
        ANN{e} = M;
    end
    L = emptyMetrics(N_REPEATS);
    if RUN_LINEAR_BASELINE
        for r = 1:N_REPEATS
            L = accumMetrics(L, r, linPred(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
        end
    end

    f = @(fld, fun) cellfun(@(M) fun(M.(fld)), ANN);
    epMean = squeeze(mean(epUsed, [1 2], 'omitnan'));
    epMin  = squeeze(min(epUsed, [], [1 2]));

    colNames = {'模型', '設定epoch', '實際epoch平均', '實際epoch最小', ...
        'MAE_mm', 'MAE_SD', '校正後MAE_mm', '校正後MAE_SD', ...
        'RMSE_mm', 'R_Square', '理想比率_pct', '過長率_pct'};

    Summary = table(repmat("ANN", nE, 1), EPOCH_LIST(:), round(epMean,1), epMin, ...
        round(f('mae',@mean),3),    round(f('mae',@std),3), ...
        round(f('offMae',@mean),3), round(f('offMae',@std),3), ...
        round(f('rmse',@mean),3),   round(f('r2',@mean),3), ...
        round(f('ideal',@mean),1),  round(f('over',@mean),1), ...
        'VariableNames', colNames);

    if RUN_LINEAR_BASELINE
        LinRow = table("線性迴歸", NaN, NaN, NaN, ...
            round(mean(L.mae),3),    round(std(L.mae),3), ...
            round(mean(L.offMae),3), round(std(L.offMae),3), ...
            round(mean(L.rmse),3),   round(mean(L.r2),3), ...
            round(mean(L.ideal),1),  round(mean(L.over),1), ...
            'VariableNames', colNames);
        Summary = [Summary; LinRow];
    end

    % 逐次重複(長表，之後要做檢定可用)
    [RR, EE] = ndgrid(1:N_REPEATS, EPOCH_LIST);
    maeMat    = cell2mat(cellfun(@(M) M.mae,    ANN', 'UniformOutput', false));
    offMaeMat = cell2mat(cellfun(@(M) M.offMae, ANN', 'UniformOutput', false));
    PerRep = table(EE(:), RR(:), round(maeMat(:),4), round(offMaeMat(:),4), ...
        'VariableNames', {'epoch', '重複', 'MAE_mm', '校正後MAE_mm'});

    runTag = char(datetime('now', 'Format', 'yyyyMMdd_HHmm'));
    SettingsTable = table( ...
        ["量測方法"; "輸入檔"; "輸入特徵"; "n"; "摺數"; "重複次數"; "BASE_SEED"; ...
         "TRAIN_FCN"; "HIDDEN_SIZE"; "EPOCH_LIST"; "NO_EARLY_STOP"; "OFFSET_MM"; "run tag"], ...
        [string(METHOD); string(filename); string(featDesc); string(n); string(K); ...
         string(N_REPEATS); string(BASE_SEED); string(TRAIN_FCN); string(HIDDEN_SIZE); ...
         string(mat2str(EPOCH_LIST)); string(NO_EARLY_STOP); string(OFFSET_MM); string(runTag)], ...
        'VariableNames', {'項目', '值'});

    %% --- 4. 匯出(不覆寫，檔名含時間戳) ---
    outXlsx = sprintf('C1b_epoch掃描_%s%s_%s.xlsx', METHOD, FEAT_TAG, runTag);
    writetable(Summary,       outXlsx, 'Sheet', 'epoch對照');
    writetable(PerRep,        outXlsx, 'Sheet', '逐次重複');
    writetable(SettingsTable, outXlsx, 'Sheet', '設定');

    fig = figure('Color', 'w');
    errorbar(EPOCH_LIST, f('offMae',@mean), f('offMae',@std), '-o', 'LineWidth', 1.2);
    hold on;
    if RUN_LINEAR_BASELINE
        yline(mean(L.offMae), '--', sprintf('線性迴歸 %.3f', mean(L.offMae)));
    end
    xlabel('epoch'); ylabel('校正後 MAE (mm)');
    title(sprintf('%s | %s | %d次重複', METHOD, featDesc, N_REPEATS));
    grid on;
    outPng = strrep(outXlsx, '.xlsx', '.png');
    exportgraphics(fig, outPng, 'Resolution', 200);

    %% --- 5. 輸出摘要 ---
    fprintf('\n✅ 已匯出：%s\n   %s\n\n', outXlsx, outPng);
    disp(Summary);

    nEarly = sum(epUsed < reshape(EPOCH_LIST, 1, 1, []), 'all');
    if nEarly > 0
        fprintf('⚠️ 有 %d 次訓練在設定 epoch 前就停止(多半是 mu 達 mu_max)，看「實際epoch」欄。\n', nEarly);
        fprintf('   停得比設定早的話，較大的 epoch 結果會跟停止點相同，不代表訓練更久。\n');
    end

    [~, iBest] = min(f('offMae', @mean));
    fprintf('\n校正後 MAE 最低：epoch %d(%.3f mm)\n', EPOCH_LIST(iBest), mean(ANN{iBest}.offMae));
    fprintf('   注意：用同一批 OOF 結果挑 epoch 再報告這個數字會偏樂觀，僅供描述趨勢。\n');
end


%% ============================================================
%  子函式(同 C1_v3)
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
