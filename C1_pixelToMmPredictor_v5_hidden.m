function C1_pixelToMmPredictor_v5()
%% C1_pixelToMmPredictor_v5.m
% ============================================================
% v5 相對 v4 的兩個改動(模型、分摺、標準化、重複次數、代表性重複
% 的挑法一律沒動)：
%
%   (1) 報告主軸改為「校正後(offset)」：C 列現在把 RMSE / R / r^2 /
%       R2_OOF / Bias 全部補滿，終端機的報告區塊也直接印這一組。
%       ⚠️ 數學事實：常數 offset 只平移預測值，所以
%         R(Pearson)、r^2、Spearman、SD_res、LoA 寬度 → 校正前後完全相同
%         MAE、RMSE、Bias、R2_OOF(=1-SSE/SST)   → 會改變
%       因此「校正後 R」= 「未校正 R」，是同一個數字，不是重算出來的，
%       報告裡寫成兩個不同的值會是錯的。
%
%   (2) SHOW_OVERLENGTH 開關(預設 false)：關掉後「過長率」與「嚴重
%       過長率」不會出現在任何輸出(欄位直接不產生)。
%       搭配 C2_plotMyResults_v4.m 使用(v3 會因為找不到欄位而報錯)。
%
% v4 已有的指標(都算在同一組 out-of-fold 預測上)：
%
% 【相關 / 決定係數】
%   R          Pearson r(含方向，附全資料 p 值)
%   R_Square   = r^2（沿用 v3 欄位，定義不變，向下相容）
%   R2_OOF     = 1 - SSE/SST ← 真正的「解釋變異比例」，會被 bias/
%              斜率偏誤懲罰，可能是負值。r^2 高但 R2_OOF 低 = 相關好
%              但校準差，這兩個一起報才誠實。
%   Spearman   單調相關(對離群值穩健)
%
% 【一致性(method comparison 的標準做法)】
%   CCC        Lin's concordance correlation coefficient
%   ICC(2,1)   two-way random effects, absolute agreement, single measure
%   SD_diff / LoA_low / LoA_high   Bland-Altman 95% 一致性界線
%   比例偏誤   差值對均值回歸的斜率 + p(看誤差是否隨長度放大)
%
% 【誤差分布(n 小、怕離群值主導 MAE 時很重要)】
%   MedAE / P90 / P95 / MaxAE / SD_res / SEE
%   MAPE / MPE(相對誤差，%)
%
% 【校準】
%   校準斜率 / 截距：以 Y ~ a + b*Yp 迴歸，理想 b=1、a=0
%
% 【臨床門檻(除了原本 ±1.0mm)】
%   ±0.5mm 命中率、±1.5mm 命中率、過短率、嚴重過長率(>1mm)
%
% 【不確定度與對照檢定】(新 sheet「指標_CI與檢定」)
%   bootstrap 95% CI(重抽樣本)：v3 的「±」只是重跑 10 次的訓練波動，
%   不是樣本抽樣誤差，兩者意義不同，論文要寫的是後者。
%   ANN vs 線性：逐筆絕對誤差配對 Wilcoxon signed-rank + ΔMAE 的
%   bootstrap CI(比只看兩個平均值大小有力得多)。
%
% ⚠️ 誠實標註
%   1. bootstrap CI 建在 OOF 預測上，摺內樣本被重複使用，CI 會略窄。
%   2. 指標一次看十幾個、未校正多重比較，p 值只當描述。
%   3. ICC/CCC 把「預測」當成另一個量測者，前提是兩者可互換；
%      offset 校正後的版本才接近臨床情境。
% ============================================================

    %% ---------------- 輸入設定區 ----------------
    METHOD = 'mask幾何_長邊正規化';   % 'mask幾何' | '冠寬比例尺' | 'mask幾何_長邊正規化'
    filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];

    FOLD_COLUMN  = '折數fold';
    SPLIT_COLUMN = '資料夾來源';

    %% ---------------- 特徵設定區 ----------------
    USE_AGE = false;
    USE_SEX = false;
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    %% ---------------- 評估設定區 ----------------
    USE_KFOLD  = true;
    N_REPEATS  = 10;
    BASE_SEED  = 42;

    HIDDEN_SIZE = 3;
    TRAIN_FCN   = 'trainbr';
    MAX_EPOCHS  = 200;

    RUN_LINEAR_BASELINE = true;

    %% ---------------- 新增：CI / 檢定設定 ----------------
    DO_BOOTSTRAP = true;
    N_BOOT       = 2000;
    BOOT_SEED    = 20261008;
    ALPHA        = 0.05;

    %% ---------------- 報告設定區(新增) ----------------
    SHOW_OVERLENGTH = false;   % false = 任何輸出都不出現「過長率」
                               % ⚠️ C2 的同名開關要設成一樣

    %% ---------------- 臨床設定區 ----------------
    OFFSET_MM          = 0.5;
    IDEAL_TOLERANCE_MM = 1.0;
    TIGHT_TOLERANCE_MM = 0.5;    % 新增：更嚴格門檻
    LOOSE_TOLERANCE_MM = 1.5;    % 新增：較寬門檻
    SEVERE_OVER_MM     = 1.0;    % 新增：過長超過這個量算「嚴重」

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
               '   請在 B4 的 MM_EXTRA_COLS 加上這兩欄並重跑 B4。'], ...
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

    Xmat = X;
    featNames = {'像素長度'};
    if USE_AGE, Xmat = [Xmat age]; featNames{end+1} = '年齡'; end
    if USE_SEX, Xmat = [Xmat sex]; featNames{end+1} = '性別(M=1)'; end
    p = size(Xmat, 2);
    featDesc = strjoin(featNames, ' + ');
    FEAT_TAG = featTag(USE_AGE, USE_SEX);
    nParam = HIDDEN_SIZE*p + HIDDEN_SIZE + HIDDEN_SIZE + 1;

    CFG = struct('offset', OFFSET_MM, 'tol', IDEAL_TOLERANCE_MM, ...
                 'tight', TIGHT_TOLERANCE_MM, 'loose', LOOSE_TOLERANCE_MM, ...
                 'severe', SEVERE_OVER_MM);

    fprintf('=== C1 v4：%s ===\n', METHOD);
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
            fprintf('沿用 Excel 的「%s」欄分摺。\n', FOLD_COLUMN);
        else
            K = 5;
            [~, ord] = sort(Y);
            foldId = zeros(n, 1);
            foldId(ord) = mod(0:n-1, K) + 1;
            fprintf('⚠️ Excel 沒有「%s」欄，改用依實際長度分層的 %d 摺。\n', FOLD_COLUMN, K);
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

    %% --- 3. 交叉驗證主迴圈(與 v3 完全相同) ---
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
        M = accumMetrics(M, r, annPred(:,r), Y, CFG);
    end
    L = emptyMetrics(N_REPEATS);
    if RUN_LINEAR_BASELINE
        for r = 1:N_REPEATS
            L = accumMetrics(L, r, linPred(:,r), Y, CFG);
        end
    end
    foldSpread = mean(std(foldMAE, 0, 1, 'omitnan'), 'omitnan');

    %% --- 5. 代表性重複(挑法同 v3) ---
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
    % 新增：10 次重複的平均預測，之後做 CI / 檢定都用這一欄
    ResultsTable.('重複平均預測_mm') = mean(annPred, 2, 'omitnan');
    if RUN_LINEAR_BASELINE
        ResultsTable.('線性重複平均預測_mm') = mean(linPred, 2, 'omitnan');
    end

    % ---- 6a. 原本的「模型評估指標」sheet：欄位與列關鍵字一律不動 ----
    MetricNames = {
        sprintf('A_Test set表現 (out-of-fold, n=%d, ANN, %d次重複平均, 輸入=%s)', n, N_REPEATS, featDesc);
        sprintf('B_重複間標準差 (同一份資料重跑%d次的波動)', N_REPEATS);
        '--- 分界線 ---';
        sprintf('C_臨床安全性評估_Test set校正後 (offset=-%.2fmm)', OFFSET_MM);
        sprintf('D_線性迴歸對照組_未校正 (同一組摺, n=%d)', n);
        sprintf('E_線性迴歸對照組_校正後 (offset=-%.2fmm)', OFFSET_MM)
    };

    nanpad = NaN;
    hasLin = RUN_LINEAR_BASELINE;

    % 列序：A 未校正ANN / B 重複SD / 分界 / C 校正後ANN / D 線性未校正 / E 線性校正後
    MAE_val = round([mean(M.mae); std(M.mae); nanpad; mean(M.offMae); ...
                     meanOr(L.mae, hasLin); meanOr(L.offMae, hasLin)], 3);
    RMSE_val = round([mean(M.rmse); std(M.rmse); nanpad; mean(M.offRmse); ...
                      meanOr(L.rmse, hasLin); meanOr(L.offRmse, hasLin)], 3);
    % r^2 / R(Pearson)：offset 只是平移，校正前後同值 → C、E 列直接沿用
    R2_val = round([mean(M.r2); std(M.r2); nanpad; mean(M.r2); ...
                    meanOr(L.r2, hasLin); meanOr(L.r2, hasLin)], 3);
    R_val  = round([mean(M.r); std(M.r); nanpad; mean(M.r); ...
                    meanOr(L.r, hasLin); meanOr(L.r, hasLin)], 3);
    % R2_OOF = 1-SSE/SST：offset 會改變，C、E 列是重算的
    R2oof_val = round([mean(M.r2oof); std(M.r2oof); nanpad; mean(M.offR2oof); ...
                       meanOr(L.r2oof, hasLin); meanOr(L.offR2oof, hasLin)], 3);
    Bias_val = round([mean(M.bias); std(M.bias); nanpad; mean(M.offBias); ...
                      meanOr(L.bias, hasLin); meanOr(L.offBias, hasLin)], 3);
    SDres_val = round([mean(M.sdres); std(M.sdres); nanpad; mean(M.offSdres); ...
                       meanOr(L.sdres, hasLin); meanOr(L.offSdres, hasLin)], 3);
    IdealRate_pct = round([nanpad; std(M.ideal); nanpad; mean(M.ideal); ...
                           nanpad; meanOr(L.ideal, hasLin)], 1);

    MetricsTable = table(MetricNames, MAE_val, RMSE_val, R2_val, Bias_val, ...
        IdealRate_pct, R_val, R2oof_val, SDres_val, ...
        'VariableNames', {'評估範圍_與_嚴格程度', 'MAE_mm', 'RMSE_mm', 'R_Square', ...
        'Mean_Bias_mm', '理想比率_pct', 'R_Pearson', 'R2_OOF', 'SD_res_mm'});

    if SHOW_OVERLENGTH
        MetricsTable.('過長率_pct') = round([nanpad; std(M.over); nanpad; ...
            mean(M.over); nanpad; meanOr(L.over, hasLin)], 1);
    end

    % ---- 6b. 新 sheet：指標_延伸(長表，ANN / 線性並排) ----
    ExtTable = buildExtTable(M, L, RUN_LINEAR_BASELINE, CFG, foldSpread, SHOW_OVERLENGTH);

    SettingsTable = table( ...
        ["量測方法"; "輸入特徵"; "特徵標籤"; "n"; "ANN參數數"; "樣本每參數"; ...
         "重複次數"; "BASE_SEED"; "bootstrap次數"; "理想容忍_mm"; "嚴格容忍_mm"; ...
         "寬容忍_mm"; "offset_mm"; "報告過長率"; "腳本版本"], ...
        [string(METHOD); string(featDesc); string(FEAT_TAG); string(n); ...
         string(nParam); string(sprintf('%.2f', n/nParam)); string(N_REPEATS); ...
         string(BASE_SEED); string(DO_BOOTSTRAP*N_BOOT); string(IDEAL_TOLERANCE_MM); ...
         string(TIGHT_TOLERANCE_MM); string(LOOSE_TOLERANCE_MM); ...
         string(OFFSET_MM); string(SHOW_OVERLENGTH); "C1_v5"], ...
        'VariableNames', {'項目', '值'});

    % ---- 6c. 全資料線性係數(同 v3) ----
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

    % ---- 6d. 新 sheet：bootstrap CI + ANN vs 線性配對檢定 ----
    CITable = [];
    if DO_BOOTSTRAP
        YpBar = mean(annPred, 2, 'omitnan');
        LpBar = mean(linPred, 2, 'omitnan');
        CITable = buildCITable(Y, YpBar, LpBar, RUN_LINEAR_BASELINE, ...
                               CFG, N_BOOT, BOOT_SEED, ALPHA, SHOW_OVERLENGTH);
    end

    %% --- 7. 匯出 ---
    output_filename = sprintf('預測結果與評估指標_kfold_%s%s.xlsx', METHOD, FEAT_TAG);
    if isfile(output_filename), delete(output_filename); end
    writetable(ResultsTable,  output_filename, 'Sheet', '所有牙齒預測結果');
    writetable(MetricsTable,  output_filename, 'Sheet', '模型評估指標');
    writetable(ExtTable,      output_filename, 'Sheet', '指標_延伸');
    writetable(SettingsTable, output_filename, 'Sheet', '模型設定');
    if ~isempty(CITable)
        writetable(CITable,   output_filename, 'Sheet', '指標_CI與檢定');
    end
    if ~isempty(CoefTable)
        writetable(CoefTable, output_filename, 'Sheet', '線性係數_全資料');
    end

    fprintf('\n✅ 已匯出：%s\n\n', output_filename);
    disp('=== 評估指標(C2 讀這張) ===');
    disp(MetricsTable);
    fprintf('\n=== 指標_延伸 ===\n');
    disp(ExtTable);
    if ~isempty(CITable)
        fprintf('\n=== bootstrap 95%% CI 與配對檢定 ===\n');
        disp(CITable);
    end

    %% --- 8. 報告用區塊：一律以「校正後」為準 ---
    fprintf('\n=== 臨床評估(校正後，報告用) ===\n');
    fprintf('量測方法：%s | 輸入：%s | offset = -%.2f mm | 理想容忍 = ±%.2f mm\n', ...
        METHOD, featDesc, OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('[Out-of-fold, n=%d, %d次重複]\n', n, N_REPEATS);
    fprintf('  校正後 MAE   : %.3f ± %.3f mm\n', mean(M.offMae),  std(M.offMae));
    fprintf('  校正後 RMSE  : %.3f ± %.3f mm\n', mean(M.offRmse), std(M.offRmse));
    fprintf('  校正後 Bias  : %.3f ± %.3f mm   (正=高估)\n', mean(M.offBias), std(M.offBias));
    fprintf('  R (Pearson)  : %.3f ± %.3f      ← offset 不改變，與未校正同值\n', ...
        mean(M.r), std(M.r));
    fprintf('  r^2          : %.3f             ← 同上，offset 不改變\n', mean(M.r2));
    fprintf('  R2_OOF       : %.3f ± %.3f      ← 1-SSE/SST，這個才會被 offset/bias 影響\n', ...
        mean(M.offR2oof), std(M.offR2oof));
    fprintf('  SD_res       : %.3f mm，95%% LoA [%.3f, %.3f] mm\n', ...
        mean(M.offSdres), mean(M.offLoaLo), mean(M.offLoaHi));
    fprintf('  理想比率     : %.1f ± %.1f %%\n', mean(M.ideal), std(M.ideal));
    if SHOW_OVERLENGTH
        fprintf('  過長率       : %.1f ± %.1f %%\n', mean(M.over), std(M.over));
    end
    fprintf('  摺間 MAE 標準差 : %.3f mm\n', foldSpread);

    if RUN_LINEAR_BASELINE
        fprintf('\n[線性迴歸對照組，校正後] MAE %.3f mm vs ANN %.3f mm  →  ', ...
            mean(L.offMae), mean(M.offMae));
        if mean(L.offMae) <= mean(M.offMae)
            fprintf('線性不輸 ANN\n');
        else
            fprintf('ANN 勝出 %.3f mm\n', mean(L.offMae) - mean(M.offMae));
        end
        fprintf('   差距是否顯著看「指標_CI與檢定」的 ΔMAE CI 與 Wilcoxon p，不要只比平均值。\n');
    end

    fprintf('\n⚠️ 「± SD」= 重跑 %d 次的訓練波動；樣本抽樣誤差請看 bootstrap CI。\n', N_REPEATS);
    fprintf('⚠️ r^2(R_Square) 與 R2_OOF 定義不同，報告時要寫清楚用的是哪一個。\n');
    if ~SHOW_OVERLENGTH
        fprintf('ℹ️ SHOW_OVERLENGTH=false：所有輸出都沒有「過長率」欄，C2 請用 v4。\n');
    end

    if ~isempty(CoefTable)
        fprintf('\n=== 全資料線性係數(解讀用，非效能) ===\n');
        disp(CoefTable);
    end
end


%% ============================================================
%  指標計算
%  ============================================================

function names = metricFields()
    names = {'mae','rmse','bias','sdres','see', ...
             'r','r2','r2oof','rho','ccc','icc', ...
             'medae','p90','p95','maxae','mape','mpe', ...
             'loaLo','loaHi','calSlope','calInt','propSlope', ...
             'offMae','offRmse','offBias','offSdres','offLoaLo','offLoaHi', ...
             'offCcc','offIcc','offR2oof', ...
             'ideal','idealTight','idealLoose','over','overSevere','under'};
end


function S = emptyMetrics(N)
    f = metricFields();
    S = struct();
    for i = 1:numel(f)
        S.(f{i}) = nan(N, 1);
    end
end


function S = accumMetrics(S, r, Yp, Y, CFG)
% 一次算完所有指標；未校正版與 offset 校正版都算
    ok = ~isnan(Yp) & ~isnan(Y);
    Yp = Yp(ok); Y = Y(ok);
    m  = coreMetrics(Y, Yp);                       % 未校正
    mo = coreMetrics(Y, Yp - CFG.offset);          % 校正後

    S.mae(r)   = m.mae;    S.rmse(r)  = m.rmse;   S.bias(r)  = m.bias;
    S.sdres(r) = m.sdres;  S.see(r)   = m.see;
    S.r(r)     = m.r;      S.r2(r)    = m.r2;     S.r2oof(r) = m.r2oof;
    S.rho(r)   = m.rho;    S.ccc(r)   = m.ccc;    S.icc(r)   = m.icc;
    S.medae(r) = m.medae;  S.p90(r)   = m.p90;    S.p95(r)   = m.p95;
    S.maxae(r) = m.maxae;  S.mape(r)  = m.mape;   S.mpe(r)   = m.mpe;
    S.loaLo(r) = m.loaLo;  S.loaHi(r) = m.loaHi;
    S.calSlope(r) = m.calSlope;  S.calInt(r) = m.calInt;
    S.propSlope(r) = m.propSlope;

    S.offMae(r)   = mo.mae;    S.offRmse(r)  = mo.rmse;
    S.offBias(r)  = mo.bias;   S.offSdres(r) = mo.sdres;
    S.offLoaLo(r) = mo.loaLo;  S.offLoaHi(r) = mo.loaHi;
    S.offCcc(r)   = mo.ccc;    S.offIcc(r)   = mo.icc;
    S.offR2oof(r) = mo.r2oof;

    offRes = (Yp - CFG.offset) - Y;
    S.ideal(r)      = mean(abs(offRes) <= CFG.tol)   * 100;
    S.idealTight(r) = mean(abs(offRes) <= CFG.tight) * 100;
    S.idealLoose(r) = mean(abs(offRes) <= CFG.loose) * 100;
    S.over(r)       = mean(offRes > 0) * 100;
    S.overSevere(r) = mean(offRes > CFG.severe) * 100;
    S.under(r)      = mean(offRes < -CFG.tol) * 100;
end


function m = coreMetrics(Y, Yp)
% 所有「不依賴臨床門檻」的指標。Y=實際、Yp=預測(可為已校正)
    res = Yp - Y;
    nn  = numel(Y);
    m.mae   = mean(abs(res));
    m.rmse  = sqrt(mean(res.^2));
    m.bias  = mean(res);
    m.sdres = std(res, 0);
    SSE = sum(res.^2);
    SST = sum((Y - mean(Y)).^2);
    m.r2oof = 1 - SSE / SST;                 % 可為負
    m.see   = sqrt(SSE / max(nn - 2, 1));
    m.medae = median(abs(res));
    m.p90   = prctile(abs(res), 90);
    m.p95   = prctile(abs(res), 95);
    m.maxae = max(abs(res));
    pos = Y > 0;
    m.mape = mean(abs(res(pos) ./ Y(pos))) * 100;
    m.mpe  = mean(res(pos) ./ Y(pos)) * 100;

    if nn >= 3
        m.r   = corr(Y, Yp);
        m.rho = corr(Y, Yp, 'Type', 'Spearman');
    else
        m.r = NaN; m.rho = NaN;
    end
    m.r2 = m.r^2;                            % 與 v3 的 R_Square 定義相同

    % Lin's CCC（母體動差）
    mx = mean(Y); my = mean(Yp);
    vx = mean((Y - mx).^2); vy = mean((Yp - my).^2);
    cxy = mean((Y - mx) .* (Yp - my));
    m.ccc = 2*cxy / (vx + vy + (mx - my)^2);

    % ICC(2,1)：two-way random effects, absolute agreement, single measure
    m.icc = icc21([Y Yp]);

    % Bland-Altman 95% 一致性界線
    m.loaLo = m.bias - 1.96 * m.sdres;
    m.loaHi = m.bias + 1.96 * m.sdres;

    % 校準：Y ~ a + b*Yp，理想 b=1、a=0
    vp = var(Yp);
    if vp > 0
        c = cov(Yp, Y);
        m.calSlope = c(1,2) / vp;
        m.calInt   = mean(Y) - m.calSlope * mean(Yp);
    else
        m.calSlope = NaN; m.calInt = NaN;
    end

    % 比例偏誤：差值 vs 均值的斜率(Bland-Altman trend)
    avgv = (Y + Yp) / 2;
    if var(avgv) > 0
        cc = cov(avgv, res);
        m.propSlope = cc(1,2) / var(avgv);
    else
        m.propSlope = NaN;
    end
end


function v = icc21(Mx)
% ICC(2,1)：n 個受試 × k 個「量測者」(此處 k=2：實際 vs 預測)
    [nn, kk] = size(Mx);
    if nn < 2 || kk < 2, v = NaN; return; end
    gm = mean(Mx(:));
    rowM = mean(Mx, 2);
    colM = mean(Mx, 1);
    SSR = kk * sum((rowM - gm).^2);
    SSC = nn * sum((colM - gm).^2);
    SST = sum((Mx(:) - gm).^2);
    SSE = SST - SSR - SSC;
    MSR = SSR / (nn - 1);
    MSC = SSC / (kk - 1);
    MSE = SSE / ((nn - 1) * (kk - 1));
    v = (MSR - MSE) / (MSR + (kk - 1)*MSE + kk*(MSC - MSE)/nn);
end


%% ============================================================
%  延伸指標表
%  ============================================================

function T = buildExtTable(M, L, hasLin, CFG, foldSpread, showOver)
    rows = {
    % field          顯示名稱                               單位    小數  說明
    'mae',          'MAE 平均絕對誤差',                     'mm',   3, '最常報的點估計';
    'medae',        'MedAE 中位絕對誤差',                   'mm',   3, 'n 小、怕離群值主導 MAE 時一起報';
    'p90',          'P90 絕對誤差',                         'mm',   3, '90% 的樣本誤差在此以下';
    'p95',          'P95 絕對誤差',                         'mm',   3, '尾端風險';
    'maxae',        'MaxAE 最大絕對誤差',                   'mm',   3, '最壞情況';
    'rmse',         'RMSE',                                 'mm',   3, '對大誤差敏感；RMSE/MAE 越大表示誤差分布越長尾';
    'sdres',        'SD_res 殘差標準差',                    'mm',   3, 'Bland-Altman 的 SD_diff';
    'see',          'SEE 估計標準誤',                       'mm',   3, 'sqrt(SSE/(n-2))';
    'bias',         'Mean Bias 系統偏移',                   'mm',   3, '正=高估';
    'loaLo',        'LoA 下界 (bias-1.96SD)',               'mm',   3, 'Bland-Altman 95% 一致性界線';
    'loaHi',        'LoA 上界 (bias+1.96SD)',               'mm',   3, '臨床上要問：這個寬度可接受嗎';
    'propSlope',    '比例偏誤斜率 (差值 vs 均值)',          '-',    4, '≠0 表示誤差隨長度變化';
    'mape',         'MAPE 平均絕對百分誤差',                '%',    2, '相對誤差，跨不同牙長可比';
    'mpe',          'MPE 平均百分誤差',                     '%',    2, '相對系統偏移';
    'r',            'R (Pearson)',                          '-',    3, '含方向的線性相關';
    'r2',           'r^2 (= R_Square，同 v3)',              '-',    3, '只看相關，不懲罰 bias/斜率';
    'r2oof',        'R2_OOF = 1 - SSE/SST',                 '-',    3, '真正的解釋變異比例，可為負 ← 論文應以此為主';
    'rho',          'Spearman rho',                         '-',    3, '單調相關，對離群值穩健';
    'ccc',          'CCC (Lin 一致性)',                     '-',    3, '同時懲罰相關不足與校準偏差';
    'icc',          'ICC(2,1) 絕對一致性',                  '-',    3, '把預測當另一個量測者';
    'calSlope',     '校準斜率 (Y ~ a+b*Ŷ)',                 '-',    3, '理想 1；<1 = 高值低估、低值高估(迴歸向均值)';
    'calInt',       '校準截距',                             'mm',   3, '理想 0';
    'offMae',       '校正後 MAE',                           'mm',   3, 'offset 後，臨床用的主指標';
    'offRmse',      '校正後 RMSE',                          'mm',   3, '';
    'offBias',      '校正後 Mean Bias',                     'mm',   3, '理想接近 0 或略負(偏短較安全)';
    'offSdres',     '校正後 SD_res',                        'mm',   3, 'offset 不改變，供對照';
    'offLoaLo',     '校正後 LoA 下界',                      'mm',   3, '';
    'offLoaHi',     '校正後 LoA 上界',                      'mm',   3, '';
    'offCcc',       '校正後 CCC',                           '-',    3, '';
    'offIcc',       '校正後 ICC(2,1)',                      '-',    3, '';
    'offR2oof',     '校正後 R2_OOF',                        '-',    3, '';
    'idealTight',   sprintf('命中率 ±%.1fmm', CFG.tight),   '%',    1, '更嚴格門檻';
    'ideal',        sprintf('理想比率 ±%.1fmm', CFG.tol),   '%',    1, '同 v3';
    'idealLoose',   sprintf('命中率 ±%.1fmm', CFG.loose),   '%',    1, '較寬門檻';
    'over',         '過長率 (>0mm)',                        '%',    1, '臨床風險主指標';
    'overSevere',   sprintf('嚴重過長率 (>%.1fmm)', CFG.severe), '%', 1, '風險分級';
    'under',        sprintf('過短率 (<-%.1fmm)', CFG.tol),  '%',    1, '治療不足';
    };

    if ~showOver
        rows(ismember(rows(:,1), {'over','overSevere'}), :) = [];
    end

    nR = size(rows, 1);
    name = strings(nR,1); unit = strings(nR,1); note = strings(nR,1);
    aM = nan(nR,1); aS = nan(nR,1); lM = nan(nR,1); lS = nan(nR,1);
    for i = 1:nR
        f = rows{i,1}; d = rows{i,4};
        name(i) = string(rows{i,2});
        unit(i) = string(rows{i,3});
        note(i) = string(rows{i,5});
        aM(i) = round(mean(M.(f), 'omitnan'), d);
        aS(i) = round(std(M.(f), 0, 'omitnan'), d);
        if hasLin
            lM(i) = round(mean(L.(f), 'omitnan'), d);
            lS(i) = round(std(L.(f), 0, 'omitnan'), d);
        end
    end

    T = table(name, unit, aM, aS, lM, lS, note, 'VariableNames', ...
        {'指標', '單位', 'ANN_平均', 'ANN_重複SD', '線性_平均', '線性_重複SD', '說明'});

    T = [T; table("摺間 MAE 標準差", "mm", round(foldSpread,3), NaN, NaN, NaN, ...
        "各摺難易度差異", 'VariableNames', T.Properties.VariableNames)];
end


%% ============================================================
%  bootstrap CI 與配對檢定
%  ============================================================

function T = buildCITable(Y, Yp, Lp, hasLin, CFG, B, seed, alpha, showOver)
    off = CFG.offset; tol = CFG.tol;
    % 報告以校正後為主，故主要統計量都套 offset；R/r^2 offset 不變故只列一次
    stats = {
    '校正後 MAE',       @(y,q) mean(abs(q-off-y)),                          'mm';
    '校正後 RMSE',      @(y,q) sqrt(mean((q-off-y).^2)),                    'mm';
    '校正後 Mean Bias', @(y,q) mean(q-off-y),                               'mm';
    '校正後 R2_OOF',    @(y,q) 1-sum((q-off-y).^2)/sum((y-mean(y)).^2),     '-';
    'R (Pearson, offset不影響)', @(y,q) safeCorr(y,q),                      '-';
    'r^2 (offset不影響)',        @(y,q) safeCorr(y,q)^2,                    '-';
    '校正後 CCC',       @(y,q) cccFun(y,q-off),                             '-';
    '校正後 ICC(2,1)',  @(y,q) icc21([y q-off]),                            '-';
    sprintf('理想比率±%.1fmm', tol), @(y,q) mean(abs(q-off-y)<=tol)*100,    '%';
    '過長率',           @(y,q) mean((q-off-y)>0)*100,                       '%';
    };

    if ~showOver
        stats(strcmp(stats(:,1), '過長率'), :) = [];
    end

    rng(seed);
    nS = size(stats,1);
    nm = strings(nS,1); un = strings(nS,1);
    aP = nan(nS,1); aLo = nan(nS,1); aHi = nan(nS,1);
    lP = nan(nS,1); lLo = nan(nS,1); lHi = nan(nS,1);
    for i = 1:nS
        f = stats{i,2};
        nm(i) = string(stats{i,1});
        un(i) = string(stats{i,3});
        aP(i) = f(Y, Yp);
        ci = bootCI(f, Y, Yp, B, alpha);
        aLo(i) = ci(1); aHi(i) = ci(2);
        if hasLin && ~all(isnan(Lp))
            lP(i) = f(Y, Lp);
            ci = bootCI(f, Y, Lp, B, alpha);
            lLo(i) = ci(1); lHi(i) = ci(2);
        end
    end

    T = table(nm, un, round(aP,3), round(aLo,3), round(aHi,3), ...
              round(lP,3), round(lLo,3), round(lHi,3), ...
        'VariableNames', {'指標','單位','ANN_點估計','ANN_CI下界','ANN_CI上界', ...
                          '線性_點估計','線性_CI下界','線性_CI上界'});

    % --- ANN vs 線性：逐筆絕對誤差配對比較 ---
    if hasLin && ~all(isnan(Lp))
        eA = abs(Yp - off - Y);
        eL = abs(Lp - off - Y);
        d  = eL - eA;                      % >0 表示 ANN 較好
        dMAE = mean(d);
        rng(seed + 1);
        ciD = bootCI(@(y,q) mean(abs(q(:,2)-off-y) - abs(q(:,1)-off-y)), ...
                     Y, [Yp Lp], B, alpha);
        pW = NaN;
        if exist('signrank','file') == 2 && any(d ~= 0)
            pW = signrank(eA, eL);
        end
        nWin = sum(d > 0);
        extra = table( ...
            ["校正後MAE 差 (線性-ANN)"; "Wilcoxon signed-rank p"; "ANN 較準的樣本數"], ...
            ["mm"; "-"; "筆"], ...
            [round(dMAE,3); round(pW,4); nWin], ...
            [round(ciD(1),3); NaN; NaN], ...
            [round(ciD(2),3); NaN; numel(Y)], ...
            [NaN; NaN; NaN], [NaN; NaN; NaN], [NaN; NaN; NaN], ...
            'VariableNames', T.Properties.VariableNames);
        T = [T; extra];
    end
end


function ci = bootCI(f, Y, Yp, B, alpha)
    nn = numel(Y);
    v = nan(B,1);
    for b = 1:B
        idx = randi(nn, nn, 1);
        try
            v(b) = f(Y(idx), Yp(idx,:));
        catch
            v(b) = NaN;
        end
    end
    ci = prctile(v, [100*alpha/2, 100*(1-alpha/2)]);
end


function v = safeCorr(y, q)
    if std(y) == 0 || std(q) == 0
        v = NaN;
    else
        v = corr(y, q);
    end
end


function v = cccFun(y, q)
    mx = mean(y); my = mean(q);
    vx = mean((y-mx).^2); vy = mean((q-my).^2);
    cxy = mean((y-mx).*(q-my));
    v = 2*cxy / (vx + vy + (mx-my)^2);
end


%% ============================================================
%  沿用 v3 的小工具
%  ============================================================

function v = meanOr(x, flag)
    if flag
        v = mean(x, 'omitnan');
    else
        v = NaN;
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
