function C1g_pixelToMmPredictor_GPR()
%% C1g_pixelToMmPredictor_GPR.m   (論文二：pose + GPR)
% ============================================================
% 跟 C1 v6 同一種輸入格式、同一組摺，只把後端從 ANN 換成 GPR(高斯過程迴歸)。
% GPR 對每顆牙輸出「預測值 ± 標準差 σ」，本腳本多算三組 C1 沒有的指標：
%
%   1) 基準對照(同一組摺)：猜平均值、上下顎各自平均
%      影像方法要贏過這兩個，才能說影像有貢獻。
%   2) 不確定性：50/80/95% 預測區間的實際涵蓋率、95% 區間平均寬度、
%      |誤差| 與 σ 的 Spearman 相關
%   3) 轉診曲線(risk–coverage)：依 σ 由大到小轉診 k%，剩下案例的 MAE；
%      保留 80% 時的 MAE；讓剩下 MAE < 目標所需的最少轉診比例。
%      附 oracle(依真實誤差轉診 = 上限)與隨機轉診(= 整體 MAE)當參考線。
%
% ⚠️ 讀結果前要知道的限制(不是 bug)：
%   GPR 的 σ 只反映「這筆的特徵離訓練資料多遠」，不知道「這張片子的根尖
%   好不好認」。輸入只有像素長度(+性別)時，σ 大的只會是長度特別長或短
%   的牙，未必是誤差大的牙。所以 ρ 不顯著、轉診曲線貼著隨機線都是可能的
%   結果——那代表「用 σ 轉診」這個故事不成立，要改用影像品質分數。
%   → pose 檔有關鍵點信心度欄的話，填 TRIAGE_SCORE_COLUMN，會一起比。
%
% ⚠️ GPR 沒有隨機初始化，摺固定時重跑結果相同 → 不做重複，沒有重複間 SD。
%
% ⚠️ 核參數(輸入已標準化)：squaredexponential 的第 1 個是長度尺度 σL。
%    σL 遠大於 1(例如 >10)代表 GP 退化成「線性迴歸 + 常數誤差帶」，
%    這在 n≈60、單一特徵時很正常，不代表跑錯。
%
% 需要：Statistics and Machine Learning Toolbox(fitrgp)、同資料夾的 C1g_lib.m
% 輸出：C1g_<run tag>.xlsx / _轉診曲線.png / _console.log(只新增不覆寫)
% ============================================================

    %% ---------------- 輸入設定區 ----------------
    METHOD   = 'pose原始預測';      % 👈 pose 版 C1 輸入檔的後綴(要自己產生，見 L.loadC1Input 的錯誤訊息)
    filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];
    FOLD_COLUMN = '折數fold';
    POS_COLUMN  = '牙位';

    %% ---------------- 特徵設定區(與 C1 v6 對齊) ----------------
    USE_AGE = false;
    USE_SEX = true;
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    %% ---------------- GPR 設定區 ----------------
    GPR_KERNEL = 'squaredexponential';   % 'ardsquaredexponential' = 每個特徵各自的長度尺度
    GPR_BASIS  = 'linear';               % 平均函數用線性：GP 只學線性迴歸剩下的部分

    %% ---------------- 臨床 / 報告設定區 ----------------
    OFFSET_MM          = 0.5;
    IDEAL_TOLERANCE_MM = 1.0;
    REPORT_WITH_OFFSET = true;     % true = 主結果與轉診曲線用校正後誤差(與 C1 v6 報告一致)
                                   % 涵蓋率 / ρ 一律用未校正輸出(offset 不屬於模型分布)
    TARGET_MAE_MM      = 1.0;      % 轉診後要達到的 MAE(對齊論文一)
    REFER_GRID         = 0:0.05:0.50;
    RETAIN_FOR_TABLE   = 0.80;     % 主表報「保留 80% 案例」時的 MAE

    TRIAGE_SCORE_COLUMN      = ''; % 例：'關鍵點信心_B'；留空 = 只比 σ
    TRIAGE_HIGHER_IS_RISKIER = false;  % 信心度：越低越該轉診 → false

    SAVE_FIG = true;

    %% --- 0. 準備 ---
    L = C1g_lib();
    FEAT_TAG = L.featTag(USE_AGE, USE_SEX);
    tag = sprintf('GPR_%s%s_%s', METHOD, FEAT_TAG, L.runTag());
    diary(['C1g_' tag '_console.log']);
    cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

    %% --- 1. 讀取資料 ---
    cfg = struct('AGE_COLUMN', AGE_COLUMN, 'SEX_COLUMN', SEX_COLUMN, ...
                 'FOLD_COLUMN', FOLD_COLUMN, 'POS_COLUMN', POS_COLUMN, ...
                 'USE_AGE', USE_AGE, 'USE_SEX', USE_SEX, ...
                 'SCORE_COLUMN', TRIAGE_SCORE_COLUMN);
    D = L.loadC1Input(filename, cfg);
    n = D.n;
    [Xmat, featNames] = L.buildFeatures(D, USE_AGE, USE_SEX);
    featDesc = strjoin(featNames, ' + ');
    K = max(D.fold);
    hasScore = ~isempty(TRIAGE_SCORE_COLUMN);
    hasArch  = any(~isnan(D.arch));

    fprintf('=== C1g：%s + GPR ===\n', METHOD);
    fprintf('載入 %d 筆(剔除 %d 筆缺值)。輸入特徵：%s\n', n, D.nDropped, featDesc);
    fprintf('GPR：kernel = %s，basis = %s\n', GPR_KERNEL, GPR_BASIS);
    fprintf('沿用「%s」欄：%d 摺，每摺樣本數：', FOLD_COLUMN, K);
    fprintf('%d ', accumarray(D.fold(D.fold > 0), 1)); fprintf('\n');
    if ~hasArch
        fprintf('⚠️ 沒有「%s」欄或全部缺值 → 「上下顎各自平均」基準跳過。\n', POS_COLUMN);
    end

    %% --- 2. 交叉驗證(GPR + 三個對照) ---
    P = struct('GPR_KERNEL', GPR_KERNEL, 'GPR_BASIS', GPR_BASIS);
    fprintf('\n訓練中...\n');
    [gMu, gSd, gprInfo] = L.cvGPR(Xmat, D.Y, D.fold, P);
    linPred  = L.cvLinear(Xmat, D.Y, D.fold);
    meanPred = L.cvMean(D.Y, D.fold);
    archPred = L.cvArchMean(D.Y, D.fold, D.arch);

    %% --- 3. 點預測指標 ---
    rowNames = {'GPR'; '線性迴歸(同特徵)'; '基準：猜平均(訓練摺平均)'; '基準：上下顎各自平均'};
    preds = {gMu; linPred; meanPred; archPred};
    S = cellfun(@(p) L.pointMetrics(p, D.Y, OFFSET_MM, IDEAL_TOLERANCE_MM), ...
                preds, 'UniformOutput', false);
    col = @(fld, d) cellfun(@(s) round(s.(fld), d), S);

    PointTable = table(rowNames, col('mae', 3), col('offMae', 3), col('offRmse', 3), ...
        col('offBias', 3), col('r', 3), col('r2', 3), col('ideal', 1), ...
        'VariableNames', {'模型', 'MAE_未校正_mm', 'MAE_校正後_mm', 'RMSE_校正後_mm', ...
        'Bias_校正後_mm', 'R_Pearson', 'R_Square', '理想比率_校正後_pct'});

    if REPORT_WITH_OFFSET
        hf = 'offMae'; hLabel = sprintf('校正後(offset=-%.2fmm)', OFFSET_MM); off = OFFSET_MM;
    else
        hf = 'mae';    hLabel = '未校正';                                   off = 0;
    end

    %% --- 4. 不確定性指標(未校正) ---
    [UncTable, U] = L.uncertaintyMetrics(gMu, gSd, D.Y, [0.50 0.80 0.95]);

    %% --- 5. 轉診曲線 ---
    absErr = abs(gMu - off - D.Y);
    [rcG, nRef] = L.riskCoverage(absErr, gSd,    REFER_GRID, true);
    rcO         = L.riskCoverage(absErr, absErr, REFER_GRID, true);
    rcRand      = repmat(mean(absErr), numel(REFER_GRID), 1);

    RcTable = table(round(REFER_GRID(:) * 100), nRef, round(rcG, 3), round(rcRand, 3), ...
        round(rcO, 3), 'VariableNames', {'轉診比例_pct', '轉診筆數', ...
        '剩餘MAE_依GPRσ', '剩餘MAE_隨機(期望值)', '剩餘MAE_oracle(上限)'});
    if hasScore
        rcQ = L.riskCoverage(absErr, D.score, REFER_GRID, TRIAGE_HIGHER_IS_RISKIER);
        RcTable.(['剩餘MAE_依' TRIAGE_SCORE_COLUMN]) = round(rcQ, 3);
    end

    retG = L.riskCoverage(absErr, gSd,    1 - RETAIN_FOR_TABLE, true);
    retO = L.riskCoverage(absErr, absErr, 1 - RETAIN_FOR_TABLE, true);
    [fG, kG] = L.minReferForTarget(absErr, gSd,    TARGET_MAE_MM, true);
    [fO, kO] = L.minReferForTarget(absErr, absErr, TARGET_MAE_MM, true);

    sumItem = ["整體 MAE(不轉診)"; ...
               sprintf("保留 %.0f%% 時的 MAE：依 GPR σ", RETAIN_FOR_TABLE * 100); ...
               sprintf("保留 %.0f%% 時的 MAE：oracle(上限)", RETAIN_FOR_TABLE * 100); ...
               sprintf("剩餘 MAE < %.1f mm 所需最少轉診比例(%%)：依 GPR σ", TARGET_MAE_MM); ...
               sprintf("剩餘 MAE < %.1f mm 所需最少轉診比例(%%)：oracle", TARGET_MAE_MM)];
    sumVal  = [mean(absErr); retG; retO; fG * 100; fO * 100];
    sumNote = [hLabel; "隨機轉診的期望值 = 整體 MAE"; "理論最佳"; ...
               sprintf("轉 %s 筆；NaN = 轉到剩 1 筆都達不到", num2str(kG)); ...
               sprintf("轉 %s 筆", num2str(kO))];
    if hasScore
        retQ = L.riskCoverage(absErr, D.score, 1 - RETAIN_FOR_TABLE, TRIAGE_HIGHER_IS_RISKIER);
        [fQ, kQ] = L.minReferForTarget(absErr, D.score, TARGET_MAE_MM, TRIAGE_HIGHER_IS_RISKIER);
        sumItem = [sumItem; sprintf("保留 %.0f%% 時的 MAE：依 %s", RETAIN_FOR_TABLE * 100, TRIAGE_SCORE_COLUMN); ...
                   sprintf("剩餘 MAE < %.1f mm 所需最少轉診比例(%%)：依 %s", TARGET_MAE_MM, TRIAGE_SCORE_COLUMN)];
        sumVal  = [sumVal; retQ; fQ * 100];
        sumNote = [sumNote; ""; sprintf("轉 %s 筆", num2str(kQ))];
    end
    RcSummary = table(sumItem, round(sumVal, 3), sumNote, 'VariableNames', {'項目', '值', '說明'});

    %% --- 6. 逐筆表 ---
    z95 = norminv(0.975);
    lo95 = gMu - z95 * gSd;
    hi95 = gMu + z95 * gSd;
    [~, ordSd] = sort(gSd, 'descend');
    rk = zeros(n, 1); rk(ordSd) = 1:n;

    ResultsTable = table(D.ids, D.fold, D.pos, D.X, D.Y, round(gMu, 3), round(gSd, 3), ...
        round(lo95, 3), round(hi95, 3), (D.Y >= lo95 & D.Y <= hi95), ...
        round(abs(gMu - D.Y), 3), round(gMu - OFFSET_MM, 3), round(abs(gMu - OFFSET_MM - D.Y), 3), rk, ...
        'VariableNames', {'檔名', '摺別', '牙位', '像素長度', '實際長度_mm', 'GPR預測_mm', ...
        'GPR標準差σ_mm', '95區間下界_mm', '95區間上界_mm', '落在95區間內', ...
        '絕對誤差_未校正_mm', '校正後預測_mm', '絕對誤差_校正後_mm', '轉診順位_σ由大到小'});
    if hasScore
        ResultsTable.(TRIAGE_SCORE_COLUMN) = D.score;
    end

    SettingsTable = table( ...
        ["輸入檔"; "量測方法"; "輸入特徵"; "n"; "摺數"; "GPR kernel"; "GPR basis"; ...
         "offset_mm"; "主結果用校正後"; "目標MAE_mm"; "品質分數欄"; "腳本版本"; "run tag"], ...
        [string(filename); string(METHOD); string(featDesc); string(n); string(K); ...
         string(GPR_KERNEL); string(GPR_BASIS); string(OFFSET_MM); string(REPORT_WITH_OFFSET); ...
         string(TARGET_MAE_MM); string(TRIAGE_SCORE_COLUMN); "C1g_v1"; string(tag)], ...
        'VariableNames', {'項目', '值'});

    %% --- 7. 匯出 ---
    outXlsx = ['C1g_' tag '.xlsx'];
    if isfile(outXlsx), delete(outXlsx); end
    writetable(ResultsTable,  outXlsx, 'Sheet', '逐筆預測');
    writetable(PointTable,    outXlsx, 'Sheet', '點預測指標');
    writetable(UncTable,      outXlsx, 'Sheet', '不確定性指標');
    writetable(RcSummary,     outXlsx, 'Sheet', '轉診摘要');
    writetable(RcTable,       outXlsx, 'Sheet', '轉診曲線');
    writetable(gprInfo,       outXlsx, 'Sheet', 'GPR每摺超參數');
    writetable(SettingsTable, outXlsx, 'Sheet', '模型設定');

    if SAVE_FIG
        fig = figure('Visible', 'off', 'Color', 'w');
        hold on;
        x = REFER_GRID * 100;
        plot(x, rcG, '-o', 'LineWidth', 1.5, 'DisplayName', 'GPR σ 轉診');
        if hasScore
            plot(x, rcQ, '-s', 'LineWidth', 1.5, 'DisplayName', [TRIAGE_SCORE_COLUMN ' 轉診']);
        end
        plot(x, rcO, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.2, ...
             'DisplayName', 'Oracle(依真實誤差，上限)');
        yline(mean(absErr), ':', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.2, ...
              'DisplayName', '隨機轉診(期望值)');
        yline(TARGET_MAE_MM, '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.0, ...
              'DisplayName', sprintf('目標 %.1f mm', TARGET_MAE_MM));
        xlabel('轉診給醫師複查的比例 (%)');
        ylabel(sprintf('其餘案例 MAE (mm，%s)', hLabel));
        legend('Location', 'northeast'); grid on; box on;
        outPng = ['C1g_' tag '_轉診曲線.png'];
        exportgraphics(fig, outPng, 'Resolution', 200);
        close(fig);
    end

    %% --- 8. 終端報告 ---
    fprintf('\n✅ 已匯出：%s\n', outXlsx);
    if SAVE_FIG, fprintf('🖼️  %s\n', outPng); end

    fprintf('\n=== 點預測(out-of-fold, n=%d) ===\n', n);
    disp(PointTable);

    hG = S{1}.(hf); hL = S{2}.(hf); hM = S{3}.(hf); hA = S{4}.(hf);
    fprintf('[主結果：%s MAE]\n', hLabel);
    fprintf('  GPR %.3f | 線性 %.3f | 猜平均 %.3f | 上下顎平均 %.3f mm\n', hG, hL, hM, hA);
    if hG < hM
        fprintf('  ✅ GPR 贏過「猜平均」%.3f mm\n', hM - hG);
    else
        fprintf('  ❌ GPR 沒贏過「猜平均」→ MAE 不能當論文二的主結果\n');
    end
    if hasArch
        if hG < hA
            fprintf('  ✅ GPR 贏過「上下顎平均」%.3f mm\n', hA - hG);
        else
            fprintf('  ❌ GPR 沒贏過「上下顎平均」→ 影像目前沒有提供牙弓以外的資訊\n');
        end
    end
    fprintf('  GPR vs 線性：差 %.3f mm(差很小 = GP 退化成線性，看「GPR每摺超參數」的 σL)\n', hL - hG);

    fprintf('\n=== 不確定性(未校正輸出) ===\n');
    disp(UncTable);

    fprintf('\n=== 轉診(%s) ===\n', hLabel);
    disp(RcSummary);
    if ~(U.rho > 0 && U.p < 0.05)
        fprintf('⚠️ σ 與誤差沒有顯著正相關 → 「依 σ 轉診」目前站不住。\n');
        fprintf('   轉診曲線應該貼著隨機線；論文二要改用影像品質分數(TRIAGE_SCORE_COLUMN)，\n');
        fprintf('   或改用異質變異(heteroscedastic)模型。\n');
    end
    fprintf('\n👉 消融表(pose/seg × ANN/GPR)請跑 C1g2_ablation_frontend_backend.m\n');
end
