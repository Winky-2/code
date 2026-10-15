function C1g2_ablation_frontend_backend()
%% C1g2_ablation_frontend_backend.m   (論文一的消融表：前端 × 後端)
% ============================================================
%              |  ANN              |  GPR
%     pose     |  pose+ANN         |  pose+GPR(論文二)
%     seg      |  seg+ANN(論文一)   |  seg+GPR
%
% 回答審稿人會問的：MAE 從論文二降到論文一，是前端(pose→seg)
% 還是後端(GPR→ANN)造成的？
%   前端效果 = 固定後端，pose 換 seg 的 ΔMAE(兩個後端各算一次)
%   後端效果 = 固定前端，GPR 換 ANN 的 ΔMAE(兩個前端各算一次)
%
% 控制變因：
%   - 兩份輸入依檔名編號取交集，只用兩邊都有的牙
%   - 摺一律用 seg 檔的「折數fold」(pose 列重排成 seg 的順序)
%   - 特徵、標準化、offset 四格完全相同
%   - ANN 設定與亂數順序同 C1 v6 → 若交集沒少任何 seg 列，seg+ANN 的數字
%     應與 C1 v6(METHOD = METHOD_SEG、同 USE_AGE/USE_SEX)逐位相同 ← 驗證用
%
% ⚠️ 本表 pose+GPR 可能跟 C1g 單獨跑的數字不同：C1g 用 pose 檔自己的摺與
%    全部列；這裡用交集 + seg 的摺。論文二報 C1g 的，論文一消融表報這裡的。
% ⚠️ METHOD_SEG 預設 'mask幾何'(原圖 px)，跟 pose 的原圖 px 長度對等。
%    '冠寬比例尺' 用到 seg 才有的冠寬，pose 沒有對應版本——拿它比會把
%    「冠寬正規化」混進前端效果。
% ⚠️ GPR 與線性迴歸是決定性的，重複次數 = 1，SD 欄為 0 不是算錯。
%
% 成對檢定：每顆牙的絕對誤差(ANN 取 N_REPEATS 次的平均)做 Wilcoxon
%   signed-rank。CV 下各牙誤差不完全獨立，p 值當參考。
%
% 需要：Statistics and Machine Learning Toolbox、Deep Learning Toolbox、
%       同資料夾的 C1g_lib.m
% 輸出：C1g2_<run tag>.xlsx / _console.log(只新增不覆寫)
% ============================================================

    %% ---------------- 輸入設定區 ----------------
    METHOD_POSE = 'pose原始預測';   % 👈 與 C1g 的 METHOD 相同
    METHOD_SEG  = 'mask幾何';
    PREFIX      = '根管填充物像素長度_已配對_';
    FOLD_COLUMN = '折數fold';
    POS_COLUMN  = '牙位';

    %% ---------------- 特徵設定區(四格共用) ----------------
    USE_AGE = false;
    USE_SEX = true;
    AGE_COLUMN = '年齡';
    SEX_COLUMN = '性別';

    %% ---------------- ANN 設定區(同 C1 v6) ----------------
    N_REPEATS   = 10;
    BASE_SEED   = 42;
    HIDDEN_SIZE = 5;
    TRAIN_FCN   = 'trainbr';
    MAX_EPOCHS  = 200;

    %% ---------------- GPR 設定區(同 C1g) ----------------
    GPR_KERNEL = 'squaredexponential';
    GPR_BASIS  = 'linear';

    %% ---------------- 臨床 / 報告設定區 ----------------
    OFFSET_MM          = 0.5;
    IDEAL_TOLERANCE_MM = 1.0;
    REPORT_WITH_OFFSET = true;     % 2×2 表、成對檢定用校正後誤差(與 C1 v6 報告一致)

    %% --- 0. 準備 ---
    L = C1g_lib();
    FEAT_TAG = L.featTag(USE_AGE, USE_SEX);
    tag = sprintf('pose-vs-seg_%s%s_%s', METHOD_SEG, FEAT_TAG, L.runTag());
    diary(['C1g2_' tag '_console.log']);
    cleanupDiary = onCleanup(@() diary('off')); %#ok<NASGU>

    if REPORT_WITH_OFFSET
        off = OFFSET_MM; hf = 'offMae'; hLabel = sprintf('校正後(offset=-%.2fmm)', OFFSET_MM);
    else
        off = 0;         hf = 'mae';    hLabel = '未校正';
    end

    %% --- 1. 讀取 + 對齊 ---
    cfg = struct('AGE_COLUMN', AGE_COLUMN, 'SEX_COLUMN', SEX_COLUMN, ...
                 'FOLD_COLUMN', FOLD_COLUMN, 'POS_COLUMN', POS_COLUMN, ...
                 'USE_AGE', USE_AGE, 'USE_SEX', USE_SEX, 'SCORE_COLUMN', '');
    Dseg  = L.loadC1Input([PREFIX METHOD_SEG '.xlsx'],  cfg);
    Dpose = L.loadC1Input([PREFIX METHOD_POSE '.xlsx'], cfg);
    [Dseg, Dpose, info] = L.alignByName(Dseg, Dpose);

    n = info.nCommon;
    Y = Dseg.Y;
    fold = Dseg.fold;
    K = max(fold);

    fprintf('=== C1g2：前端(pose / seg) × 後端(ANN / GPR) ===\n');
    fprintf('seg 檔 %d 筆、pose 檔 %d 筆 → 交集 %d 筆\n', info.nA, info.nB, n);
    if info.nCommon < info.nA
        fprintf('⚠️ seg 有 %d 筆不在 pose 檔：seg+ANN 不會等於 C1 v6(n 不同)。\n', ...
                info.nA - info.nCommon);
    else
        fprintf('✅ seg 列沒有減少：seg+ANN 應與 C1 v6(METHOD=%s%s)逐位相同，可拿來驗證。\n', ...
                METHOD_SEG, FEAT_TAG);
    end
    if ~info.foldAgree
        fprintf('ℹ️  pose 檔原本的摺與 seg 不同，已一律改用 seg 的摺。\n');
    end
    fprintf('%d 摺，每摺樣本數：', K);
    fprintf('%d ', accumarray(fold(fold > 0), 1)); fprintf('\n');

    [XP, featNames] = L.buildFeatures(Dpose, USE_AGE, USE_SEX);
    XS              = L.buildFeatures(Dseg,  USE_AGE, USE_SEX);
    featDesc = strjoin(featNames, ' + ');
    fprintf('輸入特徵(四格相同)：%s\n', featDesc);

    %% --- 2. 交叉驗證 ---
    PA = struct('N_REPEATS', N_REPEATS, 'BASE_SEED', BASE_SEED, ...
                'HIDDEN_SIZE', HIDDEN_SIZE, 'TRAIN_FCN', TRAIN_FCN, 'MAX_EPOCHS', MAX_EPOCHS);
    PG = struct('GPR_KERNEL', GPR_KERNEL, 'GPR_BASIS', GPR_BASIS);

    fprintf('\n[pose] ANN %d 摺 × %d 次...\n', K, N_REPEATS);
    annP = L.cvANN(XP, Y, fold, PA);
    fprintf('[seg]  ANN %d 摺 × %d 次...\n', K, N_REPEATS);
    annS = L.cvANN(XS, Y, fold, PA);     % 每次重複都重設 rng → 不受上面 pose 影響

    fprintf('GPR / 線性 / 基準...\n');
    gprP = L.cvGPR(XP, Y, fold, PG);
    gprS = L.cvGPR(XS, Y, fold, PG);
    linP = L.cvLinear(XP, Y, fold);
    linS = L.cvLinear(XS, Y, fold);
    meanPred = L.cvMean(Y, fold);
    archPred = L.cvArchMean(Y, fold, Dseg.arch);

    %% --- 3. 消融表 ---
    E = {  % 前端, 後端, 預測矩陣(n × 重複)
        'pose',          'ANN',            annP;
        'pose',          'GPR',            gprP;
        'seg',           'ANN',            annS;
        'seg',           'GPR',            gprS;
        'pose',          '線性迴歸',        linP;
        'seg',           '線性迴歸',        linS;
        '—(不看影像)',   '猜平均',          meanPred;
        '—(不看影像)',   '上下顎各自平均',   archPred };
    nE = size(E, 1);

    v = @(x, d) round(x, d);
    cols = zeros(nE, 11);
    for i = 1:nE
        M = L.repMetrics(E{i, 3}, Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
        cols(i, :) = [size(E{i, 3}, 2), ...
            v(mean(M.mae), 3),    v(std(M.mae), 3), ...
            v(mean(M.offMae), 3), v(std(M.offMae), 3), ...
            v(mean(M.offRmse), 3), v(mean(M.offBias), 3), ...
            v(mean(M.r), 3), v(mean(M.r2), 3), v(mean(M.ideal), 1), ...
            v(mean(M.(hf)), 3)];
    end
    AblTable = table(string(E(:, 1)), string(E(:, 2)), repmat(n, nE, 1), ...
        cols(:, 1), cols(:, 2), cols(:, 3), cols(:, 4), cols(:, 5), ...
        cols(:, 6), cols(:, 7), cols(:, 8), cols(:, 9), cols(:, 10), ...
        'VariableNames', {'前端', '後端', 'n', '重複次數', ...
        'MAE_未校正_mm', 'MAE_未校正_SD', 'MAE_校正後_mm', 'MAE_校正後_SD', ...
        'RMSE_校正後_mm', 'Bias_校正後_mm', 'R_Pearson', 'R_Square', '理想比率_校正後_pct'});
    head = cols(:, 11);          % 主結果欄(依 REPORT_WITH_OFFSET)

    Grid = table(["pose"; "seg"], [head(1); head(3)], [head(2); head(4)], ...
        'VariableNames', {'前端', 'ANN_MAE_mm', 'GPR_MAE_mm'});

    %% --- 4. 每顆牙的絕對誤差 + 成對檢定 ---
    ae = @(pm) mean(abs(pm - off - Y), 2);       % ANN：先逐次取絕對值再平均
    PT = struct('poseANN', ae(annP), 'poseGPR', ae(gprP), ...
                'segANN',  ae(annS), 'segGPR',  ae(gprS));

    comps = {
        '前端效果(固定 ANN)：pose → seg',  'poseANN', 'segANN';
        '前端效果(固定 GPR)：pose → seg',  'poseGPR', 'segGPR';
        '後端效果(固定 pose)：GPR → ANN',  'poseGPR', 'poseANN';
        '後端效果(固定 seg)：GPR → ANN',   'segGPR',  'segANN';
        '論文二 → 論文一(前後端都換)',      'poseGPR', 'segANN' };
    nC = size(comps, 1);
    cDelta = zeros(nC, 1); cMed = zeros(nC, 1); cBetter = zeros(nC, 1); cP = zeros(nC, 1);
    cFrom = zeros(nC, 1); cTo = zeros(nC, 1);
    for i = 1:nC
        a = PT.(comps{i, 2});
        b = PT.(comps{i, 3});
        cFrom(i)   = mean(a);
        cTo(i)     = mean(b);
        cDelta(i)  = mean(b - a);
        cMed(i)    = median(b - a);
        cBetter(i) = sum(b < a);
        cP(i)      = signrank(a, b);
    end
    PairTable = table(string(comps(:, 1)), string(comps(:, 2)), string(comps(:, 3)), ...
        round(cFrom, 3), round(cTo, 3), round(cDelta, 3), round(cMed, 3), ...
        cBetter, round(cP, 4), ...
        'VariableNames', {'比較', '從', '到', 'MAE_從_mm', 'MAE_到_mm', ...
        'ΔMAE_到減從_mm', '逐牙差中位數_mm', '變好的牙數', 'p_Wilcoxon'});

    ToothTable = table(Dseg.ids, fold, Dseg.pos, Y, round(PT.poseANN, 3), round(PT.poseGPR, 3), ...
        round(PT.segANN, 3), round(PT.segGPR, 3), ...
        'VariableNames', {'檔名_seg', '摺別', '牙位', '實際長度_mm', ...
        'pose_ANN_絕對誤差', 'pose_GPR_絕對誤差', 'seg_ANN_絕對誤差', 'seg_GPR_絕對誤差'});

    m = max(numel(info.onlyA), numel(info.onlyB));
    padS = @(s) [s(:); strings(m - numel(s), 1)];
    AlignTable = table(padS(info.onlyA), padS(info.onlyB), ...
        'VariableNames', {'只在seg檔_檔名編號', '只在pose檔_檔名編號'});

    SettingsTable = table( ...
        ["pose 輸入"; "seg 輸入"; "交集 n"; "摺來源"; "輸入特徵"; "ANN"; "GPR"; ...
         "offset_mm"; "主結果用校正後"; "腳本版本"; "run tag"], ...
        [string(Dpose.file); string(Dseg.file); string(n); "seg 檔的 " + FOLD_COLUMN; ...
         string(featDesc); ...
         string(sprintf('fitnet(%d), %s, %d epochs, %d 次重複, seed %d', ...
                HIDDEN_SIZE, TRAIN_FCN, MAX_EPOCHS, N_REPEATS, BASE_SEED)); ...
         string(sprintf('%s kernel, %s basis', GPR_KERNEL, GPR_BASIS)); ...
         string(OFFSET_MM); string(REPORT_WITH_OFFSET); "C1g2_v1"; string(tag)], ...
        'VariableNames', {'項目', '值'});

    %% --- 5. 匯出 ---
    outXlsx = ['C1g2_' tag '.xlsx'];
    if isfile(outXlsx), delete(outXlsx); end
    writetable(AblTable,      outXlsx, 'Sheet', '消融表');
    writetable(Grid,          outXlsx, 'Sheet', '2x2_MAE');
    writetable(PairTable,     outXlsx, 'Sheet', '成對檢定');
    writetable(ToothTable,    outXlsx, 'Sheet', '逐牙絕對誤差');
    writetable(AlignTable,    outXlsx, 'Sheet', '對齊排除');
    writetable(SettingsTable, outXlsx, 'Sheet', '設定');

    %% --- 6. 終端報告 ---
    fprintf('\n✅ 已匯出：%s\n', outXlsx);
    fprintf('\n=== 消融表(out-of-fold, n=%d) ===\n', n);
    disp(AblTable);

    fprintf('\n=== 2×2 MAE(%s) ===\n', hLabel);
    fprintf('             ANN      GPR\n');
    fprintf('  pose   %7.3f  %7.3f\n', head(1), head(2));
    fprintf('  seg    %7.3f  %7.3f\n', head(3), head(4));
    fprintf('  基準：猜平均 %.3f | 上下顎平均 %.3f mm\n', head(7), head(8));

    frontANN = head(3) - head(1);  frontGPR = head(4) - head(2);
    backPose = head(1) - head(2);  backSeg  = head(3) - head(4);
    fprintf('\n前端效果(pose→seg)：ANN %+.3f、GPR %+.3f mm   (負 = seg 較好)\n', frontANN, frontGPR);
    fprintf('後端效果(GPR→ANN)：pose %+.3f、seg %+.3f mm   (負 = ANN 較好)\n', backPose, backSeg);
    fe = mean(abs([frontANN frontGPR]));
    be = mean(abs([backPose backSeg]));
    if fe > be
        fprintf('👉 改善主要來自「前端」(平均 |Δ| %.3f vs 後端 %.3f mm)\n', fe, be);
    else
        fprintf('👉 改善主要來自「後端」(平均 |Δ| %.3f vs 前端 %.3f mm)\n', be, fe);
    end
    if sign(frontANN) ~= sign(frontGPR) || sign(backPose) ~= sign(backSeg)
        fprintf('⚠️ 兩個後端(或前端)的效果方向不一致 → 有交互作用，論文不能只報單一方向。\n');
    end

    fprintf('\n=== 成對檢定(逐牙絕對誤差，Wilcoxon signed-rank) ===\n');
    disp(PairTable);
    fprintf('p 值當參考：n=%d、CV 下各牙誤差不完全獨立、未做多重比較校正。\n', n);
end
