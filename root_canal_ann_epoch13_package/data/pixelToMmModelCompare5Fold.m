function pixelToMmModelCompare5Fold(maxEpochs, outputDir, inputFile, groupByPatient)
% 多模型比較：traced pixel length -> reference mm（治療後 31 筆）
% 參數完全沿用 pixelToMmPredictorANN5Fold.m：
%   rng(42,'twister') → cvpartition(n,'KFold',5)，單次切分
%   每摺只用訓練部分做 z-score；fitnet(5,'trainbr')、dividetrain、processFcns 全關、epochs=200
%   R2 = 1 - SSE/SST（同 calculateMetrics）
% ANN 的亂數消耗順序與原腳本相同（切分 → 5 摺 → 全資料模型），之後才跑其他模型，
%   所以 ANN 那一列應與原腳本輸出逐位相同（同一 MATLAB 版本下）。
% 所有模型共用同一組摺 → 模型是唯一變因。其他模型超參數固定，不在摺內調參。
% groupByPatient=true 時改為依病歷號分摺（同一人的牙不會同時在 train/test）；預設 false 以對齊原腳本。
% 需要：Deep Learning Toolbox、Statistics and Machine Learning Toolbox

    if nargin < 1 || isempty(maxEpochs),      maxEpochs = 200; end
    if nargin < 2 || isempty(outputDir),      outputDir = fullfile(pwd, 'outputs', 'model_compare_5fold'); end
    if nargin < 3 || isempty(inputFile),      inputFile = fullfile(pwd, '預測結果與評估指標_像素轉mm.xlsx'); end
    if nargin < 4 || isempty(groupByPatient), groupByPatient = false; end
    hiddenUnits = 5;
    K = 5;

    if ~exist(outputDir, 'dir'), mkdir(outputDir); end

    % ---------- 讀資料（同原腳本：第 1~3 欄 = 病歷號 / px / mm）----------
    data = readtable(inputFile, 'VariableNamingRule', 'preserve');
    ids = string(data{:, 1});
    X = double(data{:, 2});
    Y = double(data{:, 3});
    valid = isfinite(X) & isfinite(Y);
    ids = ids(valid); X = X(valid); Y = Y(valid);
    n = numel(Y);
    fprintf('載入 %d 筆、%d 位病人\n', n, numel(unique(ids)));

    % ---------- 切分（亂數順序同原腳本）----------
    rng(42, 'twister');
    if groupByPatient
        [~, ~, g] = unique(ids, 'stable');
        gp = cvpartition(max(g), 'KFold', K);
        foldId = zeros(n, 1);
        for f = 1:K, foldId(ismember(g, find(test(gp, f)))) = f; end
    else
        partition = cvpartition(n, 'KFold', K);
        foldId = zeros(n, 1);
        for f = 1:K, foldId(test(partition, f)) = f; end
    end

    names = {'ANN(fitnet5,trainbr)', '線性', 'Huber穩健線性', 'Ridge(k=1)', '二次多項式', ...
             'GPR(SE核+線性基底)', 'SVR(RBF)', 'KNN(k=5)', '隨機森林', '梯度提升(LSBoost)', ...
             '比例尺 k·px', '平均值(基準)'};
    M = numel(names);
    P = nan(n, M);

    % ---------- 1) ANN：與原腳本相同的流程與亂數順序 ----------
    for f = 1:K
        tr = foldId ~= f; te = foldId == f;
        [xm, xs] = deal(mean(X(tr)), std(X(tr), 0));
        net = makeNet(hiddenUnits, maxEpochs);
        net = train(net, ((X(tr) - xm) ./ xs)', Y(tr)');
        P(te, 1) = net(((X(te) - xm) ./ xs)')';
    end
    finalNet = makeNet(hiddenUnits, maxEpochs);          % 部署模型：全 31 筆重訓（同原腳本）
    finalNet = train(finalNet, ((X - mean(X)) ./ std(X, 0))', Y'); %#ok<NASGU>

    % ---------- 2) 其他模型（同一組摺）----------
    for f = 1:K
        tr = foldId ~= f; te = foldId == f;
        xm = mean(X(tr)); xs = std(X(tr), 0);
        ztr = (X(tr) - xm) ./ xs;  zte = (X(te) - xm) ./ xs;
        ytr = Y(tr);
        rng(42 + f, 'twister');

        P(te, 2)  = predict(fitlm(ztr, ytr), zte);
        P(te, 3)  = predict(fitlm(ztr, ytr, 'RobustOpts', 'huber'), zte);
        b = ridge(ytr, ztr, 1, 0);                    P(te, 4) = b(1) + b(2) * zte;
        P(te, 5)  = predict(fitlm(ztr, ytr, 'quadratic'), zte);
        P(te, 6)  = predict(fitrgp(ztr, ytr, 'KernelFunction', 'squaredexponential', ...
                        'BasisFunction', 'linear', 'Standardize', false), zte);
        P(te, 7)  = predict(fitrsvm(ztr, ytr, 'KernelFunction', 'gaussian', ...
                        'BoxConstraint', 10, 'Epsilon', 0.5), zte);
        idx = knnsearch(ztr, zte, 'K', 5);            P(te, 8) = mean(ytr(idx), 2);
        rf = TreeBagger(300, ztr, ytr, 'Method', 'regression', 'MinLeafSize', 3);
        P(te, 9)  = predict(rf, zte);
        gb = fitrensemble(ztr, ytr, 'Method', 'LSBoost', 'NumLearningCycles', 100, ...
                        'LearnRate', 0.05, 'Learners', templateTree('MaxNumSplits', 3));
        P(te, 10) = predict(gb, zte);
        P(te, 11) = median(ytr ./ X(tr)) * X(te);
        P(te, 12) = mean(ytr);
    end

    % ---------- 指標（同原腳本定義）----------
    T = table('Size', [M 7], 'VariableTypes', [{'string'}, repmat({'double'}, 1, 6)], ...
        'VariableNames', {'Model', 'MAE_mm', 'RMSE_mm', 'R2', 'Mean_Bias_mm', 'Within1p5mm_pct', 'dMAE_vs_Linear'});
    foldMAE = nan(K, M);
    for m = 1:M
        r = P(:, m) - Y;
        T.Model(m) = names{m};
        T.MAE_mm(m) = mean(abs(r));
        T.RMSE_mm(m) = sqrt(mean(r.^2));
        T.R2(m) = 1 - sum(r.^2) / sum((Y - mean(Y)).^2);
        T.Mean_Bias_mm(m) = mean(r);
        T.Within1p5mm_pct(m) = mean(abs(r) <= 1.5) * 100;
        for f = 1:K, foldMAE(f, m) = mean(abs(r(foldId == f))); end
    end
    T.dMAE_vs_Linear = T.MAE_mm - T.MAE_mm(2);
    T = sortrows(T, 'MAE_mm');

    FT = array2table(foldMAE, 'VariableNames', names);
    FT = addvars(FT, (1:K)', 'Before', 1, 'NewVariableNames', 'Fold');
    PT = array2table(P, 'VariableNames', names);
    PT = addvars(PT, ids, X, Y, foldId, 'Before', 1, ...
        'NewVariableNames', {'ID', 'Pixel_px', 'Ref_mm', 'Fold'});

    tag = 'ungrouped'; if groupByPatient, tag = 'byPatient'; end
    outXlsx = fullfile(outputDir, sprintf('模型比較_5fold_%s.xlsx', tag));
    if exist(outXlsx, 'file'), delete(outXlsx); end
    writetable(T,  outXlsx, 'Sheet', '模型評估指標');
    writetable(FT, outXlsx, 'Sheet', '逐摺MAE');
    writetable(PT, outXlsx, 'Sheet', 'OOF預測');

    disp(T);
    fprintf('ANN OOF MAE = %.3f mm（應與 pixelToMmPredictorANN5Fold 輸出相同）\n', mean(abs(P(:,1) - Y)));
    fprintf('✅ 已匯出：%s\n', outXlsx);
    makePlot(T, outputDir, tag);
end

function net = makeNet(hiddenUnits, maxEpochs)
    net = fitnet(hiddenUnits, 'trainbr');
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    net.trainParam.epochs = maxEpochs;
    net.divideFcn = 'dividetrain';
    net.inputs{1}.processFcns = {};
    net.outputs{2}.processFcns = {};
end

function makePlot(T, outputDir, tag)
    figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 600]);
    barh(flipud(T.MAE_mm), 'FaceColor', [0 0.25 0.85]);
    set(gca, 'YTickLabel', flipud(T.Model), 'TickLabelInterpreter', 'none');
    hold on;
    xline(T.MAE_mm(T.Model == "線性"), 'k--', 'LineWidth', 1.2);
    grid on; xlabel('OOF MAE (mm)'); title('5-fold OOF MAE by model');
    exportgraphics(gcf, fullfile(outputDir, sprintf('模型比較_MAE_%s.png', tag)), 'Resolution', 180);
    close(gcf);
end
