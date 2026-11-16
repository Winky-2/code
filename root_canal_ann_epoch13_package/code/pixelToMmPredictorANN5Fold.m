function pixelToMmPredictorANN5Fold(maxEpochs, outputDir, hiddenUnits, trainingFunction, regularization, inputFile)
% Strict 5-fold ANN regression for traced pixel length -> reference mm.
% Architecture follows the supplied approach: fitnet(5, 'trainbr').

    rootDir = fileparts(fileparts(mfilename('fullpath')));
    if nargin < 1 || isempty(maxEpochs)
        maxEpochs = 200;
    end
    if nargin < 2 || isempty(outputDir)
        outputDir = fullfile(rootDir, 'outputs', 'ann_fitnet_trainbr_5fold');
    end
    if nargin < 3 || isempty(hiddenUnits)
        hiddenUnits = 5;
    end
    if nargin < 4 || isempty(trainingFunction)
        trainingFunction = 'trainbr';
    end
    if nargin < 5 || isempty(regularization)
        regularization = 0;
    end
    if nargin < 6 || isempty(inputFile)
        inputFile = fullfile(rootDir, 'work', 'root_canal_1_5mm', ...
            '根管_已填充_1.5mm', '預測結果與評估指標_像素轉mm.xlsx');
    end
    if maxEpochs < 1 || maxEpochs ~= floor(maxEpochs)
        error('maxEpochs must be a positive integer.');
    end
    if hiddenUnits < 1 || hiddenUnits ~= floor(hiddenUnits)
        error('hiddenUnits must be a positive integer.');
    end
    if regularization < 0 || regularization > 1
        error('regularization must be between 0 and 1.');
    end

    rng(42, 'twister');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    data = readtable(inputFile, 'VariableNamingRule', 'preserve');
    ids = string(data{:, 1});
    X = double(data{:, 2});
    Y = double(data{:, 3});
    valid = isfinite(X) & isfinite(Y);
    ids = ids(valid);
    X = X(valid);
    Y = Y(valid);
    n = numel(Y);
    if n ~= 31
        error('Expected 31 valid records but found %d.', n);
    end

    K = 5;
    partition = cvpartition(n, 'KFold', K);
    YPredOOF = nan(n, 1);
    foldId = nan(n, 1);
    foldMetrics = repmat(struct('fold', 0, 'n', 0, 'mae_mm', 0, ...
        'rmse_mm', 0, 'r2', 0, 'mean_bias_mm', 0), K, 1);
    foldNets = cell(K, 1);
    foldScaling = repmat(struct('x_mean_px', 0, 'x_std_px', 0), K, 1);

    for fold = 1:K
        trainIndex = training(partition, fold);
        testIndex = test(partition, fold);

        % Fit standardization only on the training portion of this fold.
        xMean = mean(X(trainIndex));
        xStd = std(X(trainIndex), 0);
        if xStd == 0
            error('Fold %d has zero pixel-length variation.', fold);
        end
        XTrain = (X(trainIndex) - xMean) ./ xStd;
        XTest = (X(testIndex) - xMean) ./ xStd;

        net = fitnet(hiddenUnits, trainingFunction);
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        net.trainParam.epochs = maxEpochs;
        net.divideFcn = 'dividetrain';
        net.inputs{1}.processFcns = {};
        net.outputs{2}.processFcns = {};
        if ~strcmp(trainingFunction, 'trainbr')
            net.performParam.regularization = regularization;
        end
        net = train(net, XTrain', Y(trainIndex)');

        yPrediction = net(XTest')';
        YPredOOF(testIndex) = yPrediction;
        foldId(testIndex) = fold;
        foldMetrics(fold) = calculateMetrics(Y(testIndex), yPrediction, fold);
        foldNets{fold} = net;
        foldScaling(fold).x_mean_px = xMean;
        foldScaling(fold).x_std_px = xStd;
    end

    % Deployment model: re-train on all 31 records, not on the best fold only.
    fullMean = mean(X);
    fullStd = std(X, 0);
    XFull = (X - fullMean) ./ fullStd;
    finalNet = fitnet(hiddenUnits, trainingFunction);
    finalNet.trainParam.showWindow = false;
    finalNet.trainParam.showCommandLine = false;
    finalNet.trainParam.epochs = maxEpochs;
    finalNet.divideFcn = 'dividetrain';
    finalNet.inputs{1}.processFcns = {};
    finalNet.outputs{2}.processFcns = {};
    if ~strcmp(trainingFunction, 'trainbr')
        finalNet.performParam.regularization = regularization;
    end
    finalNet = train(finalNet, XFull', Y');
    YPredFull = finalNet(XFull')';

    overallOOF = calculateMetrics(Y, YPredOOF, 0);
    fullFit = calculateMetrics(Y, YPredFull, 0);
    residual = YPredOOF - Y;
    predictionRows = struct('case_id', cell(n,1), 'curve_length_px', cell(n,1), ...
        'reference_length_mm', cell(n,1), 'fold', cell(n,1), ...
        'oof_prediction_mm', cell(n,1), 'residual_mm', cell(n,1), ...
        'absolute_error_mm', cell(n,1));
    for i = 1:n
        predictionRows(i).case_id = char(ids(i));
        predictionRows(i).curve_length_px = X(i);
        predictionRows(i).reference_length_mm = Y(i);
        predictionRows(i).fold = foldId(i);
        predictionRows(i).oof_prediction_mm = YPredOOF(i);
        predictionRows(i).residual_mm = residual(i);
        predictionRows(i).absolute_error_mm = abs(residual(i));
    end

    configuration = struct( ...
        'model_type', 'MATLAB fitnet ANN', ...
        'hidden_units', hiddenUnits, ...
        'training_function', trainingFunction, ...
        'optimizer_description', trainingDescription(trainingFunction), ...
        'performance_regularization', regularization, ...
        'epochs_maximum', maxEpochs, ...
        'input_feature', 'traced_curve_length_px', ...
        'target', 'reference_working_length_mm', ...
        'fold_standardization', 'training-fold mean and standard deviation only', ...
        'cross_validation', struct('n_folds', 5, 'rng_seed', 42), ...
        'prediction_post_processing', 'none');
    report = struct('configuration', configuration, 'overall_oof', overallOOF, ...
        'by_fold', foldMetrics, 'full_data_fit', fullFit, 'oof_predictions', predictionRows);

    fid = fopen(fullfile(outputDir, 'ann_5fold_report.json'), 'w', 'n', 'UTF-8');
    fprintf(fid, '%s', jsonencode(report, PrettyPrint=true));
    fclose(fid);
    save(fullfile(outputDir, 'ann_fitnet_trainbr_models.mat'), 'finalNet', 'foldNets', ...
        'foldScaling', 'fullMean', 'fullStd', 'configuration', 'report');
    makePlots(Y, YPredOOF, outputDir, overallOOF);

    fprintf('OOF MAE: %.3f mm\n', overallOOF.mae_mm);
    fprintf('OOF RMSE: %.3f mm\n', overallOOF.rmse_mm);
    fprintf('OOF R2: %.3f\n', overallOOF.r2);
    fprintf('OOF mean bias: %.3f mm\n', overallOOF.mean_bias_mm);
end

function description = trainingDescription(trainingFunction)
    if strcmp(trainingFunction, 'trainbr')
        description = 'Bayesian regularization backpropagation';
    elseif strcmp(trainingFunction, 'trainlm')
        description = 'Levenberg-Marquardt backpropagation';
    else
        description = trainingFunction;
    end
end

function result = calculateMetrics(actual, predicted, fold)
    residual = predicted - actual;
    denominator = sum((actual - mean(actual)).^2);
    if denominator == 0
        r2 = NaN;
    else
        r2 = 1 - sum(residual.^2) / denominator;
    end
    result = struct('fold', fold, 'n', numel(actual), ...
        'mae_mm', mean(abs(residual)), ...
        'rmse_mm', sqrt(mean(residual.^2)), ...
        'r2', r2, ...
        'mean_bias_mm', mean(residual));
end

function makePlots(actual, predicted, outputDir, metric)
    figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 900 700]);
    hold on;
    scatter(actual, predicted, 52, [0 0.25 0.85], 'filled');
    minimum = floor(min([actual; predicted])) - 1;
    maximum = ceil(max([actual; predicted])) + 1;
    plot([minimum maximum], [minimum maximum], 'k-', 'LineWidth', 1.8);
    plot([minimum maximum], [minimum+1.5 maximum+1.5], 'g--', 'LineWidth', 1.2);
    plot([minimum maximum], [minimum-1.5 maximum-1.5], 'g--', 'LineWidth', 1.2);
    grid on;
    axis equal;
    xlim([minimum maximum]);
    ylim([minimum maximum]);
    xlabel('Reference length (mm)');
    ylabel('OOF prediction (mm)');
    title('ANN 5-fold Out-of-Fold prediction');
    label = sprintf('n = %d\nMAE = %.3f mm\nRMSE = %.3f mm\nR^2 = %.3f\nBias = %.3f mm', ...
        metric.n, metric.mae_mm, metric.rmse_mm, metric.r2, metric.mean_bias_mm);
    text(minimum + 0.05*(maximum-minimum), maximum - 0.08*(maximum-minimum), label, ...
        'VerticalAlignment', 'top', 'BackgroundColor', 'white', 'EdgeColor', 'black');
    legend({'OOF prediction', 'Identity', '\pm1.5 mm'}, 'Location', 'southeast');
    exportgraphics(gcf, fullfile(outputDir, 'ann_5fold_oof_plot.png'), 'Resolution', 180);
    close(gcf);
end
