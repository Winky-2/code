function pixelToMmPredictor5Fold()
% A_pixelToMmPredictor_5fold.m
% ============================================================
% 對應流程圖階段：像素轉毫米 (ANN) -> 臨床與安全性評估
% *** 5-fold 交叉驗證版本 ***
%
% *** 跟舊版 A_pixelToMmPredictor_fixed_v2.m 的差異 ***
% 1. 分組欄位從「資料夾來源」(train/test) 改成「Fold」(1~5)，
%    這欄由 01b_split_5folds.py 決定、02(v5) 帶出、03(v2) 合併保留，
%    這裡直接沿用，絕對不在 MATLAB 這一關重新隨機切。
%    這樣才能保證「YOLO 沒看過的牙齒」跟「ANN 沒看過的牙齒」是同一批。
% 2. 訓練 5 次：第 k 輪用 Fold==k 當 test、其餘 4 組當 train。
%    每顆牙都恰好會被「沒看過它的那個模型」預測一次，
%    這種預測稱為 out-of-fold (OOF) 預測。
%    20 顆牙 -> 20 筆 OOF 預測，樣本涵蓋率 100%，
%    比舊版只有 4 顆 test 可以評估穩定得多。
% 3. 標準化參數改成「每一輪只用該輪 train 算 mu/sigma」。
%    舊版是 zscore(X) 對全部資料算，等於 test 的分佈資訊
%    滲進了標準化參數，屬於輕微但真實的資料洩漏。
% 4. 所有指標統一改呼叫 computeMetrics.m（跟 B_ 繪圖腳本共用同一份算法）。
%    R² 從 Pearson r² 改成決定係數 1-SSE/SST，詳見 computeMetrics.m 的說明。
%
% 執行前請確認：
% - computeMetrics.m 跟這支放在同一個資料夾（或在 MATLAB path 上）
% - 03_merge_data_v2_5fold.py 已跑過，產生了 INPUT_FILE
% - 需要 Deep Learning Toolbox (fitnet / train)

    %% ---------------- 設定區 ----------------
    INPUT_FILE  = '根管填充物像素長度_已配對_5fold.xlsx';
    OUTPUT_FILE = '預測結果與評估指標_5fold_像素轉mm.xlsx';
    MODEL_FILE  = 'final_model_pixel_to_mm.mat';

    FOLD_COLUMN = 'Fold';      % 由 Python 端決定的分組，不在這裡重切
    N_FOLDS     = 5;

    % ---- 臨床設定（跟舊版一致，正式值待醫師確認）----
    % 對應流程圖「轉換實際長度 (長度推估 -0.5mm)」
    OFFSET_MM = 0.5;
    % 對應流程圖「臨床與安全性評估」的理想比率容忍值
    IDEAL_TOLERANCE_MM = 1.0;

    % ---- 模型設定 ----
    % 每一輪只有約 16 筆訓練資料，隱藏層放太大很容易過擬合。
    % trainbr (Bayesian regularization) 本身有正則化，比較耐小樣本，
    % 但 HIDDEN_SIZE 仍建議保守，5 以下。
    HIDDEN_SIZE   = 5;
    TRAIN_FCN     = 'trainbr';
    MAX_EPOCHS    = 200;
    RANDOM_SEED   = 42;

    % 每一折要不要重覆訓練幾次再把預測平均起來。
    % 小樣本 + 隨機初始權重，單次訓練的結果抖動會很大；
    % 設成 3~5 可以讓指標穩定不少，而且完全不碰 test，沒有洩漏問題。
    % 預設 1 是為了跟舊版行為一致，想要穩定的數字就調大。
    REPEATS_PER_FOLD = 1;

    % 跑完 5-fold 後，要不要再用「全部資料」訓練一個最終模型存起來，
    % 給之後實際部署/預測新病例用。
    % ⚠️ 這個最終模型的 in-sample 表現不可以拿來當 model 能力，
    %    要報告的數字一律用下面的 OOF 指標。
    TRAIN_FINAL_MODEL = true;

    rng(RANDOM_SEED);

    if exist('computeMetrics', 'file') ~= 2
        error(['找不到 computeMetrics.m，請把它跟這支腳本放在同一個資料夾。' ...
               '這支腳本與 B_plotMyResults 都靠它來統一指標算法。']);
    end

    %% --- 1. 讀取 03 合併好的 Excel ---
    if exist(INPUT_FILE, 'file') ~= 2
        error('找不到 %s，請先執行 03_merge_data_v2_5fold.py。', INPUT_FILE);
    end
    opts = detectImportOptions(INPUT_FILE);
    opts.VariableNamingRule = 'preserve';   % 保留中文欄位名，不然會被改成 Var1/拼音
    data = readtable(INPUT_FILE, opts);

    %% --- 2. 欄位對接與檢查 ---
    requiredCols = {'圖片檔名', '填充物長度(mm)', '像素長度', FOLD_COLUMN};
    for i = 1:numel(requiredCols)
        if ~ismember(requiredCols{i}, data.Properties.VariableNames)
            error(['Excel 裡找不到 "%s" 欄位。目前欄位為：%s\n' ...
                   '請確認讀到的是 03_merge_data_v2_5fold.py 的輸出（5-fold 版），' ...
                   '而不是舊版單次切分的配對檔。'], ...
                   requiredCols{i}, strjoin(data.Properties.VariableNames, ', '));
        end
    end

    ids  = data.('圖片檔名');                % 只當 ID 用，不參與訓練
    Y    = data.('填充物長度(mm)');          % 目標：實際長度 (mm)
    X    = data.('像素長度');                % 特徵：YOLO 算出的像素長度
    fold = data.(FOLD_COLUMN);

    if ~isnumeric(Y) || ~isnumeric(X)
        error('「填充物長度(mm)」或「像素長度」欄位含有非數值資料，請檢查 Excel。');
    end
    if ~isnumeric(fold)
        fold = str2double(string(fold));
    end

    n = height(data);
    fprintf('成功載入資料，總計 %d 顆牙齒。\n', n);

    % 排除無效列：任何一個關鍵欄位是 NaN，或 Fold 不在 1~N_FOLDS 之內
    isValid = ~isnan(Y) & ~isnan(X) & ~isnan(fold) & ismember(fold, 1:N_FOLDS);
    if any(~isValid)
        fprintf('⚠️ 有 %d 顆牙的資料不完整或 Fold 值異常，將被排除在訓練與評估之外：\n', sum(~isValid));
        badIdx = find(~isValid);
        for i = 1:numel(badIdx)
            k = badIdx(i);
            fprintf('   - %s (像素=%g, 實際長度=%g, Fold=%g)\n', ...
                string(ids(k)), X(k), Y(k), fold(k));
        end
    end

    %% --- 3. 檢查 Fold 分佈 ---
    fprintf('\n=== Fold 分佈（沿用 Python 端分組，未重新隨機切分）===\n');
    for k = 1:N_FOLDS
        nk = sum(isValid & fold == k);
        fprintf('   fold%d: %d 顆\n', k, nk);
        if nk == 0
            error('fold%d 一顆牙都沒有，無法做 5-fold。請檢查 Excel 的 Fold 欄位。', k);
        end
    end
    if sum(isValid) < 10
        fprintf('⚠️ 有效樣本只有 %d 顆，各項指標（尤其理想比率/過長率）僅供參考，\n', sum(isValid));
        fprintf('   單一筆的對錯就會讓百分比大幅跳動，解讀時請留意。\n');
    end

    %% --- 4. 5-fold 迴圈：每輪訓練一個模型，只預測該輪的 held-out ---
    Y_oof = nan(n, 1);   % out-of-fold 預測：每顆牙都由「沒看過它」的模型填入

    fprintf('\n=== 開始 5-fold 訓練 ===\n');
    for k = 1:N_FOLDS
        testIdx  = isValid & (fold == k);
        trainIdx = isValid & (fold ~= k);

        Xtr = X(trainIdx);  Ytr = Y(trainIdx);
        Xte = X(testIdx);

        % 標準化參數只從 train 算，再套用到 test。
        % 這是 5-fold 的基本要求：test 的任何統計量都不可以參與訓練前處理。
        mu = mean(Xtr);
        sigma = std(Xtr);
        if sigma == 0
            % 所有訓練樣本像素長度都一樣（幾乎不可能，但防呆）
            sigma = 1;
        end
        Xtr_n = (Xtr - mu) / sigma;
        Xte_n = (Xte - mu) / sigma;

        % 重覆訓練 REPEATS_PER_FOLD 次，取預測平均，壓抑隨機初始權重造成的抖動
        predAccum = zeros(sum(testIdx), 1);
        for r = 1:REPEATS_PER_FOLD
            net = fitnet(HIDDEN_SIZE, TRAIN_FCN);
            net.trainParam.showWindow = false;
            net.trainParam.epochs     = MAX_EPOCHS;
            % dividetrain：丟進 train() 的資料全部當訓練集。
            % 這裡本來就只餵 train 折的資料進去，test 折從頭到尾沒進過網路，
            % 比舊版用 divideind 指定 testInd 更不容易出錯。
            net.divideFcn = 'dividetrain';

            net = train(net, Xtr_n', Ytr');
            predAccum = predAccum + net(Xte_n')';
        end
        Y_oof(testIdx) = predAccum / REPEATS_PER_FOLD;

        fprintf('   fold%d 完成：train %d 顆 / held-out %d 顆\n', ...
            k, sum(trainIdx), sum(testIdx));
    end

    if any(isnan(Y_oof(isValid)))
        error('有 %d 顆有效牙齒沒有拿到 OOF 預測，請檢查 Fold 欄位是否覆蓋完整。', ...
            sum(isnan(Y_oof(isValid))));
    end

    %% --- 5. 指標計算（全部走 computeMetrics，跟繪圖腳本共用同一份算法）---
    Yv    = Y(isValid);
    Ypv   = Y_oof(isValid);
    foldv = fold(isValid);

    % (A) 彙總 OOF：把 5 折的預測合起來一次算，這是要拿來報告的主要指標
    mPooled = computeMetrics(Yv, Ypv, OFFSET_MM, IDEAL_TOLERANCE_MM);

    % (B) 各 fold 單獨算一次，用來看「不同分組之間穩不穩定」
    foldMetrics = cell(N_FOLDS, 1);
    for k = 1:N_FOLDS
        sel = foldv == k;
        foldMetrics{k} = computeMetrics(Yv(sel), Ypv(sel), OFFSET_MM, IDEAL_TOLERANCE_MM);
    end

    fprintf('\n=== 5-fold OOF 彙總指標（主要指標）===\n');
    fprintf('n = %d | MAE: %.3f mm | RMSE: %.3f mm | R²: %.3f | Pearson r²: %.3f | Bias: %.3f mm\n', ...
        mPooled.n, mPooled.MAE, mPooled.RMSE, mPooled.R2, mPooled.PearsonR2, mPooled.Bias);
    if mPooled.R2 < 0
        fprintf('⚠️ R² 是負的，代表模型的預測比「一律猜平均長度」還差，請回頭檢查像素長度與實際長度的關係。\n');
    end

    fprintf('\n=== 各 fold 指標（看穩定度）===\n');
    fprintf('  Fold    n     MAE    RMSE      R²    Bias\n');
    for k = 1:N_FOLDS
        fm = foldMetrics{k};
        fprintf('  %4d %4d  %6.3f  %6.3f  %6.3f  %6.3f\n', ...
            k, fm.n, fm.MAE, fm.RMSE, fm.R2, fm.Bias);
    end
    fprintf('（各 fold 只有 3~4 顆牙，單折的 R² 會非常不穩定甚至為負，屬正常現象，\n');
    fprintf('  請以上面的「彙總 OOF」為準，各折數字只用來看有沒有某一折特別離譜。）\n');
    fprintf('\n⚠️ 重要：R² 不可以用「各 fold 平均」來報告，只能報彙總 OOF 的值。\n');
    fprintf('   原因：R² = 1-SSE/SST，分母 SST 是「該折自己的長度變異數」。\n');
    fprintf('   單折只有 3~4 顆牙，長度範圍很窄、SST 很小，同樣的誤差算出來的 R² 就會偏低，\n');
    fprintf('   所以各折 R² 的平均會系統性地低於彙總 R²，兩者不是同一個量，不能互相比較。\n');
    fprintf('   MAE/RMSE/Bias 沒有這個問題（等分組下折平均≈彙總），R²/Pearson r² 有。\n');

    %% --- 6. 組結果表格（逐筆的中間結果直接用 computeMetrics 回傳的，不重算）---
    Resid        = nan(n,1);  AbsErr       = nan(n,1);
    PredOffset   = nan(n,1);  ResidOffset  = nan(n,1);
    AbsErrOffset = nan(n,1);  IsOverest    = false(n,1);

    Resid(isValid)        = mPooled.Resid;
    AbsErr(isValid)       = mPooled.AbsErr;
    PredOffset(isValid)   = mPooled.PredOffset;
    ResidOffset(isValid)  = mPooled.ResidOffset;
    AbsErrOffset(isValid) = mPooled.AbsErrOffset;
    IsOverest(isValid)    = mPooled.IsOverest;

    ResultsTable = table(ids, fold, X, Y, Y_oof, Resid, AbsErr, ...
        PredOffset, ResidOffset, AbsErrOffset, IsOverest, ...
        'VariableNames', {'檔名', 'Fold', '像素長度_px', '實際長度_mm', ...
        'OOF預測_mm', '誤差Bias_mm', '絕對誤差_mm', ...
        '校正後預測_mm', '校正後誤差Bias_mm', '校正後絕對誤差_mm', '是否過長'});
    ResultsTable = sortrows(ResultsTable, {'Fold', '檔名'});

    %% --- 7. 指標表格 ---
    % 用固定不變的英文 Key 當查詢鍵，中文說明另放一欄。
    % 舊版是靠中文字串 contains 去找列（例如 'A_Test set表現'），
    % 只要有人改個字，B_ 繪圖腳本就報錯。改成 Key 之後不會再有這個問題。
    % 註：這裡的區域變數一律用 ASCII 命名。
    % 中文可以當 table 的欄位名（字串），但拿來當程式變數名在部分 MATLAB
    % 版本會直接語法錯誤，兩者不能混為一談。
    col = @(fieldName) cellfun(@(mm) mm.(fieldName), foldMetrics);

    Key = {'A_OOF_POOLED'; 'B_FOLD_MEAN'; 'C_FOLD_STD'};
    Desc = {
        sprintf('5-fold 彙總 out-of-fold 預測 (n=%d，每顆牙都由沒看過它的模型預測) 👈 要報告的主要指標', mPooled.n);
        '各 fold 指標的平均值（看整體水準）⚠️ 此列的 R_Square/Pearson_r2 不可拿來報告，理由見下方說明';
        '各 fold 指標的標準差（看折與折之間穩不穩定，數字大代表對分組很敏感）'
    };

    n_col        = [mPooled.n;           N_FOLDS;                       NaN];
    MAE_col      = [mPooled.MAE;         meanNoNan(col('MAE'));         stdNoNan(col('MAE'))];
    RMSE_col     = [mPooled.RMSE;        meanNoNan(col('RMSE'));        stdNoNan(col('RMSE'))];
    R2_col       = [mPooled.R2;          meanNoNan(col('R2'));          stdNoNan(col('R2'))];
    PearsonR2_col= [mPooled.PearsonR2;   meanNoNan(col('PearsonR2'));   stdNoNan(col('PearsonR2'))];
    Bias_col     = [mPooled.Bias;        meanNoNan(col('Bias'));        stdNoNan(col('Bias'))];
    MAEoff_col   = [mPooled.MAE_offset;  meanNoNan(col('MAE_offset'));  stdNoNan(col('MAE_offset'))];
    Biasoff_col  = [mPooled.Bias_offset; meanNoNan(col('Bias_offset')); stdNoNan(col('Bias_offset'))];
    Ideal_col    = [mPooled.IdealRate;   meanNoNan(col('IdealRate'));   stdNoNan(col('IdealRate'))];
    Overest_col  = [mPooled.OverestRate; meanNoNan(col('OverestRate')); stdNoNan(col('OverestRate'))];

    MetricsTable = table(Key, Desc, n_col, ...
        round(MAE_col,3), round(RMSE_col,3), round(R2_col,3), round(PearsonR2_col,3), ...
        round(Bias_col,3), round(MAEoff_col,3), round(Biasoff_col,3), ...
        round(Ideal_col,1), round(Overest_col,1), ...
        'VariableNames', {'Key', '說明', 'n', 'MAE_mm', 'RMSE_mm', 'R_Square', 'Pearson_r2', ...
        'Mean_Bias_mm', 'MAE_校正後_mm', 'Bias_校正後_mm', '理想比率_pct', '過長率_pct'});

    % 各 fold 明細
    FoldTable = table((1:N_FOLDS)', ...
        cellfun(@(m)m.n, foldMetrics), ...
        round(cellfun(@(m)m.MAE, foldMetrics), 3), ...
        round(cellfun(@(m)m.RMSE, foldMetrics), 3), ...
        round(cellfun(@(m)m.R2, foldMetrics), 3), ...
        round(cellfun(@(m)m.PearsonR2, foldMetrics), 3), ...
        round(cellfun(@(m)m.Bias, foldMetrics), 3), ...
        round(cellfun(@(m)m.MAE_offset, foldMetrics), 3), ...
        round(cellfun(@(m)m.Bias_offset, foldMetrics), 3), ...
        round(cellfun(@(m)m.IdealRate, foldMetrics), 1), ...
        round(cellfun(@(m)m.OverestRate, foldMetrics), 1), ...
        'VariableNames', {'Fold', 'n', 'MAE_mm', 'RMSE_mm', 'R_Square', 'Pearson_r2', ...
        'Mean_Bias_mm', 'MAE_校正後_mm', 'Bias_校正後_mm', '理想比率_pct', '過長率_pct'});

    % 設定表：B_ 繪圖腳本會讀這張表拿 offset / 容忍值，
    % 這樣兩支腳本的臨床參數也只有一份，不用兩邊各打一次
    SettingsTable = table( ...
        {'OFFSET_MM'; 'IDEAL_TOLERANCE_MM'; 'N_FOLDS'; 'HIDDEN_SIZE'; 'MAX_EPOCHS'; 'REPEATS_PER_FOLD'; 'RANDOM_SEED'}, ...
        [OFFSET_MM; IDEAL_TOLERANCE_MM; N_FOLDS; HIDDEN_SIZE; MAX_EPOCHS; REPEATS_PER_FOLD; RANDOM_SEED], ...
        'VariableNames', {'參數', '值'});

    %% --- 8. 匯出 Excel ---
    % 先刪掉舊檔：writetable 只覆蓋它寫到的儲存格，
    % 如果新結果列數比舊的少，舊檔尾巴會殘留上一次的資料，非常難察覺。
    if exist(OUTPUT_FILE, 'file') == 2
        delete(OUTPUT_FILE);
    end
    writetable(ResultsTable,  OUTPUT_FILE, 'Sheet', '所有牙齒預測結果');
    writetable(MetricsTable,  OUTPUT_FILE, 'Sheet', '模型評估指標');
    writetable(FoldTable,     OUTPUT_FILE, 'Sheet', '各fold指標');
    writetable(SettingsTable, OUTPUT_FILE, 'Sheet', '設定');

    fprintf('\n✅ 結果已匯出至：「%s」\n', OUTPUT_FILE);
    disp('=== 模型評估指標 ===');
    disp(MetricsTable(:, {'Key', 'n', 'MAE_mm', 'RMSE_mm', 'R_Square', 'Pearson_r2', 'Mean_Bias_mm'}));

    fprintf('\n=== 臨床與安全性評估（對應流程圖最後一步，基於彙總 OOF）===\n');
    fprintf('offset = -%.2f mm，理想誤差容忍 = ±%.2f mm（皆為預設值，需醫師確認後調整）\n', ...
        OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('MAE(校正後): %.3f mm | Bias(校正後): %.3f mm | 理想比率: %.1f%% | 過長率: %.1f%%\n', ...
        mPooled.MAE_offset, mPooled.Bias_offset, mPooled.IdealRate, mPooled.OverestRate);
    fprintf('⚠️ 過長率是臨床風險較高的指標（預測工作長度仍比實際長 -> 有超出根尖的風險）。\n');

    %% --- 9. （選用）用全部資料訓練最終模型存檔 ---
    if TRAIN_FINAL_MODEL
        Xall = X(isValid);  Yall = Y(isValid);
        mu_final = mean(Xall);
        sigma_final = std(Xall);
        if sigma_final == 0, sigma_final = 1; end

        net = fitnet(HIDDEN_SIZE, TRAIN_FCN);
        net.trainParam.showWindow = false;
        net.trainParam.epochs     = MAX_EPOCHS;
        net.divideFcn = 'dividetrain';
        net = train(net, ((Xall - mu_final)/sigma_final)', Yall');

        save(MODEL_FILE, 'net', 'mu_final', 'sigma_final', 'OFFSET_MM', 'IDEAL_TOLERANCE_MM');
        fprintf('\n📦 已用全部 %d 顆牙訓練最終模型並存成「%s」（供之後預測新病例用）。\n', ...
            numel(Yall), MODEL_FILE);
        fprintf('   ⚠️ 這個模型看過全部資料，它的 in-sample 誤差不能當成 model 能力，\n');
        fprintf('      要報告的數字請一律用上面的 5-fold OOF 指標。\n');
    end

    fprintf('\n👉 接下來請執行 B_plotMyResults_5fold.m 產生圖表\n');

end


% ============================================================
% 輔助函式：自己處理 NaN，不依賴 mean/std 的 'omitnan' 參數
% （'omitnan' 在較舊的 MATLAB 版本語法不一致，寫死比較保險）
% ============================================================

function v = meanNoNan(x)
    x = x(~isnan(x));
    if isempty(x), v = NaN; else, v = mean(x); end
end

function v = stdNoNan(x)
    x = x(~isnan(x));
    if numel(x) < 2, v = NaN; else, v = std(x); end
end
