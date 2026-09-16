function C1v2_pixelToMmPredictor()
%% C1v2_pixelToMmPredictor.m
% ============================================================
% 對應流程圖 C 區塊：「轉換實際長度(-0.5mm)」+「臨床與安全性評估
% (MAE/理想比率/過長率)」。
%
% *** 跟舊版 C1_pixelToMmPredictor.m 的差異 ***
% 迴歸核心(fitnet + trainbr + offset校正 + 理想比率/過長率)完全沿用，
% 改的是「怎麼切資料、怎麼評估」這一層：
%
%   舊版：單次 train/test 切分，沿用 Excel 的「資料夾來源」欄
%   本版：k 摺交叉驗證，54 筆全部輪流當過 test
%
% *** 為什麼非改不可 ***
% 醫師記錄的 mm 值只涵蓋 test 資料夾那批牙，train 資料夾沒有 Y，
% 對 ANN 完全無用。所以 ANN 能用的資料母體就是這 54 顆，一顆都不會
% 再多，train 和 test 只能從裡面切。
%
% 單次 80/20 的話 test 只有 ~14 筆，而理想比率/過長率是百分比指標，
% 一筆對錯就跳 7 個百分點，換個亂數種子數字就變樣——這種數字不適合
% 寫進論文。k 摺讓每一筆都當過一次 test，最後報的是 54 筆的整體表現，
% 外加「摺間標準差」當穩定性指標。
%
% 交叉驗證不是分類專用的，它只是一種估計「模型在沒看過的資料上表現
% 如何」的重採樣方法，迴歸一樣適用，差別只在不做 class 分層。
%
% *** 三個必須守住的規矩 ***
% 1. 標準化參數只能用 train 摺算
%    舊版 zscore(X) 是拿全部資料算 mean/std，test 的資訊會滲進標準化
%    參數裡。單次切分時影響小，k 摺之下每一摺都會沾到，必須修掉。
%
% 2. 摺的分派沿用 Excel 的「折數fold」欄，不在這裡重新亂數
%    三種量測法要跑三次做對照，如果每次的摺不一樣，比出來的差異就
%    分不清是量法造成的還是切分造成的。B4 產生的三份檔案共用同一組
%    fold，這裡照用就好。
%
% 3. 重複多次取平均
%    fitnet 的初始權重是隨機的，54 筆這種量級，同一份資料跑兩次結果
%    可能差一截。重複 N_REPEATS 次、報平均與標準差，才不會拿到一個
%    運氣好的數字。
%
% *** 附帶一個線性迴歸對照組 ***
% 輸入只有一個特徵、54 個樣本，fitnet(5) 有 16 個參數(1→5 的權重 5 個
% + 隱藏層 bias 5 個 + 5→1 的權重 5 個 + 輸出 bias 1 個)，大約 3.4 筆
% 養一個參數，偏緊。而且 letterbox scale 修正之後，px→mm 理論上應該
% 接近線性(斜率就是每像素幾毫米)。
%
% 所以同一組摺也跑一次最小平方線性迴歸。如果 ANN 贏不了它，就該用
% 線性——樣本這麼少，簡單模型比較不會過擬合，也比較好辯護。反過來
% 如果 ANN 明顯較好，那本身就是一個可以寫的發現。
%
% *** 輸出格式刻意跟舊版一致 ***
% 工作表名稱、欄位名稱、MetricNames 裡 C2 會去比對的關鍵字都沒變，
% 所以 C2 不用大改也能讀。差別是「Train_or_Test」欄現在全部是 test
% ——因為每一筆都是 out-of-fold 預測，本來就都沒被自己的模型看過。
% ============================================================

    %% ---------------- 輸入設定區 ----------------
    % B4_build_C1_input.py 會產生三份，一次跑一份，跑完換 METHOD 再跑。
    % 三份的列數、列順序、fold 完全相同，唯一差異是「像素長度」的算法。
    METHOD = '原始預測';        % 'mask幾何' | '原始預測' | '長軸投影'
    filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];

    FOLD_COLUMN  = '折數fold';
    SPLIT_COLUMN = '資料夾來源';   % 只在 USE_KFOLD=false 時才用

    %% ---------------- 評估設定區 ----------------
    USE_KFOLD  = true;    % false 則退回舊版單次切分行為(保留可回溯)
    N_REPEATS  = 10;      % 重複次數，吸收 ANN 隨機初始化的波動
    BASE_SEED  = 42;

    HIDDEN_SIZE = 5;
    TRAIN_FCN   = 'trainbr';
    MAX_EPOCHS  = 200;

    RUN_LINEAR_BASELINE = true;

    %% ---------------- 臨床設定區 ----------------
    OFFSET_MM          = 0.5;   % 臨床安全 offset，正式值待醫師確認
    IDEAL_TOLERANCE_MM = 1.0;   % 理想比率的容忍範圍，待醫師確認

    %% --- 1. 讀取資料 ---
    if ~isfile(filename)
        error(['找不到 %s\n' ...
               '   這份檔案由 B4_build_C1_input.py 產生，請先跑那支腳本。'], filename);
    end
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    data = readtable(filename, opts);

    ids = data.('圖片檔名');
    Y   = data.('填充物長度(mm)');
    X   = data.('像素長度');

    if ~isnumeric(Y) || ~isnumeric(X)
        error('實際長度或像素長度欄位含非數值資料，請檢查 Excel。');
    end
    valid = ~isnan(X) & ~isnan(Y);
    if any(~valid)
        fprintf('⚠️ 有 %d 筆含缺值，已剔除(B4 理論上不該讓缺值流出來，請回頭檢查)。\n', ...
                sum(~valid));
        ids = ids(valid); Y = Y(valid); X = X(valid);
        data = data(valid, :);
    end

    n = numel(Y);
    fprintf('=== C1v2：%s ===\n', METHOD);
    fprintf('載入 %d 筆樣本。\n', n);

    %% --- 2. 決定每一筆屬於哪一摺 ---
    if USE_KFOLD
        if ismember(FOLD_COLUMN, data.Properties.VariableNames)
            foldId = double(data.(FOLD_COLUMN));
            fprintf('沿用 Excel 的「%s」欄分摺(三種方法共用同一組切分)。\n', FOLD_COLUMN);
        else
            % 沒有 fold 欄時就地產生：依 Y 排序後輪流分派，
            % 確保每一摺都涵蓋長短牙的完整範圍，不會有某摺全是短牙。
            K = 5;
            [~, ord] = sort(Y);
            foldId = zeros(n, 1);
            foldId(ord) = mod(0:n-1, K) + 1;
            fprintf('⚠️ Excel 沒有「%s」欄，改用依實際長度分層的 %d 摺。\n', FOLD_COLUMN, K);
            fprintf('   注意：這樣三種方法的切分不保證一致，對照結果會混入切分差異。\n');
        end
        K = max(foldId);
        fprintf('共 %d 摺，每摺樣本數：', K);
        fprintf('%d ', accumarray(foldId, 1)); fprintf('\n');
    else
        if ~ismember(SPLIT_COLUMN, data.Properties.VariableNames)
            error('USE_KFOLD=false 需要「%s」欄。', SPLIT_COLUMN);
        end
        isTest = string(data.(SPLIT_COLUMN)) == "test";
        foldId = ones(n, 1);        % 只有一摺
        foldId(~isTest) = 0;        % 0 = 永遠當 train
        K = 1;
        fprintf('單次切分模式：train %d 筆 / test %d 筆\n', sum(~isTest), sum(isTest));
        if sum(isTest) < 10
            fprintf('⚠️ test 只有 %d 筆，理想比率/過長率一筆對錯就大幅跳動。\n', sum(isTest));
        end
    end

    %% --- 3. 交叉驗證主迴圈 ---
    % annPred(i, r) = 第 r 次重複時，第 i 筆的 out-of-fold 預測
    annPred = nan(n, N_REPEATS);
    linPred = nan(n, N_REPEATS);
    foldMAE = nan(K, N_REPEATS);

    fprintf('\n訓練中(%d 摺 × %d 次重複)...\n', K, N_REPEATS);
    for r = 1:N_REPEATS
        rng(BASE_SEED + r);   % 固定種子，重跑可重現
        for k = 1:K
            teIdx = (foldId == k);
            trIdx = ~teIdx & (foldId > 0);
            if ~any(teIdx) || ~any(trIdx), continue; end

            Xtr = X(trIdx); Ytr = Y(trIdx);
            Xte = X(teIdx);

            % 標準化參數只用 train 摺算，避免 test 資訊滲入
            mu = mean(Xtr);
            sg = std(Xtr);
            if sg == 0, sg = 1; end
            XtrN = (Xtr - mu) / sg;
            XteN = (Xte - mu) / sg;

            net = fitnet(HIDDEN_SIZE, TRAIN_FCN);
            net.trainParam.showWindow = false;
            net.trainParam.epochs     = MAX_EPOCHS;
            % 這一摺的資料全部拿去訓練；held-out 的部分手動預測，
            % 不交給 divideFcn 處理，才能完全掌控哪些資料被看過。
            net.divideFcn = 'dividetrain';

            net = train(net, XtrN', Ytr');
            annPred(teIdx, r) = net(XteN')';

            if RUN_LINEAR_BASELINE
                p = polyfit(Xtr, Ytr, 1);
                linPred(teIdx, r) = polyval(p, Xte);
            end

            foldMAE(k, r) = mean(abs(annPred(teIdx, r) - Y(teIdx)));
        end
        fprintf('  重複 %d/%d 完成\n', r, N_REPEATS);
    end

    %% --- 4. 逐次重複算指標，再取平均與標準差 ---
    % 注意：不是先把預測平均起來再算指標。把 N 次預測平均等於做了一個
    % 集成模型，會比單一模型樂觀；這裡要估的是「單一模型的表現」。
    M = struct('mae', nan(N_REPEATS,1), 'rmse', nan(N_REPEATS,1), ...
               'r2', nan(N_REPEATS,1), 'bias', nan(N_REPEATS,1), ...
               'offMae', nan(N_REPEATS,1), 'offBias', nan(N_REPEATS,1), ...
               'ideal', nan(N_REPEATS,1), 'over', nan(N_REPEATS,1));
    for r = 1:N_REPEATS
        M = accumMetrics(M, r, annPred(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
    end

    L = struct('mae', nan(N_REPEATS,1), 'rmse', nan(N_REPEATS,1), ...
               'r2', nan(N_REPEATS,1), 'bias', nan(N_REPEATS,1), ...
               'offMae', nan(N_REPEATS,1), 'offBias', nan(N_REPEATS,1), ...
               'ideal', nan(N_REPEATS,1), 'over', nan(N_REPEATS,1));
    if RUN_LINEAR_BASELINE
        for r = 1:N_REPEATS
            L = accumMetrics(L, r, linPred(:,r), Y, OFFSET_MM, IDEAL_TOLERANCE_MM);
        end
    end

    % 摺間標準差：跨摺的表現差異，反映結果穩不穩
    foldSpread = mean(std(foldMAE, 0, 1, 'omitnan'), 'omitnan');

    %% --- 5. 挑一次「代表性」的重複來輸出逐筆結果與畫圖 ---
    % 取 MAE 最接近中位數的那一次，讓 C2 畫出來的是一個真實存在的
    % 單一模型，而不是平均後的虛擬模型。
    [~, repIdx] = min(abs(M.mae - median(M.mae)));
    Y_pred = annPred(:, repIdx);
    fprintf('\n逐筆結果採用第 %d 次重複(MAE=%.3f，最接近中位數)。\n', repIdx, M.mae(repIdx));

    Y_pred_offset  = Y_pred - OFFSET_MM;
    residuals      = Y_pred - Y;
    abs_error      = abs(residuals);
    off_residuals  = Y_pred_offset - Y;
    off_abs_error  = abs(off_residuals);
    is_overest     = off_residuals > 0;

    %% --- 6. 組輸出表(欄位名維持跟舊版一致，C2 才讀得到) ---
    % k 摺之下每一筆都是 out-of-fold，都沒被自己的模型看過，所以全標 test
    split_label = repmat("test", n, 1);

    ResultsTable = table(ids, split_label, X, Y, ...
        Y_pred, residuals, abs_error, ...
        Y_pred_offset, off_residuals, off_abs_error, is_overest, foldId, ...
        'VariableNames', {'檔名', 'Train_or_Test', '像素長度_px', '實際長度_mm', ...
        '模型預測_mm', '誤差Bias_mm', '絕對誤差_mm', ...
        '校正後預測_mm', '校正後誤差Bias_mm', '校正後絕對誤差_mm', '是否過長', '摺別'});

    % MetricNames 裡 'A_Test set表現' 與 'C_臨床安全性評估_Test set校正後'
    % 這兩段文字是 C2 用 contains 抓列用的關鍵字，不要改，也不要讓其他列
    % 出現同樣字串(否則 C2 會抓到多列而報錯)。
    MetricNames = {
        sprintf('A_Test set表現 (out-of-fold, n=%d, ANN, %d次重複平均)', n, N_REPEATS);
        sprintf('B_重複間標準差 (同一份資料重跑%d次的波動)', N_REPEATS);
        '--- 分界線 ---';
        sprintf('C_臨床安全性評估_Test set校正後 (offset=-%.2fmm)', OFFSET_MM);
        sprintf('D_線性迴歸對照組_未校正 (同一組摺, n=%d)', n);
        sprintf('E_線性迴歸對照組_校正後 (offset=-%.2fmm)', OFFSET_MM)
    };

    nanpad = NaN;
    MAE_val = round([mean(M.mae); std(M.mae); nanpad; mean(M.offMae); ...
                     meanOr(L.mae, RUN_LINEAR_BASELINE); ...
                     meanOr(L.offMae, RUN_LINEAR_BASELINE)], 3);
    RMSE_val = round([mean(M.rmse); std(M.rmse); nanpad; nanpad; ...
                      meanOr(L.rmse, RUN_LINEAR_BASELINE); nanpad], 3);
    R2_val = round([mean(M.r2); std(M.r2); nanpad; nanpad; ...
                    meanOr(L.r2, RUN_LINEAR_BASELINE); nanpad], 3);
    Bias_val = round([mean(M.bias); std(M.bias); nanpad; mean(M.offBias); ...
                      meanOr(L.bias, RUN_LINEAR_BASELINE); ...
                      meanOr(L.offBias, RUN_LINEAR_BASELINE)], 3);
    IdealRate_pct = round([nanpad; std(M.ideal); nanpad; mean(M.ideal); ...
                           nanpad; meanOr(L.ideal, RUN_LINEAR_BASELINE)], 1);
    OverestRate_pct = round([nanpad; std(M.over); nanpad; mean(M.over); ...
                             nanpad; meanOr(L.over, RUN_LINEAR_BASELINE)], 1);

    MetricsTable = table(MetricNames, MAE_val, RMSE_val, R2_val, Bias_val, ...
        IdealRate_pct, OverestRate_pct, ...
        'VariableNames', {'評估範圍_與_嚴格程度', 'MAE_mm', 'RMSE_mm', 'R_Square', ...
        'Mean_Bias_mm', '理想比率_pct', '過長率_pct'});

    %% --- 7. 匯出 ---
    output_filename = sprintf('預測結果與評估指標_kfold_%s.xlsx', METHOD);
    if isfile(output_filename), delete(output_filename); end
    writetable(ResultsTable, output_filename, 'Sheet', '所有牙齒預測結果');
    writetable(MetricsTable, output_filename, 'Sheet', '模型評估指標');

    fprintf('\n✅ 已匯出：%s\n\n', output_filename);
    disp('=== 評估指標 ===');
    disp(MetricsTable);

    fprintf('\n=== 臨床與安全性評估 ===\n');
    fprintf('量測方法：%s | offset = -%.2f mm | 理想容忍 = ±%.2f mm\n', ...
        METHOD, OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('[Out-of-fold, n=%d, %d次重複]\n', n, N_REPEATS);
    fprintf('  校正後 MAE  : %.3f ± %.3f mm\n', mean(M.offMae), std(M.offMae));
    fprintf('  理想比率    : %.1f ± %.1f %%\n', mean(M.ideal), std(M.ideal));
    fprintf('  過長率      : %.1f ± %.1f %%  ← 臨床風險較高的指標\n', ...
        mean(M.over), std(M.over));
    fprintf('  摺間 MAE 標準差 : %.3f mm(跨摺的表現差異)\n', foldSpread);

    if RUN_LINEAR_BASELINE
        fprintf('\n[線性迴歸對照組] MAE %.3f mm vs ANN %.3f mm  →  ', ...
            mean(L.mae), mean(M.mae));
        if mean(L.mae) <= mean(M.mae)
            fprintf('線性不輸 ANN\n');
            fprintf('  這種樣本量下，簡單模型贏不是壞消息。除非 ANN 有明顯優勢，\n');
            fprintf('  否則用線性迴歸比較不會過擬合，也比較好向口委解釋。\n');
        else
            fprintf('ANN 勝出 %.3f mm\n', mean(L.mae) - mean(M.mae));
            fprintf('  代表 px→mm 存在非線性成分，ANN 有存在價值，這點值得寫進論文。\n');
        end
    end

    fprintf('\n⚠️ 樣本母體就是這 %d 顆牙，百分比指標請連同 n 與標準差一起報告。\n', n);
    fprintf('👉 接下來執行 C2_plotMyResults 產生圖表(記得把 METHOD 設成一樣)。\n');
end


%% ============================================================
%  子函式
%  ============================================================

function S = accumMetrics(S, r, Yp, Y, offset, tol)
% 把第 r 次重複的各項指標塞進結構。
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
    S.over(r)    = mean(offRes > 0) * 100;   % 高估工作長度 = 有超出根尖的風險
end


function v = meanOr(x, flag)
% 對照組沒跑時回傳 NaN，讓表格欄位維持對齊。
    if flag
        v = mean(x, 'omitnan');
    else
        v = NaN;
    end
end
