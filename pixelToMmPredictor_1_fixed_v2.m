function pixelToMmPredictor()

    % ---------------- 資料切分設定區 ----------------
    % 👈 改版重點：train/test 不再由 MATLAB 自己重新隨機切分。
    % 原本這裡用 cvpartition 對「配對完的Excel列」重新洗牌 80/20，
    % 但 Excel 裡很多列其實是同一顆牙齒的 Roboflow 擴增版本(_aug1~_aug6)，
    % 如果切分發生在這裡，同一顆牙的不同擴增版本就可能一邊被分進train、
    % 一邊被分進test，等於test偷看過train看過的牙齒 -> 資料洩漏，
    % 也讓test的樣本量看起來比實際「獨立牙齒數」大很多。
    %
    % 正確做法：train/test 的歸屬應該在 Roboflow 匯出、且在做離線擴增
    % 「之前」就決定好(哪幾顆牙是test，永遠不擴增、不出現在train)。
    % 02_train_yolo_pose.py 已經把這個歸屬記錄在Excel的「資料夾來源」
    % 欄位裡(值是 'train' 或 'test')，這裡直接讀這欄位來用，
    % 確保跟YOLO那邊的切分完全一致，不會重新洗牌。
    SPLIT_COLUMN = '資料夾來源';

    % ---------------- 臨床設定區 ----------------
    % 對應流程圖「轉換實際長度 (長度推估 -0.5mm)」：
    % 影像/像素轉換出來的長度會再扣掉這個臨床安全offset，才是實際建議
    % 使用的工作長度。目前先用 -0.5mm，正式值待醫師確認後在這裡改。
    OFFSET_MM = 0.5;

    % 對應流程圖「臨床與安全性評估 (MAE / 理想比率 / 過長率)」：
    % 「理想比率」= 校正後預測值與實際長度的絕對誤差在容忍範圍內的比例。
    % 容忍值待醫師確認臨床可接受誤差後再調整，先用 ±1.0mm 當預設值。
    IDEAL_TOLERANCE_MM = 1.0;

    %% --- 1. 讀取 Python 剛剛輸出的全新 Excel 檔案 ---
    filename = '根管填充物像素長度_已配對.xlsx'; % 👈 直接讀取同資料夾下配對好的新檔
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';  % 👈 保留中文欄位名稱，不然MATLAB會自動改成Var1/拼音等合法變數名，導致後面 data.('圖片檔名') 抓不到欄位
    data = readtable(filename, opts);
    
    %% --- 2. 欄位適應與清理（直接對接醫生的欄位名稱）---
    % 我們直接用「圖片檔名」當作這顆牙齒的 ID，用「填充物長度(mm)」當作 Y，用「像素長度」當作 X
    toothIDs = data.('圖片檔名');               % 👈 改成對接醫生的「圖片檔名」
    actualLengths = data.('填充物長度(mm)');    % 👈 改成對接醫生的「填充物長度(mm)」
    pixelLengths = data.('像素長度');           % 👈 改成對接 Python 算出來的「像素長度」
    
    % 檢查資料是否讀取成功且為數值
    if ~isnumeric(actualLengths) || ~isnumeric(pixelLengths)
        error('錯誤：實際長度或像素長度欄位包含非數值資料，請檢查 Excel！');
    end
    
    % 統計資料筆數
    n = height(data);
    fprintf('成功載入資料，總計 %d 筆牙齒樣本。\n', n);
    
    %% --- [後續的 ANN 網路模型訓練程式碼完全不變，維持你原本的即可] ---
    % 後面如果有用到 data.('長度') 或 data.('像素長度') 的地方，
    % 請確認它們已經被我們上面宣告的變數 actualLengths 和 pixelLengths 取代。
    % 👈 補上：後面第3節開始用的 X/Y/ids 從沒被賦值過，這裡對接起來，
    %    否則會在 zscore(X) 那行直接報 Undefined function or variable 'X'。
    ids = toothIDs;          % 病歷號/圖片檔名，只當ID用，不參與訓練
    X   = pixelLengths;      % 輸入特徵：像素長度
    Y   = actualLengths;     % 輸出目標：實際長度(mm)

    %% 3. 資料標準化
    [Xn, ~, ~] = zscore(X);
    Yn = Y;  % 輸出的長度不標準化，維持 mm 單位

    %% 4. 直接沿用 Roboflow/Python 那邊已經決定好的 train/test 分組
    % 不再自己重新切一次，避免蓋掉「同一顆牙的所有擴增版本都在同一邊」
    % 這個前提。
    if ~ismember(SPLIT_COLUMN, data.Properties.VariableNames)
        error(['錯誤：Excel裡找不到 "', SPLIT_COLUMN, '" 欄位。請確認 02_train_yolo_pose.py ', ...
            '已經是有加上「資料夾來源」欄位的版本，並且重新跑過一次產生新的配對Excel。']);
    end
    splitLabel = string(data.(SPLIT_COLUMN));
    trainIdx = splitLabel == "train";
    testIdx  = splitLabel == "test";

    n_train = sum(trainIdx);
    n_test  = sum(testIdx);

    % 保險檢查：每一列都應該明確屬於 train 或 test 其中一種，
    % 如果有列兩者都不是(例如欄位裡混進 valid 或空值)，要提早示警，
    % 不要讓它們悄悄地被排除在訓練/評估之外而不自知。
    n_unclassified = n - n_train - n_test;
    if n_unclassified > 0
        fprintf('⚠️ 有 %d 筆資料的「%s」欄位既不是train也不是test(可能是valid或空值)，\n', ...
            n_unclassified, SPLIT_COLUMN);
        fprintf('   這些列不會參與訓練也不會參與評估，請檢查Excel內容是否符合預期。\n');
    end

    fprintf('=== 資料切分(沿用Roboflow原始分組，非重新隨機切分)：train %d 筆 (%.0f%%) / test %d 筆 (%.0f%%) ===\n', ...
        n_train, 100 * n_train / n, n_test, 100 * n_test / n);
    if n_test < 10
        fprintf('⚠️ test 只有 %d 筆，樣本數偏小，以下指標(尤其理想比率/過長率)僅供參考，\n', n_test);
        fprintf('   單一筆的對錯就會讓百分比大幅跳動，解讀時請留意這一點。\n');
    end
    if n_test == 0
        error('錯誤：test 筆數為 0，請檢查Excel裡「%s」欄位是否真的有值為 "test" 的列。', SPLIT_COLUMN);
    end

    %% 5. 訓練模型（只用 train，test 完全不參與訓練）
    net = fitnet(5, 'trainbr');
    net.trainParam.showWindow = false;
    net.trainParam.epochs     = 200;

    net.divideFcn            = 'divideind';
    net.divideParam.trainInd = find(trainIdx);
    net.divideParam.valInd   = [];
    net.divideParam.testInd  = find(testIdx);

    net = train(net, Xn', Yn');

    %% 6. 在 test set 上做唯一一次評估（held-out，不會拿來調參）
    Y_test  = Y(testIdx);
    Yp_test = net(Xn(testIdx)')';

    test_residuals = Yp_test - Y_test;
    test_abs_error = abs(test_residuals);

    test_mae  = mean(test_abs_error);
    test_rmse = sqrt(mean(test_residuals.^2));
    test_bias = mean(test_residuals);
    if n_test >= 2
        [R_test, ~] = corr(Y_test, Yp_test);
        test_r2 = R_test^2;
    else
        test_r2 = NaN;
    end

    fprintf('[Test set] RMSE: %.3f | MAE: %.3f | R²: %.3f | Bias: %.3f\n', ...
        test_rmse, test_mae, test_r2, test_bias);

    % --- 6b. 套用臨床offset，算出「實際建議工作長度」，這是流程圖裡
    %         真正會拿去跟醫師臨床判斷比較的數字（不是原始迴歸輸出）---
    Yp_test_offset      = Yp_test - OFFSET_MM;
    test_offset_residuals = Yp_test_offset - Y_test;
    test_offset_abs_error = abs(test_offset_residuals);

    test_offset_mae  = mean(test_offset_abs_error);
    test_offset_bias = mean(test_offset_residuals);

    % 理想比率：絕對誤差落在容忍範圍內的比例
    test_ideal_rate = mean(test_offset_abs_error <= IDEAL_TOLERANCE_MM) * 100;

    % 過長率：offset後的預測值仍然「比實際長度長」的比例
    % （臨床上高估工作長度風險較大，優先於低估，需要單獨追蹤）
    test_overest_rate = mean(test_offset_residuals > 0) * 100;

    %% 7. 用同一個模型預測全部資料（含train，只作參考，不能當成model表現指標）
    Y_pred_all    = net(Xn')';
    all_residuals = Y_pred_all - Y;
    all_abs_error = abs(all_residuals);

    all_mae  = mean(all_abs_error);
    all_rmse = sqrt(mean(all_residuals.^2));
    all_bias = mean(all_residuals);
    [R_all, ~] = corr(Y, Y_pred_all);
    all_r2   = R_all^2;

    Y_pred_all_offset    = Y_pred_all - OFFSET_MM;
    all_offset_residuals = Y_pred_all_offset - Y;
    all_offset_abs_error = abs(all_offset_residuals);

    all_offset_mae     = mean(all_offset_abs_error);
    all_offset_bias    = mean(all_offset_residuals);
    all_ideal_rate      = mean(all_offset_abs_error <= IDEAL_TOLERANCE_MM) * 100;
    all_overest_rate    = mean(all_offset_residuals > 0) * 100;

    %% 8. 整理預測結果，準備輸出到 Excel
    split_label = repmat("train", n, 1);
    split_label(testIdx) = "test";

    is_overest_all = all_offset_residuals > 0;

    ResultsTable = table(ids, split_label, X, Y, ...
        Y_pred_all, all_residuals, all_abs_error, ...
        Y_pred_all_offset, all_offset_residuals, all_offset_abs_error, is_overest_all, ...
        'VariableNames', {'檔名', 'Train_or_Test', '像素長度_px', '實際長度_mm', ...
        '模型預測_mm', '誤差Bias_mm', '絕對誤差_mm', ...
        '校正後預測_mm', '校正後誤差Bias_mm', '校正後絕對誤差_mm', '是否過長'});

    MetricNames = {
        sprintf('A_Test set表現 (held-out %d筆，唯一一次的嚴格指標)', n_test);
        'B_全部資料表現 (含train，僅供參考，不可當model實際能力)';
        '--- 分界線 ---';
        sprintf('C_臨床安全性評估_Test set校正後 (offset=-%.2fmm)', OFFSET_MM);
        sprintf('D_臨床安全性評估_全部資料校正後 (offset=-%.2fmm，僅供參考)', OFFSET_MM)
    };

    MAE_val     = round([test_mae;  all_mae;  NaN; test_offset_mae;  all_offset_mae],  3);
    RMSE_val    = round([test_rmse; all_rmse; NaN; NaN;              NaN],             3);
    R2_val      = round([test_r2;   all_r2;   NaN; NaN;              NaN],             3);
    Bias_val    = round([test_bias; all_bias; NaN; test_offset_bias; all_offset_bias], 3);
    IdealRate_pct   = round([NaN; NaN; NaN; test_ideal_rate;   all_ideal_rate],   1);
    OverestRate_pct = round([NaN; NaN; NaN; test_overest_rate; all_overest_rate], 1);

    MetricsTable = table(MetricNames, MAE_val, RMSE_val, R2_val, Bias_val, IdealRate_pct, OverestRate_pct, ...
        'VariableNames', {'評估範圍_與_嚴格程度', 'MAE_mm', 'RMSE_mm', 'R_Square', 'Mean_Bias_mm', ...
        '理想比率_pct', '過長率_pct'});

    %% 9. 匯出至 Excel 檔案
    output_filename = '預測結果與評估指標_train80_test20_像素轉mm.xlsx';

    writetable(ResultsTable, output_filename, 'Sheet', '所有牙齒預測結果');
    writetable(MetricsTable, output_filename, 'Sheet', '模型評估指標');

    fprintf('\n✅ 所有結果已成功匯出至 Excel 檔案：「 %s 」\n\n', output_filename);
    disp('=== 最終模型評估指標 (已存入 Excel 第二工作表) ===');
    disp(MetricsTable);

    fprintf('\n=== 臨床與安全性評估 (對應流程圖最後一步) ===\n');
    fprintf('offset = -%.2f mm，理想誤差容忍 = ±%.2f mm（兩者皆為預設值，需醫師確認後調整）\n', OFFSET_MM, IDEAL_TOLERANCE_MM);
    fprintf('[Test set，唯一嚴格指標]  MAE(校正後): %.3f mm | 理想比率: %.1f%% | 過長率: %.1f%%\n', ...
        test_offset_mae, test_ideal_rate, test_overest_rate);
    fprintf('⚠️ 過長率是臨床風險較高的指標（預測工作長度仍比實際長 -> 有超出根尖的風險）。\n');
    fprintf('⚠️ 請以 Test set 的數字為準，全部資料(含train)的數字只是參考，不能拿來當model實際表現。\n');

    % 註：畫圖已統一移到 2_plotMyResults.m，避免跟該檔重複維護同一份繪圖程式碼。
    % 這支腳本只負責訓練 + 匯出 Excel，匯出後請執行 2_plotMyResults.m 看圖，
    % 它會自動讀取這裡輸出的 output_filename，不需要手動改檔名或抄指標數字。
    fprintf('👉 接下來請執行 2_plotMyResults.m 產生圖表\n');

end
