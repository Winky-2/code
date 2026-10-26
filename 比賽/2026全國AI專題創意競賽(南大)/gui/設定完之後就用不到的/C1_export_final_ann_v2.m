function C1_export_final_ann()
%% C1_export_final_ann.m — 把 C1 的 ANN 訓練成「最終模型」匯出 reg_model.json 給 GUI
% ============================================================
% C1 是 k-fold 評估：每摺訓練完就丟，沒有可以給新影像用的模型。
% 這支用「完全相同」的設定(z-score → fitnet(5,'trainbr')、dividetrain、
% epochs 200、seed = 42 + r)，在「全部 n 筆」上訓練 N_REPEATS 個網路，
% 匯出成 JSON，GUI 端取 N 個網路的平均當預測值。
%
% ⚠️ 這個模型沒有獨立測試集可評估 —— 論文/報告的效能數字一律用 C1 的
%    out-of-fold 結果；這支印出的「訓練集 MAE」只是確認匯出正確，會偏樂觀。
% ⚠️ 下面設定要跟產出論文主結果的那次 C1 一模一樣(尤其 METHOD、USE_AGE)。
% ============================================================

    %% ---------------- 設定區(抄 C1_v6) ----------------
    RUN_NOW = true;
    METHOD   = 'mask幾何_長邊正規化';
    filename = ['根管填充物像素長度_已配對_' METHOD '.xlsx'];
    USE_AGE  = true;          % TODO：跟論文主結果那次 C1 一致(C1_v6 檔內預設是 false)
    AGE_COLUMN = '年齡';

    N_REPEATS   = 10;
    BASE_SEED   = 42;
    HIDDEN_SIZE = 5;
    TRAIN_FCN   = 'trainbr';
    MAX_EPOCHS  = 200;
    OFFSET_MM   = 0.5;        % 只用來印訊息；GUI 端自己減

    OUT_JSON = 'reg_model.json';   % 放到 app.py / pipeline.py 同一個資料夾

    %% --- 1. 讀資料(剔缺值邏輯同 C1) ---
    if ~isfile(filename), error('找不到 %s(B4 的輸出)', filename); end
    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    data = readtable(filename, opts);
    X = data.('像素長度');
    Y = data.('填充物長度(mm)');
    Xmat = X; featNames = {'像素長度'};
    if USE_AGE
        age = data.(AGE_COLUMN);
        if ~isnumeric(age), age = str2double(string(age)); end
        Xmat = [Xmat age]; featNames{end+1} = '年齡';
    end
    ok = all(~isnan([Xmat Y]), 2);
    Xmat = Xmat(ok, :); Y = Y(ok); n = numel(Y);
    fprintf('n = %d，特徵：%s\n', n, strjoin(featNames, ' + '));
    % 注意：C1 在有年齡/性別欄時也會剔「性別缺值」的列；若 n 跟 C1 印的不同，就是這裡差了。

    %% --- 2. 全資料 z-score(C1 每摺是用訓練摺算，這裡用全部) ---
    mu = mean(Xmat, 1);
    sg = std(Xmat, 0, 1); sg(sg == 0) = 1;
    XN = (Xmat - mu) ./ sg;

    %% --- 3. 訓練 N_REPEATS 個網路 ---
    nets = cell(1, N_REPEATS);
    P = zeros(n, N_REPEATS);
    for r = 1:N_REPEATS
        rng(BASE_SEED + r);
        net = fitnet(HIDDEN_SIZE, TRAIN_FCN);
        net.trainParam.showWindow = false;
        net.trainParam.epochs     = MAX_EPOCHS;
        net.divideFcn = 'dividetrain';
        net = train(net, XN', Y');
        P(:, r) = net(XN')';
        nets{r} = exportNet(net);
        fprintf('  網路 %d/%d 完成\n', r, N_REPEATS);
    end
    yhat = mean(P, 2);
    fprintf('訓練集 MAE(未校正) %.3f mm，校正後 %.3f mm  ← 只是檢查用，會比 C1 的 CV 樂觀\n', ...
        mean(abs(yhat - Y)), mean(abs(yhat - OFFSET_MM - Y)));

    %% --- 4. 自我驗證樣本(GUI 載入時用 Python 重算比對) ---
    idx = unique(round(linspace(1, n, 5)));
    st = struct('x', {}, 'y', {});
    for i = idx
        st(end+1) = struct('x', Xmat(i, :), 'y', yhat(i)); %#ok<AGROW>
    end

    if ~RUN_NOW, disp('RUN_NOW=false：未寫檔'); return; end
    out = struct('kind', 'matlab_ann', 'use_age', USE_AGE, 'features', {featNames}, ...
        'mu', mu, 'sg', sg, 'nets', {nets}, 'selftest', st, 'n', n, ...
        'source', filename, 'hidden', HIDDEN_SIZE, 'train_fcn', TRAIN_FCN, ...
        'time', char(datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss')));
    fid = fopen(OUT_JSON, 'w', 'n', 'UTF-8');
    fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
    fclose(fid);
    fprintf('✅ 已寫出 %s(%d 個網路取平均)\n', OUT_JSON, N_REPEATS);
end


function s = exportNet(net)
% 只支援 fitnet 預設結構：1 隱藏層 tansig + 輸出 purelin，前後處理 mapminmax
    assert(strcmp(net.layers{1}.transferFcn, 'tansig') && strcmp(net.layers{2}.transferFcn, 'purelin'), ...
        '網路結構不是 tansig/purelin，Python 端無法重現');
    s.in_ps  = getMapminmax(net.inputs{1});
    s.out_ps = getMapminmax(net.outputs{2});
    s.IW = net.IW{1, 1};      % HIDDEN x 特徵數
    s.b1 = net.b{1};
    s.LW = net.LW{2, 1};      % 1 x HIDDEN
    s.b2 = net.b{2};
end


function ps = getMapminmax(port)
    f = port.processFcns;
    k = find(strcmp(f, 'removeconstantrows'), 1);
    if ~isempty(k)
        assert(isempty(port.processSettings{k}.remove), ...
            '有特徵是常數被 removeconstantrows 拿掉了，Python 端沒處理這種情況');
    end
    k = find(strcmp(f, 'mapminmax'), 1);
    assert(~isempty(k), '找不到 mapminmax 前處理');
    p = port.processSettings{k};
    ps = struct('xoffset', p.xoffset, 'gain', p.gain, 'ymin', p.ymin);
end
