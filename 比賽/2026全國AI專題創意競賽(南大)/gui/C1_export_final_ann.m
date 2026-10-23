function C1_export_final_ann()
%% C1_export_final_ann.m — 把 C1 的 ANN 在「全部資料」上訓練一次，匯出給 GUI
% ============================================================
% C1_v6 只做 k-fold 評估，不保存模型。這支用跟 C1 完全相同的設定
% (標準化方式、fitnet、trainbr、hidden 5、epochs 200)在全部資料上訓練一個，
% 把權重與 mapminmax 參數寫成 reg_model.json，Python 端 Regressor 會讀。
%
% JSON 裡附 5 筆 MATLAB 自己算的預測(selftest)，Python 載入時逐筆比對，
% 差超過 1e-4 就報錯 → 確保兩邊算的是同一個函數。
%
% ⚠️ 未在 MATLAB 實機測試過(撰寫環境沒有 MATLAB)，第一次跑請看 selftest 有沒有過。
% ⚠️ ANN 有隨機性；這個模型的數字不會等於論文表裡的 CV 平均，論文數字仍以 C1 為準。
% ============================================================
    METHOD   = 'mask幾何_長邊正規化';
    filename = ['根管填充物像素長度_已配對_26n_' METHOD '.xlsx'];
    USE_AGE  = true;
    AGE_COLUMN = '年齡';
    HIDDEN_SIZE = 5;  TRAIN_FCN = 'trainbr';  MAX_EPOCHS = 200;
    SEED = 43;        % = C1 的 BASE_SEED + 1
    OUT_JSON = 'reg_model.json';

    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    data = readtable(filename, opts);
    Y = data.('填充物長度(mm)');
    Xmat = data.('像素長度');
    if USE_AGE
        age = data.(AGE_COLUMN);
        if ~isnumeric(age), age = str2double(string(age)); end
        Xmat = [Xmat age];
    end
    ok = all(~isnan(Xmat), 2) & ~isnan(Y);
    Xmat = Xmat(ok, :);  Y = Y(ok);

    mu = mean(Xmat, 1);  sg = std(Xmat, 0, 1);  sg(sg == 0) = 1;
    XN = (Xmat - mu) ./ sg;

    rng(SEED);
    net = fitnet(HIDDEN_SIZE, TRAIN_FCN);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = MAX_EPOCHS;
    net.divideFcn = 'dividetrain';
    net = train(net, XN', Y');

    inPs  = getPs(net.inputs{1},  'mapminmax');
    outPs = getPs(net.outputs{2}, 'mapminmax');
    rcIn  = getPs(net.inputs{1},  'removeconstantrows');
    if ~isempty(rcIn) && ~isempty(rcIn.remove)
        error('有輸入欄被 removeconstantrows 移除(某特徵是常數)，Python 前向不支援，請檢查資料。');
    end

    n = min(5, numel(Y));
    st = struct('x', {}, 'y', {});
    for i = 1:n
        st(i).x = Xmat(i, :);
        st(i).y = net(XN(i, :)');
    end

    S = struct();
    S.kind = 'matlab_ann';
    S.use_age = USE_AGE;
    S.n = numel(Y);
    S.source = filename;
    S.mu = mu;  S.sg = sg;
    S.in_ps  = struct('xoffset', inPs.xoffset',  'gain', inPs.gain',  'ymin', inPs.ymin);
    S.out_ps = struct('xoffset', outPs.xoffset, 'gain', outPs.gain, 'ymin', outPs.ymin);
    S.IW = net.IW{1, 1};
    S.b1 = net.b{1}';
    S.LW = net.LW{2, 1};
    S.b2 = net.b{2};
    S.selftest = st;

    fid = fopen(OUT_JSON, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(S, 'PrettyPrint', true), 'char');
    fclose(fid);
    fprintf('✅ 已寫出 %s（n=%d，特徵數 %d）\n', OUT_JSON, S.n, size(Xmat, 2));
    fprintf('   把它放到 GUI 資料夾，Python 載入時會自動比對 selftest。\n');
end


function ps = getPs(port, fcnName)
    ps = [];
    idx = find(strcmp(port.processFcns, fcnName), 1);
    if ~isempty(idx)
        ps = port.processSettings{idx};
    end
end
