function C1_export_final_gpr()
%% C1_export_final_gpr.m — 用 fitrgp 在「全部資料」訓練一次，匯出 gpr_model.json 給 GUI
% ============================================================
% 設定、標準化、fitrgp 參數全部照 C1_pxToMm_GPR.m(只差在不切摺)。
% GUI 端(gpr_core.py)用匯出的訓練資料 + 超參數 + β 重算預測，數學與 fitrgp
% 'exact' 相同；JSON 附 5 筆 MATLAB predict 的結果(selftest)，Python 載入時逐筆
% 比對，差超過 1e-4 就報錯。
%
% ⚠️ 撰寫環境沒有 MATLAB，未實機跑過；第一次請看 Python 端 selftest 有沒有過。
% ⚠️ 論文數字仍以 C1_pxToMm_GPR.m 的 k-fold 結果為準。
% ============================================================
    METHOD     = 'mask幾何_長邊正規化';      % 跟 GUI 要用的設定一致
    filename   = ['根管填充物像素長度_已配對_11n_tophat_' METHOD '.xlsx'];
    USE_AGE    = true;
    AGE_COLUMN = '年齡';
    GPR_KERNEL = 'squaredexponential';
    GPR_BASIS  = 'linear';
    OUT_JSON   = 'gpr_model.json';

    opts = detectImportOptions(filename);
    opts.VariableNamingRule = 'preserve';
    data = readtable(filename, opts);
    Y = data.('填充物長度(mm)');
    X = data.('像素長度');
    feats = {'像素長度'};
    if USE_AGE
        a = data.(AGE_COLUMN);
        if ~isnumeric(a), a = str2double(string(a)); end
        X = [X a];  feats{end+1} = '年齡';
    end
    ok = all(~isnan(X), 2) & ~isnan(Y);
    X = X(ok, :);  Y = Y(ok);

    mu = mean(X, 1);  sg = std(X, 0, 1);  sg(sg == 0) = 1;
    XN = (X - mu) ./ sg;

    gpr = fitrgp(XN, Y, 'KernelFunction', GPR_KERNEL, 'BasisFunction', GPR_BASIS, ...
                 'Standardize', false);

    n = numel(Y);
    idx = unique(round(linspace(1, n, min(5, n))));
    [m_st, s_st] = predict(gpr, XN(idx, :));
    st = struct('x', {}, 'mean', {}, 'sd', {});
    for i = 1:numel(idx)
        st(i).x = X(idx(i), :);  st(i).mean = m_st(i);  st(i).sd = s_st(i);
    end

    S = struct();
    S.kind = 'gpr';  S.trained_by = 'matlab fitrgp';  S.method = METHOD;
    S.normalized = contains(METHOD, '長邊正規化');
    S.use_age = USE_AGE;  S.features = feats;
    S.kernel = GPR_KERNEL;  S.basis = GPR_BASIS;  S.n = n;
    S.mu = mu;  S.sg = sg;  S.Xn = XN;  S.y = Y;
    S.theta = gpr.KernelInformation.KernelParameters;
    S.sigma = gpr.Sigma;
    S.beta  = gpr.Beta;
    S.feat_range = [min(X(:,1)) max(X(:,1))];
    if USE_AGE, S.age_range = [min(X(:,2)) max(X(:,2))]; end
    S.selftest = st;
    S.source = filename;

    fid = fopen(OUT_JSON, 'w', 'n', 'UTF-8');
    fwrite(fid, jsonencode(S), 'char');
    fclose(fid);
    fprintf('✅ 已寫出 %s（n=%d）\n', OUT_JSON, n);
    fprintf('   σL=%s  σf=%.4f  σn=%.4f  β=%s\n', mat2str(S.theta(1:end-1)', 4), ...
        S.theta(end), S.sigma, mat2str(S.beta', 4));
    fprintf('   把它放到 GUI 資料夾；Python 載入時會自動比對 selftest。\n');
end
