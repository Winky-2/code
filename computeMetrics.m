function m = computeMetrics(Y, Yp, offsetMm, tolMm)
% computeMetrics  全流程唯一的指標計算入口（A_ 與 B_ 兩支腳本共用）
% ============================================================
% *** 為什麼要抽成獨立檔案 ***
% 之前 A_pixelToMmPredictor 算一次指標寫進 Excel，B_plotMyResults 再從
% Excel 讀數字印在圖上。只要有人手動改過其中一邊的算法（例如 R² 換定義、
% MAE 改成只算 test），兩邊就會悄悄地不一致，而且圖表上的數字看起來
% 一樣正常，很難發現。現在兩支都呼叫這個函式，算法只有一份，
% 不可能對不起來。
%
% 輸入：
%   Y        實際長度 (mm)，欄向量
%   Yp       模型預測 (mm)，欄向量，長度需與 Y 相同
%   offsetMm 臨床安全 offset (mm)，正值。校正後預測 = Yp - offsetMm
%   tolMm    理想比率的容忍值 (mm)，例如 1.0 代表 ±1.0mm 內算理想
%
% 輸出 struct m 的欄位：
%   n            樣本數
%   MAE          平均絕對誤差 (mm)
%   RMSE         均方根誤差 (mm)
%   Bias         平均誤差（正值 = 平均而言高估）
%   R2           決定係數 1 - SSE/SST  👈 主要指標
%   PearsonR2    Pearson 相關係數平方  👈 僅供對照
%   MAE_offset / Bias_offset      校正後的 MAE / Bias
%   IdealRate    理想比率 (%)：校正後絕對誤差 <= tolMm 的比例
%   OverestRate  過長率 (%)：校正後預測仍然比實際長的比例
%   Resid / AbsErr / PredOffset / ResidOffset / AbsErrOffset / IsOverest
%       逐筆的中間結果，讓呼叫端直接拿去組表格或畫圖，
%       不用自己再算一次（自己再算一次就又有不一致的風險）
%
% *** R² 用哪一種定義 ***
% 這裡的 R2 是決定係數 1 - SSE/SST，會把系統性偏移(bias)算進去：
% 如果模型每一筆都高估 3mm，R2 會很難看，但 PearsonR2 依然接近 1。
% 臨床上我們在意的是「預測值本身準不準」，不是「趨勢對不對」，
% 所以主要指標用 R2。兩個都輸出是為了方便你解讀：
%   R2 低但 PearsonR2 高  -> 模型抓到趨勢了，但有系統性偏移，
%                            通常靠校正 offset 或重新訓練就能改善
%   兩個都低              -> 模型根本沒抓到 像素->mm 的關係
% 注意 R2 可能是負的（代表比「直接猜平均值」還差），這是正常的，不是 bug。

    Y  = double(Y(:));
    Yp = double(Yp(:));

    if numel(Y) ~= numel(Yp)
        error('computeMetrics: Y 與 Yp 長度不一致 (%d vs %d)', numel(Y), numel(Yp));
    end

    % 只用兩邊都有效的樣本，避免 NaN 把整組指標污染成 NaN
    valid = ~isnan(Y) & ~isnan(Yp);
    Y  = Y(valid);
    Yp = Yp(valid);

    n = numel(Y);
    m.n = n;

    if n == 0
        [m.MAE, m.RMSE, m.Bias, m.R2, m.PearsonR2] = deal(NaN);
        [m.MAE_offset, m.Bias_offset, m.IdealRate, m.OverestRate] = deal(NaN);
        [m.Resid, m.AbsErr, m.PredOffset, m.ResidOffset, m.AbsErrOffset, m.IsOverest] = deal([]);
        return;
    end

    %% --- 原始（未校正）指標 ---
    resid  = Yp - Y;
    absErr = abs(resid);

    m.Resid  = resid;
    m.AbsErr = absErr;
    m.MAE    = mean(absErr);
    m.RMSE   = sqrt(mean(resid.^2));
    m.Bias   = mean(resid);

    % 決定係數：1 - SSE/SST
    SST = sum((Y - mean(Y)).^2);
    if n < 2 || SST == 0
        % n=1 或所有實際長度都一樣時，SST=0，R² 沒有定義
        m.R2 = NaN;
    else
        m.R2 = 1 - sum(resid.^2) / SST;
    end

    % Pearson r²：這裡手動算，不呼叫 corr()，
    % 免得沒有 Statistics Toolbox 的電腦跑不動
    if n < 3
        % n<3 時相關係數幾乎必然是 ±1，沒有參考價值，直接給 NaN
        m.PearsonR2 = NaN;
    else
        dY  = Y  - mean(Y);
        dYp = Yp - mean(Yp);
        denom = sqrt(sum(dY.^2) * sum(dYp.^2));
        if denom == 0
            m.PearsonR2 = NaN;
        else
            r = sum(dY .* dYp) / denom;
            m.PearsonR2 = r^2;
        end
    end

    %% --- 校正後（臨床）指標 ---
    % 對應流程圖「轉換實際長度 (長度推估 -0.5mm)」+「臨床與安全性評估」
    predOffset   = Yp - offsetMm;
    residOffset  = predOffset - Y;
    absErrOffset = abs(residOffset);

    m.PredOffset   = predOffset;
    m.ResidOffset  = residOffset;
    m.AbsErrOffset = absErrOffset;
    m.IsOverest    = residOffset > 0;

    m.MAE_offset  = mean(absErrOffset);
    m.Bias_offset = mean(residOffset);

    % 理想比率：校正後絕對誤差落在容忍範圍內的比例
    m.IdealRate = mean(absErrOffset <= tolMm) * 100;

    % 過長率：校正後預測「仍然比實際長」的比例
    % 臨床上高估工作長度的風險高於低估（有超出根尖的可能），單獨追蹤
    m.OverestRate = mean(residOffset > 0) * 100;

end
