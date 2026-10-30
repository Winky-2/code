function C2s_plotSingleModel_v2()
%% C2s_plotSingleModel_v2.m
% ============================================================
% 與 v1 相同的四張圖，但全部放進「一個視窗、四個分頁」(uitabgroup)。
%
%   分頁1 排序點圖
%   分頁2 散布圖（未校正）＋ 指標文字框
%   分頁3 散布圖（校正後，報告用）＋ 指標文字框
%   分頁4 誤差 vs 年齡 / 性別
%
% 資料來源：C1x_modelCompare_v1 的「模型對照_kfold_*.xlsx」。
% 年齡/性別不在該檔內，會自動從 B4 的輸入檔用檔名 join 回來；
% join 不到就把分頁4 留空訊息，不會中斷。
% ============================================================

    %% ---------------- 設定（要跟 C1x 那次一致）----------------
    METHOD  = 'mask幾何_長邊正規化';
    USE_AGE = true;
    USE_SEX = false;

    MODEL   = 'gpr';      % constmean/ratio/linear/robust/ann/gpr/svr/lsboost/rf

    SAVE_PNG = true;      % 逐分頁存 PNG
    TOL_MM   = 1.0;

    FEAT_TAG = featTag(USE_AGE, USE_SEX);
    inFile   = sprintf('模型對照_kfold_%s%s.xlsx', METHOD, FEAT_TAG);
    if ~isfile(inFile)
        error(['找不到 %s\n' ...
               '   請先用同樣的 METHOD / USE_AGE / USE_SEX 跑一次 C1x_modelCompare_v1。'], inFile);
    end
    fprintf('讀取 %s（模型 = %s）...\n', inFile, MODEL);

    %% ---------------- 讀 C1x 輸出 ----------------
    Cmp = readtable(inFile, 'Sheet', '模型對照總表', ...
                    'VariableNamingRule','preserve', 'TextType','string');
    Per = readtable(inFile, 'Sheet', '逐筆預測', 'VariableNamingRule','preserve');
    Set = readtable(inFile, 'Sheet', '設定', ...
                    'VariableNamingRule','preserve', 'TextType','string');

    predCol = ['預測_' MODEL];
    if ~ismember(predCol, Per.Properties.VariableNames)
        avail = Per.Properties.VariableNames(startsWith(Per.Properties.VariableNames, '預測_'));
        error('逐筆預測裡沒有「%s」。可用的模型：%s', MODEL, strjoin(erase(avail,'預測_'), ', '));
    end
    i0 = find(string(Cmp.('模型')) == string(MODEL), 1);
    if isempty(i0), error('總表裡找不到模型 %s。', MODEL); end

    offsetMM = str2double(getSetting(Set, 'offset_mm', '0.5'));
    featDesc = getSetting(Set, '輸入特徵', '像素長度');
    nRep     = getSetting(Set, '重複次數', '?');

    ids    = Per.('檔名');
    Y      = Per.('實際長度_mm');
    Y_off  = Per.(predCol);
    Y_pred = Y_off + offsetMM;
    n      = numel(Y);

    mae_off   = Cmp.('校正後MAE_mm')(i0);
    sd_off    = Cmp.('校正後MAE_SD')(i0);
    mae_raw   = Cmp.('未校正MAE_mm')(i0);
    rmse_raw  = Cmp.('未校正RMSE_mm')(i0);
    bias_off  = Cmp.('校正後Bias_mm')(i0);
    r_val     = Cmp.('R_Pearson')(i0);
    r2_val    = Cmp.('R_Square')(i0);
    ideal_val = Cmp.('理想比率_pct')(i0);
    rmse_off  = sqrt(mean((Y_off - Y).^2));   % 總表沒存，由代表性重複重算
    bias_raw  = mean(Y_pred - Y);

    %% ---------------- join 年齡 / 性別 ----------------
    demoFile = sprintf('根管填充物像素長度_已配對_%s.xlsx', METHOD);
    age = nan(n,1); sex = nan(n,1); hasDemo = false; demoMsg = '';
    if isfile(demoFile)
        try
            o = detectImportOptions(demoFile); o.VariableNamingRule = 'preserve';
            if ismember('性別', o.VariableNames), o = setvartype(o, '性別', 'string'); end
            D = readtable(demoFile, o);
            if ismember('圖片檔名', D.Properties.VariableNames)
                [tf, loc] = ismember(string(ids), string(D.('圖片檔名')));
                if ismember('年齡', D.Properties.VariableNames)
                    a = toNum(D.('年齡')); age(tf) = a(loc(tf));
                end
                if ismember('性別', D.Properties.VariableNames)
                    s = parseSex(D.('性別')); sex(tf) = s(loc(tf));
                end
                hasDemo = any(~isnan(age)) || any(~isnan(sex));
                fprintf('已從 %s join 到 %d/%d 筆的年齡/性別。\n', demoFile, sum(tf), n);
            else
                demoMsg = sprintf('%s 沒有「圖片檔名」欄，無法對應。', demoFile);
            end
        catch ME
            demoMsg = sprintf('讀 %s 失敗：%s', demoFile, ME.message);
        end
    else
        demoMsg = sprintf('找不到 %s，無法取得年齡/性別。', demoFile);
    end
    if ~isempty(demoMsg), fprintf('%s\n', demoMsg); end

    dotC = [0.20 0.40 0.70];   % F / 一般
    dotM = [0.20 0.75 0.75];   % M
    ttl  = sprintf('%s｜%s｜輸入：%s｜Out-of-Fold n=%d', MODEL, METHOD, featDesc, n);

    %% ---------------- 建立單一視窗 + 四個分頁 ----------------
    fig = uifigure('Name', sprintf('單模型結果 - %s（%s）', MODEL, METHOD), ...
                   'Position', [100 80 1020 720]);
    tg  = uitabgroup(fig, 'Units','normalized', 'Position', [0 0 1 1]);

    %% ===== 分頁1：排序點圖 =====
    [tl1, ax1] = newTab(tg, '1  排序點圖', 1, 1);
    hold(ax1, 'on');
    [Ys, si] = sort(Y);
    Yps = Y_pred(si);
    x = (1:n)';
    hl = plot(ax1, [x x]', [Ys Yps]', '-', 'Color',[0.7 0.7 0.7], 'LineWidth',0.8);
    set(hl, 'HandleVisibility','off');
    scatter(ax1, x, Ys,  30, 'r',  'filled', 'o', 'DisplayName','實際長度 (Actual)');
    scatter(ax1, x, Yps, 30, dotC, 'filled', 's', 'DisplayName','Out-of-fold 預測長度');
    xlabel(ax1, sprintf('樣本排序編號（依實際長度由小到大，n=%d）', n));
    ylabel(ax1, '長度 (mm)');
    title(tl1, {'實際 vs 預測 排序點圖', ttl});
    legend(ax1, 'Location','northwest'); grid(ax1,'on'); hold(ax1,'off');

    %% ===== 分頁2：散布圖（未校正）=====
    [tl2, ax2] = newTab(tg, '2  散布圖（未校正）', 1, 1);
    hold(ax2, 'on');
    if any(~isnan(sex))
        isM = sex == 1;
        scatter(ax2, Y(~isM), Y_pred(~isM), 42, dotC, 'filled','o','DisplayName','OOF 預測 (F)');
        scatter(ax2, Y(isM),  Y_pred(isM),  46, dotM, 'filled','s','DisplayName','OOF 預測 (M)');
    else
        scatter(ax2, Y, Y_pred, 42, dotC, 'filled', 'DisplayName','Out-of-fold 預測');
    end
    lo = floor(min([Y; Y_pred])) - 1; hi = ceil(max([Y; Y_pred])) + 1;
    plot(ax2, [lo hi],[lo hi], 'k-', 'LineWidth',2, 'DisplayName','完美預測線 (誤差 0)');
    plot(ax2, [lo hi],[lo+0.5 hi+0.5], 'r--','LineWidth',1.4,'DisplayName','+0.5 mm');
    plot(ax2, [lo hi],[lo-0.5 hi-0.5], 'r--','LineWidth',1.4,'DisplayName','-0.5 mm');
    plot(ax2, [lo hi],[lo+1.0 hi+1.0], 'g:', 'LineWidth',1.4,'DisplayName','+1.0 mm');
    plot(ax2, [lo hi],[lo-1.0 hi-1.0], 'g:', 'LineWidth',1.4,'DisplayName','-1.0 mm');
    xlabel(ax2, '實際長度 (mm)'); ylabel(ax2, '預測長度 (mm)');
    title(tl2, {'實際 vs 預測（未校正）', ttl});
    legend(ax2, 'Location','southeast'); grid(ax2,'on');
    axis(ax2, [lo hi lo hi]);
    t2 = sprintf(['【未校正 OOF 指標】\n模型 : %s\n量測法 : %s\n輸入 : %s\n' ...
                  'MAE : %.3f mm\nRMSE : %.3f mm\nMean Bias : %.3f mm\n' ...
                  'R : %.3f\nR^2 : %.3f\n(n = %d，%s 次重複平均)'], ...
                  MODEL, METHOD, featDesc, mae_raw, rmse_raw, bias_raw, r_val, r2_val, n, nRep);
    placeText(ax2, t2, 0.22);
    hold(ax2, 'off');

    %% ===== 分頁3：散布圖（校正後）=====
    [tl3, ax3] = newTab(tg, '3  散布圖（校正後）', 1, 1);
    hold(ax3, 'on');
    scatter(ax3, Y, Y_off, 46, dotC, 'filled', 'DisplayName','校正後預測');
    lo2 = floor(min([Y; Y_off])) - 1; hi2 = ceil(max([Y; Y_off])) + 1;
    plot(ax3, [lo2 hi2],[lo2 hi2], 'k-', 'LineWidth',2,'DisplayName','完美預測線 (誤差 0)');
    plot(ax3, [lo2 hi2],[lo2+TOL_MM hi2+TOL_MM], 'g:','LineWidth',1.4, ...
        'DisplayName',sprintf('+%.1f mm 容忍線',TOL_MM));
    plot(ax3, [lo2 hi2],[lo2-TOL_MM hi2-TOL_MM], 'g:','LineWidth',1.4, ...
        'DisplayName',sprintf('-%.1f mm 容忍線',TOL_MM));
    xlabel(ax3, '實際長度 (mm)');
    ylabel(ax3, sprintf('校正後預測長度 = 模型輸出 - %.2f mm', offsetMM));
    title(tl3, {'實際 vs 校正後預測（報告用）', ttl});
    legend(ax3, 'Location','southeast'); grid(ax3,'on');
    axis(ax3, [lo2 hi2 lo2 hi2]);
    t3 = sprintf(['【校正後 OOF 指標（報告用）】\n模型 : %s\n量測法 : %s\n輸入 : %s\n' ...
                  'MAE : %.3f ± %.3f mm\nRMSE : %.3f mm\nMean Bias : %.3f mm\n' ...
                  'R : %.3f\nR^2 : %.3f\n理想比率 : %.1f%%\n(n = %d，%s 次重複平均)'], ...
                  MODEL, METHOD, featDesc, mae_off, sd_off, rmse_off, bias_off, ...
                  r_val, r2_val, ideal_val, n, nRep);
    placeText(ax3, t3, 0.30);
    hold(ax3, 'off');

    %% ===== 分頁4：誤差 vs 年齡 / 性別 =====
    res = Y_pred - Y;
    [tl4, ax4a, ax4b] = newTab2(tg, '4  誤差分析');
    title(tl4, {'誤差分析', ttl});

    % --- 年齡 ---
    okA = ~isnan(age);
    if sum(okA) >= 3 && range(age(okA)) > 0
        hold(ax4a, 'on');
        if any(~isnan(sex))
            isM = sex == 1;
            scatter(ax4a, age(okA & ~isM), res(okA & ~isM), 42, dotC, 'filled','o','DisplayName','F');
            scatter(ax4a, age(okA &  isM), res(okA &  isM), 46, dotM, 'filled','s','DisplayName','M');
        else
            scatter(ax4a, age(okA), res(okA), 42, dotC, 'filled', 'DisplayName','OOF 誤差');
        end
        pp = polyfit(age(okA), res(okA), 1);
        xa = [min(age(okA)) max(age(okA))];
        plot(ax4a, xa, polyval(pp, xa), '-', 'Color',[0.85 0.65 0.13], 'LineWidth',1.8, ...
            'DisplayName', sprintf('趨勢 %.3f mm/歲', pp(1)));
        yline(ax4a, 0, 'k-', 'HandleVisibility','off');
        rA = NaN; pA = NaN;
        try, [rA, pA] = corr(age(okA), res(okA), 'Type','Spearman'); catch, end
        title(ax4a, sprintf('誤差 vs 年齡 (Spearman \\rho=%.2f, p=%.3f)', rA, pA));
        legend(ax4a, 'Location','best');
        hold(ax4a, 'off');
        fprintf('\n=== 誤差分析 ===\n');
        fprintf('  年齡：Spearman ρ=%.3f, p=%.3f, 斜率 %.4f mm/歲\n', rA, pA, pp(1));
    else
        msg = '年齡缺值或為常數，無法分析';
        if ~isempty(demoMsg), msg = demoMsg; end
        text(ax4a, 0.5, 0.5, msg, 'HorizontalAlignment','center', 'Units','normalized');
        title(ax4a, '誤差 vs 年齡');
    end
    xlabel(ax4a, '年齡 (歲)'); ylabel(ax4a, '誤差 = 預測 - 實際 (mm)'); grid(ax4a,'on');

    % --- 性別 ---
    okS = ~isnan(sex);
    if sum(okS) >= 2 && numel(unique(sex(okS))) == 2
        isM = sex == 1;
        gx  = double(isM(okS)) + 1;      % F=1, M=2
        rs  = res(okS);
        boxchart(ax4b, gx, rs, 'MarkerStyle','none', 'BoxFaceColor',[0.35 0.55 0.80]);
        hold(ax4b, 'on');
        rng(0); jit = (rand(numel(rs),1) - 0.5) * 0.25;
        scatter(ax4b, gx + jit, rs, 26, [0.35 0.35 0.35], 'filled', 'MarkerFaceAlpha',0.6);
        yline(ax4b, 0, 'k-');
        hold(ax4b, 'off');
        xticks(ax4b, [1 2]); xticklabels(ax4b, {'F','M'}); xlim(ax4b, [0.5 2.5]);
        pS = NaN;
        try, pS = ranksum(res(okS & isM), res(okS & ~isM)); catch, end
        maeF = mean(abs(res(okS & ~isM))); maeM = mean(abs(res(okS & isM)));
        title(ax4b, sprintf('誤差依性別 (MAE F=%.2f / M=%.2f, 秩和 p=%.3f)', maeF, maeM, pS));
        fprintf('  性別：F n=%d MAE=%.3f | M n=%d MAE=%.3f | 秩和 p=%.3f\n', ...
            sum(okS & ~isM), maeF, sum(okS & isM), maeM, pS);
        fprintf('  n 小、單次檢定，p 值只當參考。\n');
    else
        msg = '性別缺值或只有單一組別';
        if ~isempty(demoMsg), msg = demoMsg; end
        text(ax4b, 0.5, 0.5, msg, 'HorizontalAlignment','center', 'Units','normalized');
        title(ax4b, '誤差依性別');
    end
    ylabel(ax4b, '誤差 = 預測 - 實際 (mm)'); grid(ax4b,'on');

    %% ---------------- 存 PNG（逐分頁）----------------
    if SAVE_PNG
        outDir = sprintf('圖_單模型_%s_%s%s', MODEL, METHOD, FEAT_TAG);
        if ~isfolder(outDir), mkdir(outDir); end
        names = {'1_排序點圖','2_散布圖_未校正','3_散布圖_校正後','4_誤差分析'};
        tls   = {tl1, tl2, tl3, tl4};
        for i = 1:4
            try
                exportgraphics(tls{i}, fullfile(outDir, [names{i} '.png']), 'Resolution', 200);
            catch ME
                fprintf('存 %s 失敗：%s\n', names{i}, ME.message);
            end
        end
        fprintf('\n圖已存到：%s\n', outDir);
    end

    %% ---------------- 終端機摘要 ----------------
    fprintf('\n=== %s 校正後 OOF 指標（報告用）===\n', MODEL);
    fprintf('  MAE       : %.3f ± %.3f mm\n', mae_off, sd_off);
    fprintf('  RMSE      : %.3f mm   （由代表性重複重算）\n', rmse_off);
    fprintf('  Mean Bias : %.3f mm   （正 = 整體偏長）\n', bias_off);
    fprintf('  R / R^2   : %.3f / %.3f   （offset 不影響）\n', r_val, r2_val);
    fprintf('  理想比率  : %.1f %%  （±%.1f mm 內）\n', ideal_val, TOL_MM);
    fprintf('  n = %d，%s 次重複平均；圖上的點是最接近中位數 MAE 的那次重複。\n', n, nRep);
    fprintf('報告時務必附上 n 與標準差。\n');
end


%% ============================================================
%  子函式
%  ============================================================
function [tl, ax] = newTab(tg, titleStr, nr, nc)
% 一個分頁 = 一個 uitab + tiledlayout（用 tiledlayout 才方便 exportgraphics 整頁）
    t  = uitab(tg, 'Title', titleStr);
    tl = tiledlayout(t, nr, nc, 'Padding','compact', 'TileSpacing','compact');
    ax = nexttile(tl);
end

function [tl, axA, axB] = newTab2(tg, titleStr)
    t   = uitab(tg, 'Title', titleStr);
    tl  = tiledlayout(t, 1, 2, 'Padding','compact', 'TileSpacing','compact');
    axA = nexttile(tl);
    axB = nexttile(tl);
end

function placeText(ax, txt, yRatio)
    xl = xlim(ax); yl = ylim(ax);
    text(ax, xl(1) + 0.05*(xl(2)-xl(1)), yl(2) - yRatio*(yl(2)-yl(1)), txt, ...
        'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Color', 'k');
end

function v = getSetting(Set, key, defaultVal)
    v = defaultVal;
    hit = string(Set.('項目')) == string(key);
    if any(hit), v = char(string(Set.('值')(find(hit,1)))); end
end

function tag = featTag(useAge, useSex)
    parts = {};
    if useAge, parts{end+1} = '年齡'; end
    if useSex, parts{end+1} = '性別'; end
    if isempty(parts), tag = ''; else, tag = ['_' strjoin(parts,'')]; end
end

function v = toNum(x)
    if iscell(x) || isstring(x), v = str2double(string(x)); else, v = double(x); end
end

function s = parseSex(v)
    t = upper(strtrim(string(v)));
    s = nan(numel(t),1);
    s(ismember(t, ["M","MALE","男"]))   = 1;
    s(ismember(t, ["F","FEMALE","女"])) = 0;
end
