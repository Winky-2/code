function C2g_plotGPR_v1()
%% C2g_plotGPR_v1.m
% ============================================================
% 畫 C1_pxToMm_GPR 的結果(四個分頁，版面與 C2x 相同)：
%
%   分頁1 排序點圖
%   分頁2 散布圖（未校正）＋ 指標文字框
%   分頁3 散布圖（校正後，報告用）＋ 指標文字框
%   分頁4 誤差 vs 年齡 / 性別
%
% 讀取：預測結果與評估指標_kfold_GPR_<METHOD><特徵標籤>.xlsx
%   sheet「所有牙齒預測結果」→ 逐筆 Y / 預測 / 校正後 / 摺別 / 年齡 / 性別
%   sheet「模型評估指標」    → A 列(未校正)、B 列(重複間SD)、C 列(校正後)、E 列(線性對照)
%   sheet「模型設定」        → 輸入特徵
%
% 跟 C2x 的差別：C2x 讀 C1x 的「模型對照_kfold_*.xlsx」，格式不同，不能混用。
% 年齡/性別直接在 GPR 輸出檔裡，不用另外 join。
% offset 由「模型預測 − 校正後預測」直接算，不靠設定值，保證跟 C1 一致。
% ============================================================

    %% ---------------- 設定（要跟 C1_pxToMm_GPR 那次一致）----------------
    METHOD   = 'mask幾何_長邊正規化';
    USE_AGE  = true;
    USE_SEX  = false;
    TOL_MM   = 1.0;        % = C1 的 IDEAL_TOLERANCE_MM
    SAVE_PNG = true;

    FEAT_TAG = featTag(USE_AGE, USE_SEX);
    inFile   = sprintf('預測結果與評估指標_kfold_GPR_%s%s.xlsx', METHOD, FEAT_TAG);
    if ~isfile(inFile)
        error(['找不到 %s\n' ...
               '   請先用同樣的 METHOD / USE_AGE / USE_SEX 跑 C1_pxToMm_GPR。'], inFile);
    end
    dd = dir(inFile);
    fprintf('讀取 %s（檔案時間 %s）\n', inFile, dd.date);

    %% ---------------- 讀 GPR 輸出 ----------------
    Per = readtable(inFile, 'Sheet', '所有牙齒預測結果', 'VariableNamingRule','preserve');
    Met = readtable(inFile, 'Sheet', '模型評估指標', ...
                    'VariableNamingRule','preserve', 'TextType','string');
    Set = readtable(inFile, 'Sheet', '模型設定', ...
                    'VariableNamingRule','preserve', 'TextType','string');

    ids    = Per.('檔名'); %#ok<NASGU>
    Y      = Per.('實際長度_mm');
    Y_pred = Per.('模型預測_mm');
    Y_off  = Per.('校正後預測_mm');
    fold   = Per.('摺別');
    n      = numel(Y);
    offsetMM = median(Y_pred - Y_off, 'omitnan');   % C1：校正後 = 預測 − OFFSET_MM

    % --- 指標列(用列名開頭字母定位) ---
    rowName = Met.('評估範圍_與_嚴格程度');
    rowOf = @(tag) find(startsWith(rowName, tag), 1);
    iA = rowOf("A_"); iB = rowOf("B_"); iC = rowOf("C_"); iE = rowOf("E_");
    if isempty(iA) || isempty(iC)
        error('「模型評估指標」找不到 A_ / C_ 列，這不是 C1_pxToMm_GPR 的輸出？');
    end
    mae_raw  = Met.MAE_mm(iA);   rmse_raw = Met.RMSE_mm(iA);   bias_raw = Met.Mean_Bias_mm(iA);
    r_val    = Met.R_Pearson(iA); r2_val  = Met.R_Square(iA);
    mae_off  = Met.MAE_mm(iC);   rmse_off = Met.RMSE_mm(iC);   bias_off = Met.Mean_Bias_mm(iC);
    ideal    = Met.('理想比率_pct')(iC);
    sd_rep   = NaN; if ~isempty(iB), sd_rep = Met.MAE_mm(iB); end
    lin_off  = NaN; if ~isempty(iE), lin_off = Met.MAE_mm(iE); end

    tk = regexp(char(rowName(iA)), '(\d+)次重複', 'tokens', 'once');
    nRep = '?'; if ~isempty(tk), nRep = tk{1}; end
    featDesc = getSetting(Set, '輸入特徵', '像素長度');
    kern = getSetting(Set, 'GPR kernel', '');
    basis = getSetting(Set, 'GPR basis', '');

    % --- 摺間 MAE 標準差(與 C1 終端機「摺間 MAE 標準差」同算法) ---
    fk = unique(fold(fold > 0));
    fmRaw = arrayfun(@(k) mean(abs(Y_pred(fold==k) - Y(fold==k))), fk);
    fmOff = arrayfun(@(k) mean(abs(Y_off(fold==k)  - Y(fold==k))), fk);
    sdFoldRaw = std(fmRaw); sdFoldOff = std(fmOff);

    % --- 年齡 / 性別(GPR 輸出檔內就有) ---
    age = nan(n,1); sex = nan(n,1); demoMsg = '';
    vn = Per.Properties.VariableNames;
    if ismember('年齡', vn), age = toNum(Per.('年齡')); end
    if ismember('性別', vn), sex = parseSex(Per.('性別')); end
    if ~ismember('年齡', vn) && ~ismember('性別', vn)
        demoMsg = 'GPR 輸出檔沒有年齡/性別欄';
    end
    fprintf('年齡有值 %d/%d、性別有值 %d/%d\n', sum(~isnan(age)), n, sum(~isnan(sex)), n);

    % --- 自我檢查：圖上的點與指標表應一致(N_REPEATS=1 時完全相等) ---
    chk = mean(abs(Y_off - Y));
    if abs(chk - mae_off) > 0.005
        fprintf('⚠️ 逐筆重算 MAE %.3f ≠ 指標表 %.3f(多次重複時正常，圖上是代表性重複)\n', chk, mae_off);
    end

    dotC = [0.20 0.40 0.70];
    dotM = [0.20 0.75 0.75];
    ttl  = sprintf('GPR｜%s｜輸入：%s｜Out-of-Fold n=%d', METHOD, featDesc, n);
    pmTxt = '';
    if ~isnan(sd_rep), pmTxt = sprintf(' ± %.3f', sd_rep); end

    %% ---------------- 視窗 + 四個分頁 ----------------
    fig = uifigure('Name', sprintf('GPR 結果 - %s', METHOD), 'Position', [100 80 1020 720]);
    set(fig, 'DefaultTextInterpreter','none', ...
             'DefaultLegendInterpreter','none', ...
             'DefaultAxesTickLabelInterpreter','none');
    tg = uitabgroup(fig, 'Units','normalized', 'Position', [0 0 1 1]);

    %% ===== 分頁1：排序點圖 =====
    [tl1, ax1] = newTab(tg, '1  排序點圖', 1, 1);
    hold(ax1, 'on');
    [Ys, si] = sort(Y);
    x = (1:n)';
    scatter(ax1, x, Ys,         30, 'r',  'filled', 'o', 'DisplayName','實際長度 (Actual)');
    scatter(ax1, x, Y_pred(si), 30, dotC, 'filled', 's', 'DisplayName','Out-of-fold 預測長度');
    xlabel(ax1, sprintf('樣本排序編號（依實際長度由小到大，n=%d）', n));
    ylabel(ax1, '長度 (mm)');
    title(tl1, {'實際 vs 預測 排序點圖', ttl}, 'Interpreter','none');
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
    title(tl2, {'實際 vs 預測（未校正）', ttl}, 'Interpreter','none');
    legend(ax2, 'Location','southeast'); grid(ax2,'on');
    axis(ax2, [lo hi lo hi]);
    t2 = sprintf(['【未校正 OOF 指標】\n模型 : GPR (%s / basis %s)\n量測法 : %s\n輸入 : %s\n' ...
                  'MAE : %.3f%s mm\nRMSE : %.3f mm\nMean Bias : %.3f mm\n' ...
                  'R : %.3f\nR² : %.3f\n摺間 MAE SD : %.3f mm\n(n = %d，%s 次重複)'], ...
                  kern, basis, METHOD, featDesc, mae_raw, pmTxt, rmse_raw, bias_raw, ...
                  r_val, r2_val, sdFoldRaw, n, nRep);
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
    title(tl3, {'實際 vs 校正後預測（報告用）', ttl}, 'Interpreter','none');
    legend(ax3, 'Location','southeast'); grid(ax3,'on');
    axis(ax3, [lo2 hi2 lo2 hi2]);
    linTxt = '';
    if ~isnan(lin_off), linTxt = sprintf('\n線性對照 MAE : %.3f mm', lin_off); end
    t3 = sprintf(['【校正後 OOF 指標（報告用）】\n模型 : GPR (%s / basis %s)\n量測法 : %s\n輸入 : %s\n' ...
                  'MAE : %.3f%s mm\nRMSE : %.3f mm\nMean Bias : %.3f mm\n' ...
                  'R : %.3f\nR² : %.3f\n理想比率 : %.1f%%\n摺間 MAE SD : %.3f mm%s\n(n = %d，%s 次重複)'], ...
                  kern, basis, METHOD, featDesc, mae_off, pmTxt, rmse_off, bias_off, ...
                  r_val, r2_val, ideal, sdFoldOff, linTxt, n, nRep);
    placeText(ax3, t3, 0.30);
    hold(ax3, 'off');

    %% ===== 分頁4：誤差 vs 年齡 / 性別 =====
    res = Y_pred - Y;
    [tl4, ax4a, ax4b] = newTab2(tg, '4  誤差分析');
    title(tl4, {'誤差分析（未校正誤差）', ttl}, 'Interpreter','none');

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
        title(ax4a, sprintf('誤差 vs 年齡 (Spearman ρ=%.2f, p=%.3f)', rA, pA));
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

    okS = ~isnan(sex);
    if sum(okS) >= 2 && numel(unique(sex(okS))) == 2
        isM = sex == 1;
        gx  = double(isM(okS)) + 1;
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

    %% ---------------- 存 PNG ----------------
    if SAVE_PNG
        outDir = sprintf('圖_GPR_%s%s', METHOD, FEAT_TAG);
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

    %% ---------------- 終端機摘要(應與 C1_GPR 的報告區塊相同) ----------------
    fprintf('\n=== GPR 校正後 OOF 指標（報告用）===\n');
    fprintf('  MAE       : %.3f%s mm（摺間 SD %.3f）\n', mae_off, pmTxt, sdFoldOff);
    fprintf('  RMSE      : %.3f mm\n', rmse_off);
    fprintf('  Mean Bias : %.3f mm（正 = 整體偏長）\n', bias_off);
    fprintf('  R / R²    : %.3f / %.3f\n', r_val, r2_val);
    fprintf('  理想比率  : %.1f %%（±%.1f mm 內）\n', ideal, TOL_MM);
    fprintf('  offset    : -%.2f mm；n = %d\n', offsetMM, n);
end


%% ============================================================
%  子函式
%  ============================================================
function [tl, ax] = newTab(tg, titleStr, nr, nc)
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
        'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Color', 'k', ...
        'Interpreter', 'none');
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
    if isnumeric(x), v = double(x); return; end
    v = str2double(regexp(string(x), '\d+(\.\d+)?', 'match', 'once'));
end

function s = parseSex(v)
    t = upper(strtrim(string(v)));
    s = nan(numel(t),1);
    s(ismember(t, ["M","MALE","男"]))   = 1;
    s(ismember(t, ["F","FEMALE","女"])) = 0;
end