function C2s_plotSingleModel_v1()
%% C2s_plotSingleModel_v1.m
% ============================================================
% 搭配 C1x_modelCompare_v1.m：從對照輸出裡挑「一個模型」，
% 畫成跟原本 C2 一樣的四張圖。
%
%   圖1 排序點圖（實際 vs 預測，依實際長度排序）
%   圖2 預測 vs 實際 散布圖（未校正）＋ 指標文字框
%   圖3 預測 vs 實際 散布圖（校正後，報告用）＋ 指標文字框
%   圖4 誤差 vs 年齡 / 性別
%
% 年齡與性別不在 C1x 的輸出裡，這支會自動從 B4 的輸入檔
% 「根管填充物像素長度_已配對_<METHOD>.xlsx」用檔名 join 回來；
% join 不到就自動略過圖4，不會中斷。
%
% 點用的是該模型「最接近中位數 MAE 的那次重複」；
% 文字框的指標是 10 次重複的平均（與 C1x 總表一致），兩者定義不同，
% 所以文字框的 MAE 不會剛好等於圖上點自己算出來的 MAE。
% ============================================================

    %% ---------------- 設定（要跟 C1x 那次一致）----------------
    METHOD  = 'mask幾何_長邊正規化';
    USE_AGE = true;
    USE_SEX = false;

    MODEL   = 'gpr';      % 要畫哪一個模型：constmean/ratio/linear/robust/ann/gpr/svr/lsboost/rf

    DARK_MODE = false;    % true = 跟原本 C2 一樣的白線＋黑底文字框（深色 figure 主題用）
    SAVE_PNG  = true;
    TOL_MM    = 1.0;

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
    Xpx    = Per.('像素長度_px');
    Y      = Per.('實際長度_mm');
    Y_off  = Per.(predCol);          % 校正後預測
    Y_pred = Y_off + offsetMM;       % 未校正預測
    n      = numel(Y);

    % --- 指標（取自總表，10 次重複平均）---
    mae_off   = Cmp.('校正後MAE_mm')(i0);
    sd_off    = Cmp.('校正後MAE_SD')(i0);
    mae_raw   = Cmp.('未校正MAE_mm')(i0);
    rmse_raw  = Cmp.('未校正RMSE_mm')(i0);
    bias_off  = Cmp.('校正後Bias_mm')(i0);
    r_val     = Cmp.('R_Pearson')(i0);
    r2_val    = Cmp.('R_Square')(i0);
    ideal_val = Cmp.('理想比率_pct')(i0);
    % 未校正 RMSE 總表有，校正後 RMSE 沒有 → 由逐筆重算（代表性重複）
    rmse_off  = sqrt(mean((Y_off - Y).^2));
    bias_raw  = mean(Y_pred - Y);

    %% ---------------- 把年齡/性別 join 回來 ----------------
    demoFile = sprintf('根管填充物像素長度_已配對_%s.xlsx', METHOD);
    age = nan(n,1); sex = nan(n,1); hasDemo = false;
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
            end
        catch ME
            fprintf('讀 %s 失敗（%s），略過圖4。\n', demoFile, ME.message);
        end
    else
        fprintf('找不到 %s，略過圖4（年齡/性別）。\n', demoFile);
    end

    %% ---------------- 外觀 ----------------
    if DARK_MODE
        lineC = 'w'; txtBG = 'k'; txtFG = 'w'; txtEdge = 'w'; dotC = [0.3 0.6 1.0];
    else
        lineC = 'k'; txtBG = 'w'; txtFG = 'k'; txtEdge = 'k'; dotC = [0.20 0.40 0.70];
    end
    outDir = sprintf('圖_單模型_%s_%s%s', MODEL, METHOD, FEAT_TAG);
    if SAVE_PNG && ~isfolder(outDir), mkdir(outDir); end
    ttl = sprintf('%s｜%s｜輸入：%s｜Out-of-Fold n=%d', MODEL, METHOD, featDesc, n);

    %% ---------------- 圖1：排序點圖 ----------------
    f1 = figure('Name',['排序點圖 - ' MODEL],'NumberTitle','off','Position',[80 80 860 520]);
    hold on;
    [Ys, si] = sort(Y);
    Yps = Y_pred(si);
    x = (1:n)';
    hl = plot([x x]', [Ys Yps]', '-', 'Color',[0.7 0.7 0.7], 'LineWidth',0.8);
    set(hl, 'HandleVisibility','off');
    scatter(x, Ys,  30, 'r', 'filled', 'o', 'DisplayName','實際長度 (Actual)');
    scatter(x, Yps, 30, dotC, 'filled', 's', 'DisplayName','Out-of-fold 預測長度');
    xlabel(sprintf('樣本排序編號（依實際長度由小到大，n=%d）', n));
    ylabel('長度 (mm)');
    title({['實際 vs 預測 排序點圖'], ttl});
    legend('Location','northwest'); grid on; hold off;
    savePNG(f1, SAVE_PNG, outDir, '1_排序點圖');

    %% ---------------- 圖2：散布圖（未校正）----------------
    f2 = figure('Name',['散布圖 未校正 - ' MODEL],'NumberTitle','off','Position',[100 90 760 640]);
    hold on;
    if hasDemo && any(~isnan(sex))
        isM = sex == 1;
        scatter(Y(~isM), Y_pred(~isM), 42, dotC, 'filled','o','DisplayName','OOF 預測 (F)');
        scatter(Y(isM),  Y_pred(isM),  46, [0.2 0.75 0.75], 'filled','s','DisplayName','OOF 預測 (M)');
    else
        scatter(Y, Y_pred, 42, dotC, 'filled', 'DisplayName','Out-of-fold 預測');
    end
    lo = floor(min([Y; Y_pred])) - 1; hi = ceil(max([Y; Y_pred])) + 1;
    plot([lo hi],[lo hi], [lineC '-'], 'LineWidth',2, 'DisplayName','完美預測線 (誤差 0)');
    plot([lo hi],[lo+0.5 hi+0.5], 'r--','LineWidth',1.4,'DisplayName','+0.5 mm');
    plot([lo hi],[lo-0.5 hi-0.5], 'r--','LineWidth',1.4,'DisplayName','-0.5 mm');
    plot([lo hi],[lo+1.0 hi+1.0], 'g:', 'LineWidth',1.4,'DisplayName','+1.0 mm');
    plot([lo hi],[lo-1.0 hi-1.0], 'g:', 'LineWidth',1.4,'DisplayName','-1.0 mm');
    xlabel('實際長度 (mm)'); ylabel('預測長度 (mm)');
    title({'實際 vs 預測（未校正）', ttl});
    legend('Location','southeast'); grid on; axis([lo hi lo hi]);
    t2 = sprintf(['【未校正 OOF 指標】\n模型 : %s\n量測法 : %s\n輸入 : %s\n' ...
                  'MAE : %.3f mm\nRMSE : %.3f mm\nMean Bias : %.3f mm\n' ...
                  'R : %.3f\nR^2 : %.3f\n(n = %d，%s 次重複平均)'], ...
                  MODEL, METHOD, featDesc, mae_raw, rmse_raw, bias_raw, r_val, r2_val, n, nRep);
    placeText(t2, 0.20, txtBG, txtFG, txtEdge);
    hold off;
    savePNG(f2, SAVE_PNG, outDir, '2_散布圖_未校正');

    %% ---------------- 圖3：散布圖（校正後，報告用）----------------
    f3 = figure('Name',['散布圖 校正後 - ' MODEL],'NumberTitle','off','Position',[120 100 760 640]);
    hold on;
    scatter(Y, Y_off, 46, dotC, 'filled', 'DisplayName','校正後預測');
    lo2 = floor(min([Y; Y_off])) - 1; hi2 = ceil(max([Y; Y_off])) + 1;
    plot([lo2 hi2],[lo2 hi2], [lineC '-'], 'LineWidth',2,'DisplayName','完美預測線 (誤差 0)');
    plot([lo2 hi2],[lo2+TOL_MM hi2+TOL_MM], 'g:','LineWidth',1.4,'DisplayName',sprintf('+%.1f mm 容忍線',TOL_MM));
    plot([lo2 hi2],[lo2-TOL_MM hi2-TOL_MM], 'g:','LineWidth',1.4,'DisplayName',sprintf('-%.1f mm 容忍線',TOL_MM));
    xlabel('實際長度 (mm)');
    ylabel(sprintf('校正後預測長度 = 模型輸出 - %.2f mm', offsetMM));
    title({'實際 vs 校正後預測（報告用）', ttl});
    legend('Location','southeast'); grid on; axis([lo2 hi2 lo2 hi2]);
    t3 = sprintf(['【校正後 OOF 指標（報告用）】\n模型 : %s\n量測法 : %s\n輸入 : %s\n' ...
                  'MAE : %.3f ± %.3f mm\nRMSE : %.3f mm\nMean Bias : %.3f mm\n' ...
                  'R : %.3f\nR^2 : %.3f\n理想比率 : %.1f%%\n(n = %d，%s 次重複平均)'], ...
                  MODEL, METHOD, featDesc, mae_off, sd_off, rmse_off, bias_off, ...
                  r_val, r2_val, ideal_val, n, nRep);
    placeText(t3, 0.28, txtBG, txtFG, txtEdge);
    hold off;
    savePNG(f3, SAVE_PNG, outDir, '3_散布圖_校正後');

    %% ---------------- 圖4：誤差 vs 年齡 / 性別 ----------------
    res = Y_pred - Y;
    if hasDemo
        f4 = figure('Name',['誤差分析 - ' MODEL],'NumberTitle','off','Position',[140 80 1000 460]);
        tiledlayout(1, 2, 'TileSpacing','compact');

        % --- 年齡 ---
        nexttile; hold on;
        okA = ~isnan(age);
        if sum(okA) >= 3 && range(age(okA)) > 0
            if any(~isnan(sex))
                isM = sex == 1;
                scatter(age(okA & ~isM), res(okA & ~isM), 42, dotC, 'filled','o','DisplayName','F');
                scatter(age(okA &  isM), res(okA &  isM), 46, [0.2 0.75 0.75],'filled','s','DisplayName','M');
            else
                scatter(age(okA), res(okA), 42, dotC, 'filled', 'DisplayName','OOF 誤差');
            end
            pp = polyfit(age(okA), res(okA), 1);
            xa = [min(age(okA)) max(age(okA))];
            plot(xa, polyval(pp, xa), '-', 'Color',[0.85 0.65 0.13], 'LineWidth',1.8, ...
                'DisplayName', sprintf('趨勢 %.3f mm/歲', pp(1)));
            yline(0, [lineC '-'], 'HandleVisibility','off');
            rA = NaN; pA = NaN;
            try, [rA, pA] = corr(age(okA), res(okA), 'Type','Spearman'); catch, end
            title(sprintf('誤差 vs 年齡 (Spearman \\rho=%.2f, p=%.3f)', rA, pA));
            legend('Location','best');
        else
            text(0.5, 0.5, '年齡缺值或為常數，無法分析', 'HorizontalAlignment','center');
            title('誤差 vs 年齡');
        end
        xlabel('年齡 (歲)'); ylabel('誤差 = 預測 - 實際 (mm)');
        grid on; hold off;

        % --- 性別 ---
        nexttile;
        okS = ~isnan(sex);
        if sum(okS) >= 2 && numel(unique(sex(okS))) == 2
            isM = sex == 1;
            gx = double(isM(okS)) + 1;          % F=1, M=2
            rs = res(okS);
            boxchart(gx, rs, 'MarkerStyle','none', 'BoxFaceColor',[0.35 0.55 0.80]);
            hold on;
            rng(0); jit = (rand(numel(rs),1) - 0.5) * 0.25;
            scatter(gx + jit, rs, 26, [0.35 0.35 0.35], 'filled', 'MarkerFaceAlpha',0.6);
            yline(0, [lineC '-']);
            hold off;
            xticks([1 2]); xticklabels({'F','M'}); xlim([0.5 2.5]);
            pS = NaN;
            try, pS = ranksum(res(okS & isM), res(okS & ~isM)); catch, end
            maeF = mean(abs(res(okS & ~isM))); maeM = mean(abs(res(okS & isM)));
            title(sprintf('誤差依性別 (MAE F=%.2f / M=%.2f, 秩和 p=%.3f)', maeF, maeM, pS));
            fprintf('\n=== 誤差分析 ===\n');
            fprintf('  性別：F n=%d MAE=%.3f | M n=%d MAE=%.3f | 秩和 p=%.3f\n', ...
                sum(okS & ~isM), maeF, sum(okS & isM), maeM, pS);
        else
            text(0.5, 0.5, '性別缺值或只有單一組別', 'HorizontalAlignment','center');
            title('誤差依性別');
        end
        ylabel('誤差 = 預測 - 實際 (mm)'); grid on;

        sgtitle(sprintf('誤差分析（%s）', ttl));
        savePNG(f4, SAVE_PNG, outDir, '4_誤差分析_年齡性別');
    end

    %% ---------------- 終端機摘要 ----------------
    fprintf('\n=== %s 校正後 OOF 指標（報告用）===\n', MODEL);
    fprintf('  MAE       : %.3f ± %.3f mm\n', mae_off, sd_off);
    fprintf('  RMSE      : %.3f mm   （由代表性重複重算）\n', rmse_off);
    fprintf('  Mean Bias : %.3f mm   （正 = 整體偏長）\n', bias_off);
    fprintf('  R / R^2   : %.3f / %.3f   （offset 不影響）\n', r_val, r2_val);
    fprintf('  理想比率  : %.1f %%  （±%.1f mm 內）\n', ideal_val, TOL_MM);
    fprintf('  n = %d，%s 次重複平均；圖上的點是最接近中位數 MAE 的那次重複。\n', n, nRep);
    if SAVE_PNG, fprintf('\n圖已存到：%s\n', outDir); end
    fprintf('報告時務必附上 n 與標準差。\n');
end


%% ============================================================
function savePNG(fh, doSave, outDir, name)
    if ~doSave, return; end
    exportgraphics(fh, fullfile(outDir, [name '.png']), 'Resolution', 200);
end

function placeText(txt, yRatio, bg, fg, ec)
    xl = xlim; yl = ylim;
    text(xl(1) + 0.05*(xl(2)-xl(1)), yl(2) - yRatio*(yl(2)-yl(1)), txt, ...
        'FontSize', 10, 'BackgroundColor', bg, 'EdgeColor', ec, 'Color', fg);
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
