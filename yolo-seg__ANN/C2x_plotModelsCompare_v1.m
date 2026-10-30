function C2x_plotModelCompare_v1()
%% C2x_plotModelCompare_v1.m
% ============================================================
% 搭配 C1x_modelCompare_v1.m，畫「模型對照」的圖。
% 原本的 C2 讀的是 C1 v6 的 sheet 結構（所有牙齒預測結果 / 模型評估指標），
% 跟 C1x 的輸出不同，所以不能直接用；這支是對照版。
%
% 產出 5 張圖 + 1 份配對檢定：
%   圖1 校正後 MAE 排行（含重複間 SD，天真基準畫成參考線）
%   圖2 ΔMAE vs 基準模型（逐重複配對）
%   圖3 各模型 預測 vs 實際 散布小圖陣列（含 ±1mm 容忍帶）
%   圖4 各模型 校正後殘差 boxplot
%   圖5 理想比率（±1mm 內）排行
%   表  最佳模型 vs 其他模型的 Wilcoxon 配對檢定 → 差距是不是真的
%
% 另有 EXPORT_FOR_C2：把其中一個模型輸出成舊 C2 吃得下的格式。
% ============================================================

    %% ---------------- 設定（要跟 C1x 那次一致）----------------
    METHOD   = 'mask幾何_長邊正規化';
    USE_AGE  = true;
    USE_SEX  = false;

    SAVE_PNG = true;           % 存 PNG 到 圖_模型對照_<METHOD><FEAT_TAG>/
    TOL_MM   = 1.0;            % 理想容忍，畫容忍帶用
    EXPORT_FOR_C2 = '';        % 填模型代號（如 'gpr'）會另外輸出舊 C2 可讀的檔；'' = 不輸出

    FEAT_TAG = featTag(USE_AGE, USE_SEX);
    inFile   = sprintf('模型對照_kfold_%s%s.xlsx', METHOD, FEAT_TAG);
    if ~isfile(inFile)
        error(['找不到 %s\n' ...
               '   請先用同樣的 METHOD / USE_AGE / USE_SEX 跑一次 C1x_modelCompare_v1。'], inFile);
    end
    fprintf('讀取 %s ...\n', inFile);

    %% ---------------- 讀檔 ----------------
    Cmp = readtable(inFile, 'Sheet', '模型對照總表', ...
                    'VariableNamingRule','preserve', 'TextType','string');
    Per = readtable(inFile, 'Sheet', '逐筆預測', 'VariableNamingRule','preserve');
    Set = readtable(inFile, 'Sheet', '設定', ...
                    'VariableNamingRule','preserve', 'TextType','string');

    baseModel = getSetting(Set, '基準模型', 'constmean');
    offsetMM  = str2double(getSetting(Set, 'offset_mm', '0.5'));
    featDesc  = getSetting(Set, '輸入特徵', '像素長度');
    nSamp     = height(Per);

    dCol = ['ΔMAE_vs_' char(baseModel)];
    if ~ismember(dCol, Cmp.Properties.VariableNames)
        hit  = startsWith(Cmp.Properties.VariableNames, 'ΔMAE_vs_');
        dCol = Cmp.Properties.VariableNames{find(hit,1)};
    end

    models = string(Cmp.('模型'));          % 已依校正後 MAE 排序
    nM     = numel(models);
    mae    = Cmp.('校正後MAE_mm');
    maeSD  = Cmp.('校正後MAE_SD');
    dmae   = Cmp.(dCol);
    dsd    = Cmp.('ΔMAE_SD');
    ideal  = Cmp.('理想比率_pct');

    Y = Per.('實際長度_mm');
    predCols = containers.Map('KeyType','char','ValueType','any');
    for i = 1:nM
        c = ['預測_' char(models(i))];
        if ismember(c, Per.Properties.VariableNames)
            predCols(char(models(i))) = Per.(c);
        end
    end

    outDir = sprintf('圖_模型對照_%s%s', METHOD, FEAT_TAG);
    if SAVE_PNG && ~isfolder(outDir), mkdir(outDir); end

    ttl = sprintf('%s｜輸入：%s｜n=%d｜offset=-%.2fmm', METHOD, featDesc, nSamp, offsetMM);

    %% ---------------- 圖1：MAE 排行 ----------------
    f1 = figure('Name','模型對照：校正後 MAE','NumberTitle','off','Position',[80 80 760 480]);
    [maeS, ord] = sort(mae, 'descend');          % barh 由下往上，descend 讓最好的在最上面
    mdlS = models(ord); sdS = maeSD(ord);
    b = barh(categorical(mdlS, mdlS), maeS, 0.6, 'FaceColor',[0.35 0.55 0.80]);
    hold on;
    errorbar(maeS, 1:nM, sdS, 'horizontal', 'k.', 'LineWidth', 1, 'CapSize', 8);
    bi = find(mdlS == baseModel);
    if ~isempty(bi)
        b.FaceColor = 'flat';
        b.CData(bi,:) = [0.75 0.35 0.35];
        xline(maeS(bi), '--', sprintf('%s = %.3f', baseModel, maeS(bi)), ...
            'Color',[0.75 0.35 0.35], 'LineWidth',1.4, 'LabelVerticalAlignment','bottom');
    end
    for i = 1:nM
        text(maeS(i) + 0.012*max(maeS), i, sprintf('%.3f', maeS(i)), ...
            'VerticalAlignment','middle', 'FontSize', 9);
    end
    xlabel('校正後 MAE (mm)，越小越好');
    title({'模型對照：校正後 MAE', ttl});
    grid on; hold off;
    savePNG(f1, SAVE_PNG, outDir, '1_MAE排行');

    %% ---------------- 圖2：ΔMAE vs 基準 ----------------
    f2 = figure('Name','模型對照：ΔMAE','NumberTitle','off','Position',[100 100 760 480]);
    [dS, ord2] = sort(dmae, 'ascend');
    mdlS2 = models(ord2); dsdS = dsd(ord2);
    cols = repmat([0.35 0.65 0.45], nM, 1);
    cols(dS > 0, :) = repmat([0.80 0.45 0.35], sum(dS>0), 1);
    bb = barh(categorical(mdlS2, flip(mdlS2)), dS, 0.6, 'FaceColor','flat');
    bb.CData = flip(cols,1);
    hold on;
    yv = double(categorical(mdlS2, flip(mdlS2)));
    errorbar(dS, yv, dsdS, 'horizontal', 'k.', 'LineWidth', 1, 'CapSize', 8);
    xline(0, 'k-', 'LineWidth', 1.2);
    xlabel(sprintf('ΔMAE = 本模型 − %s (mm)，負值代表比基準好', baseModel));
    title({sprintf('與天真基準 %s 的配對差（逐重複相減）', baseModel), ttl});
    grid on; hold off;
    savePNG(f2, SAVE_PNG, outDir, '2_ΔMAE對基準');

    %% ---------------- 圖3：預測 vs 實際 小圖陣列 ----------------
    ks = predCols.keys;
    if ~isempty(ks)
        f3 = figure('Name','模型對照：預測 vs 實際','NumberTitle','off','Position',[120 60 1100 760]);
        nc = ceil(sqrt(nM)); nr = ceil(nM/nc);
        tl = tiledlayout(nr, nc, 'TileSpacing','compact', 'Padding','compact');
        lo = floor(min(Y)) - 1; hi = ceil(max(Y)) + 1;
        for i = 1:nM
            m = char(models(i));
            nexttile;
            if ~isKey(predCols, m), axis off; title(m); continue; end
            yp = predCols(m);
            hold on;
            fill([lo hi hi lo], [lo-TOL_MM hi-TOL_MM hi+TOL_MM lo+TOL_MM], ...
                [0.5 0.8 0.5], 'FaceAlpha', 0.15, 'EdgeColor','none');
            plot([lo hi], [lo hi], 'k-', 'LineWidth', 1.2);
            scatter(Y, yp, 22, [0.20 0.40 0.70], 'filled', 'MarkerFaceAlpha', 0.7);
            xlim([lo hi]); ylim([lo hi]); axis square; grid on;
            title(sprintf('%s  (MAE %.3f)', m, mae(i)), 'FontWeight','normal');
            hold off;
        end
        xlabel(tl, '實際長度 (mm)'); ylabel(tl, '校正後預測長度 (mm)');
        title(tl, {sprintf('各模型 預測 vs 實際（綠帶 = ±%.1f mm 容忍）', TOL_MM), ttl});
        savePNG(f3, SAVE_PNG, outDir, '3_預測vs實際');
    end

    %% ---------------- 圖4：殘差 boxplot ----------------
    if ~isempty(ks)
        f4 = figure('Name','模型對照：殘差分布','NumberTitle','off','Position',[140 80 900 500]);
        hold on;
        allRes = []; allGrp = [];
        for i = 1:nM
            m = char(models(i));
            if ~isKey(predCols, m), continue; end
            res = predCols(m) - Y;
            allRes = [allRes; res];                      %#ok<AGROW>
            allGrp = [allGrp; repmat(i, numel(res), 1)]; %#ok<AGROW>
        end
        boxchart(allGrp, allRes, 'MarkerStyle','none', 'BoxFaceColor',[0.35 0.55 0.80]);
        rng(0);
        jit = (rand(numel(allRes),1) - 0.5) * 0.3;
        scatter(allGrp + jit, allRes, 14, [0.3 0.3 0.3], 'filled', 'MarkerFaceAlpha', 0.35);
        yline(0, 'k-', 'LineWidth', 1.2);
        yline( TOL_MM, 'g:', 'LineWidth', 1.4);
        yline(-TOL_MM, 'g:', 'LineWidth', 1.4);
        xticks(1:nM); xticklabels(models); xlim([0.5 nM+0.5]);
        ylabel('校正後殘差 = 預測 − 實際 (mm)');
        title({'各模型殘差分布（綠虛線 = ±1 mm）', ttl});
        grid on; hold off;
        savePNG(f4, SAVE_PNG, outDir, '4_殘差分布');
    end

    %% ---------------- 圖5：理想比率 ----------------
    f5 = figure('Name','模型對照：理想比率','NumberTitle','off','Position',[160 100 760 480]);
    [idS, ord5] = sort(ideal, 'ascend');
    barh(categorical(models(ord5), models(ord5)), idS, 0.6, 'FaceColor',[0.45 0.70 0.55]);
    for i = 1:nM
        text(idS(i) + 1, i, sprintf('%.1f%%', idS(i)), 'VerticalAlignment','middle','FontSize',9);
    end
    xlabel(sprintf('理想比率 = 校正後誤差在 ±%.1f mm 內的比例 (%%)', TOL_MM));
    title({'各模型臨床理想比率', ttl});
    xlim([0 100]); grid on;
    savePNG(f5, SAVE_PNG, outDir, '5_理想比率');

    %% ---------------- 配對檢定：最佳模型 vs 其他 ----------------
    fprintf('\n=== 最佳模型 vs 其他（Wilcoxon 逐樣本配對，n=%d）===\n', nSamp);
    fprintf('問的是：兩個模型的「逐樣本絕對誤差」是否真的有差，不是看 MAE 差幾 mm。\n\n');
    best = char(models(1));
    if isKey(predCols, best)
        aeBest = abs(predCols(best) - Y);
        names = strings(0); dm = []; pv = [];
        for i = 2:nM
            m = char(models(i));
            if ~isKey(predCols, m), continue; end
            aeM = abs(predCols(m) - Y);
            names(end+1,1) = string(m);                     %#ok<AGROW>
            dm(end+1,1)    = mean(aeBest) - mean(aeM);      %#ok<AGROW>
            try
                pv(end+1,1) = signrank(aeBest, aeM);        %#ok<AGROW>
            catch
                pv(end+1,1) = NaN;                          %#ok<AGROW>
            end
        end
        verdict = strings(numel(pv),1);
        verdict(:) = "差距沒有統計證據";
        verdict(pv < 0.05) = "有差（p<0.05）";
        verdict(isnan(pv))  = "無法檢定（缺 Stats Toolbox）";
        T = table(names, round(dm,3), round(pv,4), verdict, ...
            'VariableNames', {'對手模型', ['MAE差_' best '減對手_mm'], 'p值', '判讀'});
        disp(T);
        fprintf('注意：p 值未校正多重比較，n=%d 偏小，只當參考。\n', nSamp);
    end

    %% ---------------- 可選：輸出舊 C2 吃得下的格式 ----------------
    if ~isempty(EXPORT_FOR_C2)
        m = EXPORT_FOR_C2;
        if ~isKey(predCols, m)
            warning('逐筆預測裡沒有模型 %s，略過 C2 相容輸出。', m);
        else
            ypOff = predCols(m);
            ypRaw = ypOff + offsetMM;
            R = table(Per.('檔名'), repmat("test", nSamp, 1), Per.('像素長度_px'), Y, ...
                ypRaw, ypRaw - Y, abs(ypRaw - Y), ypOff, ypOff - Y, abs(ypOff - Y), ...
                Per.('摺別'), 'VariableNames', ...
                {'檔名','Train_or_Test','像素長度_px','實際長度_mm','模型預測_mm', ...
                 '誤差Bias_mm','絕對誤差_mm','校正後預測_mm','校正後誤差Bias_mm', ...
                 '校正後絕對誤差_mm','摺別'});
            i0 = find(models == string(m), 1);
            MetricNames = {
                sprintf('A_Test set表現 (out-of-fold, n=%d, %s, 輸入=%s)', nSamp, m, featDesc);
                'B_重複間標準差';
                '--- 分界線 ---';
                sprintf('C_臨床安全性評估_Test set校正後 (offset=-%.2fmm)', offsetMM)};
            MT = table(MetricNames, ...
                round([Cmp.('未校正MAE_mm')(i0); maeSD(i0); NaN; mae(i0)],3), ...
                round([Cmp.('未校正RMSE_mm')(i0); NaN; NaN; Cmp.('未校正RMSE_mm')(i0)],3), ...
                round([Cmp.('R_Pearson')(i0); NaN; NaN; Cmp.('R_Pearson')(i0)],3), ...
                round([Cmp.('R_Square')(i0); NaN; NaN; Cmp.('R_Square')(i0)],3), ...
                round([NaN; NaN; NaN; Cmp.('校正後Bias_mm')(i0)],3), ...
                round([NaN; NaN; NaN; ideal(i0)],1), ...
                'VariableNames', {'評估範圍_與_嚴格程度','MAE_mm','RMSE_mm','R_Pearson', ...
                'R_Square','Mean_Bias_mm','理想比率_pct'});
            ST = table(["量測方法";"輸入特徵";"n";"模型"], ...
                       [string(METHOD); string(featDesc); string(nSamp); string(m)], ...
                       'VariableNames', {'項目','值'});
            c2File = sprintf('預測結果與評估指標_kfold_%s%s.xlsx', METHOD, FEAT_TAG);
            if isfile(c2File), delete(c2File); end
            writetable(R,  c2File, 'Sheet', '所有牙齒預測結果');
            writetable(MT, c2File, 'Sheet', '模型評估指標');
            writetable(ST, c2File, 'Sheet', '模型設定');
            fprintf('\n已輸出 C2 相容檔：%s（模型 = %s）\n', c2File, m);
            fprintf('   注意：C2 的檔名若含 _H<隱藏層> 後綴，要把那段拿掉或自行改名；\n');
            fprintf('   另外逐筆預測沒有年齡/性別欄，C2 的圖4 會自動略過。\n');
        end
    end

    if SAVE_PNG
        fprintf('\n圖已存到資料夾：%s\n', outDir);
    end
end


%% ============================================================
function savePNG(fh, doSave, outDir, name)
    if ~doSave, return; end
    exportgraphics(fh, fullfile(outDir, [name '.png']), 'Resolution', 200);
end

function v = getSetting(Set, key, defaultVal)
    v = defaultVal;
    hit = string(Set.('項目')) == string(key);
    if any(hit)
        v = char(string(Set.('值')(find(hit,1))));
    end
end

function tag = featTag(useAge, useSex)
    parts = {};
    if useAge, parts{end+1} = '年齡'; end
    if useSex, parts{end+1} = '性別'; end
    if isempty(parts), tag = ''; else, tag = ['_' strjoin(parts,'')]; end
end
