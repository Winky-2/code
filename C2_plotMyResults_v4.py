"""C2 畫 C1 結果  ── MATLAB C2_plotMyResults_v4.m(內部 v5)的 Python 版

前提：先用同樣的 METHOD / USE_AGE / USE_SEX 跑過 C1。
操作：改下方設定後直接執行(只讀不寫，RUN_NOW 預設 True)。
產出：4 張圖(排序點圖、未校正散布圖、校正後散布圖、誤差 vs 年齡/性別)+ 終端機摘要。
與 MATLAB 版差異：
  - 深色背景(dark_background)，保留原版白色參考線的可見度
  - Spearman p 用 scipy 的 t 近似；秩和檢定依 MATLAB ranksum 規則選 exact/asymptotic
    → 小樣本時 p 值可能與 MATLAB 第 3 位小數有差
  - 抖動點用 numpy 亂數，位置與 MATLAB rng(0) 不同(僅外觀)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from C_function import feat_tag, hidden_tag

# ==================== 可手動修改 ====================
RUN_NOW = True

METHOD = "mask幾何_長邊正規化"   # 'mask幾何' | '冠寬比例尺'
USE_AGE = False
USE_SEX = True
HIDDEN_SIZE = [3, 3]
SHOW_ERROR_BARS = False
# ====================================================


def result_filename() -> str:
    return (f"預測結果與評估指標_kfold_{METHOD}{feat_tag(USE_AGE, USE_SEX)}"
            f"_H{hidden_tag(HIDDEN_SIZE)}.xlsx")


# ---------------- 子函式 ----------------
def pick_row(tbl, labels, key):
    idx = labels.str.contains(key, regex=False, na=False)
    if idx.sum() != 1:
        raise ValueError(f"在「模型評估指標」裡找不到唯一的「{key}」那一列(找到 {idx.sum()} 列)。\n"
                         "   請確認 C1 的 MetricNames 沒有被改動。")
    return tbl.loc[idx].iloc[0]


def pick_row_optional(tbl, labels, key):
    idx = labels.str.contains(key, regex=False, na=False)
    return tbl.loc[idx].iloc[0] if idx.sum() == 1 else None


def get_num(row, col):
    """欄位不存在就回 NaN，舊版 C1 的輸出也讀得起來"""
    if row is None or col not in row.index:
        return np.nan
    v = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
    return float(v) if pd.notna(v) else np.nan


def line_if(fmt, v):
    return "" if np.isnan(v) else fmt.format(v)


def print_if(fmt, v):
    if not np.isnan(v):
        print(fmt.format(v), end="")


def sd_suffix(sd, unit):
    if np.isnan(sd):
        return ""
    return f" ± {sd:.1f}" if unit == "%" else f" ± {sd:.3f}"


def place_text(ax, txt, y_offset_ratio):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.text(x0 + 0.05 * (x1 - x0), y1 - y_offset_ratio * (y1 - y0), txt,
            fontsize=11, color="w", va="center",
            bbox=dict(facecolor="k", edgecolor="w"))


def ranksum_p(x, y):
    """MATLAB ranksum：min(nx,ny)<10 且 nx+ny<20 用 exact，否則常態近似(含連續校正)"""
    from scipy.stats import mannwhitneyu
    method = "exact" if (min(len(x), len(y)) < 10 and len(x) + len(y) < 20) else "asymptotic"
    return mannwhitneyu(x, y, alternative="two-sided", method=method).pvalue


def _setup_fonts(plt):
    plt.style.use("dark_background")
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC",
                                       "Noto Sans TC", "Heiti TC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _new_fig(plt, name, **kw):
    fig = plt.figure(**kw)
    try:
        fig.canvas.manager.set_window_title(name)
    except Exception:
        pass
    return fig


def main():
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    _setup_fonts(plt)

    filename = result_filename()
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"找不到 {filename}\n   請先用同樣的 METHOD / USE_AGE / USE_SEX 跑一次 C1。")
    print(f"正在從 {filename} 讀取預測資料...")

    # --- 2. 逐筆預測結果 ---
    res_all = pd.read_excel(filename, sheet_name="所有牙齒預測結果")
    test = res_all[res_all["Train_or_Test"].astype(str) == "test"].reset_index(drop=True)
    Y = test["實際長度_mm"].to_numpy(float)
    Y_pred = test["模型預測_mm"].to_numpy(float)
    Y_pred_offset = test["校正後預測_mm"].to_numpy(float)
    n = Y.size

    has_demo = "年齡" in test.columns and "性別" in test.columns
    if has_demo:
        age = pd.to_numeric(test["年齡"], errors="coerce").to_numpy(float)
        sex_lbl = test["性別"].astype(str).str.strip().str.upper().to_numpy()
        is_m = sex_lbl == "M"
    print(f"圖表使用 {n} 筆 out-of-fold 預測。")

    # --- 3. 評估指標與模型設定 ---
    xls = pd.ExcelFile(filename)
    met = pd.read_excel(xls, sheet_name="模型評估指標")
    labels = met["評估範圍_與_嚴格程度"].astype(str)

    test_row = pick_row(met, labels, "A_Test set表現")
    clin_row = pick_row(met, labels, "C_臨床安全性評估_Test set校正後")
    sd_row = pick_row_optional(met, labels, "B_重複間標準差")
    lin_row = pick_row_optional(met, labels, "E_線性迴歸對照組_校正後")
    if lin_row is None:
        lin_row = pick_row_optional(met, labels, "D_線性迴歸對照組")

    test_mae, test_rmse = get_num(test_row, "MAE_mm"), get_num(test_row, "RMSE_mm")
    test_r, test_r2 = get_num(test_row, "R_Pearson"), get_num(test_row, "R_Square")
    test_bias = get_num(test_row, "Mean_Bias_mm")

    c_mae, c_rmse = get_num(clin_row, "MAE_mm"), get_num(clin_row, "RMSE_mm")
    c_bias = get_num(clin_row, "Mean_Bias_mm")
    c_r, c_r2 = get_num(clin_row, "R_Pearson"), get_num(clin_row, "R_Square")
    c_ideal = get_num(clin_row, "理想比率_pct")

    sd_mae = get_num(sd_row, "MAE_mm")
    sd_ideal = get_num(sd_row, "理想比率_pct")
    lin_mae = get_num(lin_row, "MAE_mm")

    feat_desc = "像素長度"
    if "模型設定" in xls.sheet_names:
        st = pd.read_excel(xls, sheet_name="模型設定", dtype=str)
        hit = st["項目"] == "輸入特徵"
        if hit.any():
            feat_desc = str(st.loc[hit, "值"].iloc[0])

    print("資料讀取完畢！正在繪製圖表...")
    ttl = f"{METHOD}，Out-of-Fold"

    # --- 4. 圖 1：排序點圖 ---
    _new_fig(plt, f"Sorted: Actual vs Predicted (OOF) - {METHOD}")
    ax = plt.gca()
    order = np.argsort(Y, kind="stable")
    Ys, Yps = Y[order], Y_pred[order]
    x = np.arange(1, n + 1)
    if SHOW_ERROR_BARS:
        ax.vlines(x, Ys, Yps, colors=(0.7, 0.7, 0.7), linewidth=0.8)
    ax.scatter(x, Ys, s=28, c="r", marker="o", label="實際長度 (Actual)")
    ax.scatter(x, Yps, s=28, c="b", marker="s", label="Out-of-fold 預測長度")
    ax.set_xlabel(f"樣本排序編號 (依實際長度由小到大，n={n})")
    ax.set_ylabel("長度 (mm)")
    ax.set_title(f"實際長度 vs 預測長度 排序點圖 ({ttl})\n輸入：{feat_desc}")
    ax.legend(loc="upper left")
    ax.grid(True)

    # --- 5. 圖 2：未校正散布圖 ---
    _new_fig(plt, f"OOF: Actual vs Predicted - {METHOD}")
    ax = plt.gca()
    if has_demo:
        ax.scatter(Y[~is_m], Y_pred[~is_m], s=40, c="b", marker="o", label="OOF 預測 (F)")
        ax.scatter(Y[is_m], Y_pred[is_m], s=45, c="c", marker="s", label="OOF 預測 (M)")
    else:
        ax.scatter(Y, Y_pred, s=40, c="b", label="Out-of-fold 預測")
    lo = np.floor(min(Y.min(), Y_pred.min())) - 1
    hi = np.ceil(max(Y.max(), Y_pred.max())) + 1
    L = np.array([lo, hi])
    ax.plot(L, L, "w-", lw=2, label="完美預測線 (誤差 0)")
    ax.plot(L, L + 0.5, "r--", lw=1.5, label="+0.5 mm 誤差線")
    ax.plot(L, L - 0.5, "r--", lw=1.5, label="-0.5 mm 誤差線")
    ax.plot(L, L + 1.0, "g:", lw=1.5, label="+1.0 mm 誤差線")
    ax.plot(L, L - 1.0, "g:", lw=1.5, label="-1.0 mm 誤差線")
    ax.set_xlabel("實際長度 (mm)"); ax.set_ylabel("預測長度 (mm)")
    ax.set_title(f"實際長度 vs 預測長度 ({ttl}，未校正)")
    ax.legend(loc="lower right"); ax.grid(True)
    txt = (f"【未校正 OOF 指標】\n量測法 : {METHOD}\n輸入 : {feat_desc}\n"
           f"MAE : {test_mae:.3f} mm{sd_suffix(sd_mae, 'mm')}\nRMSE : {test_rmse:.3f} mm\n"
           f"Mean Bias : {test_bias:.3f} mm\n{line_if('R : {:.3f}' + chr(10), test_r)}"
           f"R^2 : {test_r2:.3f}\n(n = {n})")
    place_text(ax, txt, 0.18)

    # --- 6. 圖 3：校正後(報告用) ---
    _new_fig(plt, f"Offset-corrected (OOF) - {METHOD}")
    ax = plt.gca()
    ax.scatter(Y, Y_pred_offset, s=45, c="b", label="校正後預測")
    lo = np.floor(min(Y.min(), Y_pred_offset.min())) - 1
    hi = np.ceil(max(Y.max(), Y_pred_offset.max())) + 1
    L = np.array([lo, hi])
    ax.plot(L, L, "w-", lw=2, label="完美預測線 (誤差 0)")
    ax.plot(L, L + 1.0, "g:", lw=1.5, label="+1.0 mm 容忍線")
    ax.plot(L, L - 1.0, "g:", lw=1.5, label="-1.0 mm 容忍線")
    ax.set_xlabel("實際長度 (mm)")
    ax.set_ylabel("校正後預測長度 = 迴歸輸出 - offset (mm)")
    ax.set_title(f"校正後預測長度 vs 實際長度 ({ttl})")
    ax.legend(loc="lower right"); ax.grid(True)
    txt = (f"【校正後 OOF 指標 (報告用)】\n量測法 : {METHOD}\n輸入 : {feat_desc}\n"
           f"MAE : {c_mae:.3f} mm{sd_suffix(sd_mae, 'mm')}\nRMSE : {c_rmse:.3f} mm\n"
           f"Mean Bias : {c_bias:.3f} mm\n{line_if('R : {:.3f}' + chr(10), c_r)}R^2 : {c_r2:.3f}\n"
           f"理想比率 : {c_ideal:.1f}%{sd_suffix(sd_ideal, '%')}\n(n = {n})")
    place_text(ax, txt, 0.26)

    # --- 7. 圖 4：誤差 vs 年齡 / 性別 ---
    if has_demo:
        res = Y_pred - Y
        fig = _new_fig(plt, f"Error Analysis: Age & Sex (OOF) - {METHOD}",
                       figsize=(12, 5), layout="constrained")
        ax1, ax2 = fig.subplots(1, 2)

        ax1.scatter(age[~is_m], res[~is_m], s=40, c="b", marker="o", label="F")
        ax1.scatter(age[is_m], res[is_m], s=45, c="c", marker="s", label="M")
        pp = np.polyfit(age, res, 1)
        xa = np.array([age.min(), age.max()])
        ax1.plot(xa, np.polyval(pp, xa), "y-", lw=1.8, label=f"趨勢 {pp[0]:.3f} mm/歲")
        ax1.axhline(0, color="w")
        r_age, p_age = spearmanr(age, res)
        ax1.set_xlabel("年齡 (歲)"); ax1.set_ylabel("誤差 = 預測 - 實際 (mm)")
        ax1.set_title(f"誤差 vs 年齡 (Spearman ρ={r_age:.2f}, p={p_age:.3f})")
        ax1.legend(loc="best"); ax1.grid(True)

        gx = np.where(is_m, 2, 1)
        groups = [res[~is_m], res[is_m]]
        ax2.boxplot([g for g in groups], positions=[1, 2], widths=0.5, showfliers=False,
                    boxprops=dict(color="w"), whiskerprops=dict(color="w"),
                    capprops=dict(color="w"), medianprops=dict(color="tab:orange"))
        jit = (np.random.default_rng(0).random(n) - 0.5) * 0.25
        ax2.scatter(gx + jit, res, s=25, c="w", alpha=0.6)
        ax2.axhline(0, color="w")
        ax2.set_xticks([1, 2]); ax2.set_xticklabels(["F", "M"]); ax2.set_xlim(0.5, 2.5)
        p_sex = ranksum_p(res[is_m], res[~is_m]) if (is_m.any() and (~is_m).any()) else np.nan
        mae_f, mae_m = np.mean(np.abs(res[~is_m])), np.mean(np.abs(res[is_m]))
        ax2.set_ylabel("誤差 = 預測 - 實際 (mm)")
        ax2.set_title(f"誤差依性別 (MAE F={mae_f:.2f} / M={mae_m:.2f}, 秩和 p={p_sex:.3f})")
        ax2.grid(True)

        fig.suptitle(f"誤差分析 ({ttl}，輸入：{feat_desc}，n={n})")

        print("\n=== 誤差分析 ===")
        print(f"  誤差 vs 年齡：Spearman ρ={r_age:.3f}, p={p_age:.3f}, 斜率 {pp[0]:.4f} mm/歲")
        print(f"  性別：F n={np.sum(~is_m)} MAE={mae_f:.3f} bias={np.mean(res[~is_m]):.3f} | "
              f"M n={np.sum(is_m)} MAE={mae_m:.3f} bias={np.mean(res[is_m]):.3f} | 秩和 p={p_sex:.3f}")
        print("  ⚠️ n 小、單次檢定，p 值只當參考。")
    else:
        print("ℹ️ 輸出檔沒有年齡/性別欄，略過圖表 4。")

    # --- 8. 終端機摘要(校正後) ---
    print("\n=== 校正後 OOF 指標(報告用) ===")
    print(f"  MAE       : {c_mae:.3f} mm{sd_suffix(sd_mae, 'mm')}")
    print(f"  RMSE      : {c_rmse:.3f} mm")
    print(f"  Mean Bias : {c_bias:.3f} mm  (正 = 整體偏長)")
    print_if("  R         : {:.3f}  (offset 不影響，與未校正同值)\n", c_r)
    print(f"  R^2       : {c_r2:.3f}  (同上)")
    print(f"  理想比率  : {c_ideal:.1f}%{sd_suffix(sd_ideal, '%')}")

    print("\n✅ 圖表繪製完成。")
    print(f"⚠️ 報告時務必附上 n={n} 與標準差。")
    if not np.isnan(lin_mae) and lin_mae <= c_mae:
        print(f"ℹ️ 線性對照組 MAE({lin_mae:.3f}) 不輸 ANN({c_mae:.3f})，解讀時請一併說明。")
    plt.show()


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)
        print(f"待讀取：{result_filename()}")
