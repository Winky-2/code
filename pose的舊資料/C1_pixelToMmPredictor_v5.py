"""C1 像素長度 → mm 預測(k-fold ANN + 線性對照)  ── MATLAB C1_pixelToMmPredictor_v5.m(內部 v6)的 Python 版

前提：先跑 B4_build_C1_input.py，產生「根管填充物像素長度_已配對_<METHOD>.xlsx」。
操作：
1. 改下方「可手動修改」區(METHOD / USE_AGE / USE_SEX ...)。
2. 確認要正式跑時，把 RUN_NOW 改成 True 再執行。
產出：預測結果與評估指標_kfold_<METHOD><特徵標籤>.xlsx
      sheet：所有牙齒預測結果 / 模型評估指標 / 模型設定 / 線性係數_全資料
與 MATLAB 版差異：
  - fitnet+trainbr 改用 C_function.FitnetBR(numpy 實作)，亂數流不同 → 逐筆預測不會位元相同
  - 線性係數表改用 statsmodels OLS(數值應與 fitlm 相同)
  - 模型設定表「腳本版本」寫 C1_v6_py，方便分辨輸出來源
數學提醒：offset 只是平移 → R、R^2 校正前後相同；MAE、RMSE、Bias 會變。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from C_function import FitnetBR, feat_tag, mround, parse_sex, to_num

# ==================== 可手動修改 ====================
RUN_NOW = False

# ---- 輸入 ----
METHOD = "冠寬比例尺"            # 'mask幾何_長邊正規化' | '冠寬比例尺'
FOLD_COLUMN = "折數fold"
SPLIT_COLUMN = "資料夾來源"      # 只在 USE_KFOLD=False 時才用

# ---- 特徵 ----
USE_AGE = False
USE_SEX = True
AGE_COLUMN = "年齡"
SEX_COLUMN = "性別"

# ---- 評估 ----
USE_KFOLD = True
N_REPEATS = 10
BASE_SEED = 42
HIDDEN_SIZE = 5
TRAIN_FCN = "trainbr"            # 只實作 trainbr
MAX_EPOCHS = 200
RUN_LINEAR_BASELINE = True

# ---- 臨床 ----
OFFSET_MM = 0.5
IDEAL_TOLERANCE_MM = 1.0
# ====================================================


def input_filename() -> str:
    return f"根管填充物像素長度_已配對_{METHOD}.xlsx"


def output_filename() -> str:
    return f"預測結果與評估指標_kfold_{METHOD}{feat_tag(USE_AGE, USE_SEX)}.xlsx"


# ---------------- 指標 ----------------
_KEYS = ["mae", "rmse", "r", "r2", "bias", "offMae", "offRmse", "offBias", "ideal"]


def empty_metrics(N):
    return {k: np.full(N, np.nan) for k in _KEYS}


def accum_metrics(S, r, Yp, Y, offset, tol):
    ok = ~np.isnan(Yp) & ~np.isnan(Y)
    Yp, Y = Yp[ok], Y[ok]
    res = Yp - Y
    S["mae"][r] = np.mean(np.abs(res))
    S["rmse"][r] = np.sqrt(np.mean(res ** 2))
    S["bias"][r] = np.mean(res)
    if Y.size >= 2:
        S["r"][r] = np.corrcoef(Y, Yp)[0, 1]       # Pearson R(含正負)
        S["r2"][r] = S["r"][r] ** 2
    off = (Yp - offset) - Y
    S["offMae"][r] = np.mean(np.abs(off))
    S["offRmse"][r] = np.sqrt(np.mean(off ** 2))
    S["offBias"][r] = np.mean(off)
    S["ideal"][r] = np.mean(np.abs(off) <= tol) * 100


def sd(x):
    """MATLAB std(x)：ddof=1"""
    return np.std(x, ddof=1)


def mean_or(x, flag):
    return np.nanmean(x) if flag else np.nan


def main():
    filename = input_filename()

    # --- 1. 讀取資料 ---
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"找不到 {filename}\n   這份檔案由 B4_build_C1_input.py 產生，請先跑那支腳本。")
    data = pd.read_excel(filename)
    vn = list(data.columns)

    ids = data["圖片檔名"].to_numpy()
    if not (pd.api.types.is_numeric_dtype(data["填充物長度(mm)"])
            and pd.api.types.is_numeric_dtype(data["像素長度"])):
        raise ValueError("實際長度或像素長度欄位含非數值資料，請檢查 Excel。")
    Y = data["填充物長度(mm)"].to_numpy(dtype=float)
    X = data["像素長度"].to_numpy(dtype=float)

    has_demo = AGE_COLUMN in vn and SEX_COLUMN in vn
    if (USE_AGE or USE_SEX) and not has_demo:
        raise ValueError(
            f"輸入檔沒有「{AGE_COLUMN}」「{SEX_COLUMN}」欄。\n"
            "   請在 B4 的 MM_EXTRA_COLS 加上這兩欄、MM_XLSX 指向新版醫師表，重跑 B4。")

    if has_demo:
        age = to_num(data[AGE_COLUMN])
        sex = parse_sex(data[SEX_COLUMN])
        if np.all(np.isnan(sex)):
            raise ValueError(f"「{SEX_COLUMN}」欄全部無法解析(只認 M/F/男/女)，請檢查 Excel。")
    else:
        age = np.full(len(data), np.nan)
        sex = np.full(len(data), np.nan)

    valid = ~np.isnan(X) & ~np.isnan(Y)
    if has_demo:
        valid &= ~np.isnan(age) & ~np.isnan(sex)
    if np.any(~valid):
        print(f"⚠️ 有 {np.sum(~valid)} 筆含缺值(長度/年齡/性別)，已剔除。")
        ids, Y, X, age, sex = ids[valid], Y[valid], X[valid], age[valid], sex[valid]
        data = data.loc[valid].reset_index(drop=True)
    n = Y.size

    # --- 組特徵矩陣 ---
    cols, feat_names = [X], ["像素長度"]
    if USE_AGE:
        cols.append(age); feat_names.append("年齡")
    if USE_SEX:
        cols.append(sex); feat_names.append("性別(M=1)")
    Xmat = np.column_stack(cols)
    p = Xmat.shape[1]
    feat_desc = " + ".join(feat_names)
    FEAT_TAG = feat_tag(USE_AGE, USE_SEX)
    n_param = HIDDEN_SIZE * p + HIDDEN_SIZE + HIDDEN_SIZE + 1

    print(f"=== C1 v6 (Python)：{METHOD} ===")
    print(f"載入 {n} 筆樣本。輸入特徵：{feat_desc}")
    print(f"ANN 參數數 {n_param}，約 {n / n_param:.1f} 筆 / 參數")
    if has_demo:
        print(f"性別：M {np.sum(sex == 1)} / F {np.sum(sex == 0)}；"
              f"年齡 {np.min(age):g}–{np.max(age):g} 歲(中位數 {np.median(age):g})")

    # --- 2. 決定每一筆屬於哪一摺 ---
    if USE_KFOLD:
        if FOLD_COLUMN in vn:
            fold_id = data[FOLD_COLUMN].to_numpy(dtype=float).astype(int)
            print(f"沿用 Excel 的「{FOLD_COLUMN}」欄分摺(三種方法共用同一組切分)。")
        else:
            K = 5
            order = np.argsort(Y, kind="stable")
            fold_id = np.zeros(n, dtype=int)
            fold_id[order] = np.arange(n) % K + 1
            print(f"⚠️ Excel 沒有「{FOLD_COLUMN}」欄，改用依實際長度分層的 {K} 摺。")
            print("   注意：這樣三種方法的切分不保證一致，對照結果會混入切分差異。")
        K = int(fold_id.max())
        counts = np.bincount(fold_id, minlength=K + 1)[1:]
        print(f"共 {K} 摺，每摺樣本數：{' '.join(map(str, counts))}")
    else:
        if SPLIT_COLUMN not in vn:
            raise ValueError(f"USE_KFOLD=False 需要「{SPLIT_COLUMN}」欄。")
        is_test = data[SPLIT_COLUMN].astype(str).to_numpy() == "test"
        fold_id = np.ones(n, dtype=int)
        fold_id[~is_test] = 0
        K = 1
        print(f"單次切分模式：train {np.sum(~is_test)} 筆 / test {np.sum(is_test)} 筆")

    # --- 3. 交叉驗證主迴圈 ---
    ann_pred = np.full((n, N_REPEATS), np.nan)
    lin_pred = np.full((n, N_REPEATS), np.nan)
    fold_mae = np.full((K, N_REPEATS), np.nan)

    print(f"\n訓練中({K} 摺 × {N_REPEATS} 次重複)...")
    for r in range(N_REPEATS):
        rng = np.random.default_rng(BASE_SEED + r + 1)     # 對應 MATLAB rng(BASE_SEED + r), r 從 1 起
        for k in range(1, K + 1):
            te = fold_id == k
            tr = ~te & (fold_id > 0)
            if not te.any() or not tr.any():
                continue
            Xtr, Ytr, Xte = Xmat[tr], Y[tr], Xmat[te]

            mu = Xtr.mean(0)
            sg = Xtr.std(0, ddof=1)
            sg[sg == 0] = 1
            net = FitnetBR(HIDDEN_SIZE, epochs=MAX_EPOCHS, rng=rng).fit((Xtr - mu) / sg, Ytr)
            ann_pred[te, r] = net.predict((Xte - mu) / sg)

            if RUN_LINEAR_BASELINE:
                A = np.column_stack([np.ones(len(Xtr)), Xtr])
                b = np.linalg.lstsq(A, Ytr, rcond=None)[0]
                lin_pred[te, r] = np.column_stack([np.ones(len(Xte)), Xte]) @ b

            fold_mae[k - 1, r] = np.mean(np.abs(ann_pred[te, r] - Y[te]))
        print(f"  重複 {r + 1}/{N_REPEATS} 完成")

    # --- 4. 逐次重複算指標 ---
    M = empty_metrics(N_REPEATS)
    for r in range(N_REPEATS):
        accum_metrics(M, r, ann_pred[:, r], Y, OFFSET_MM, IDEAL_TOLERANCE_MM)
    L = empty_metrics(N_REPEATS)
    if RUN_LINEAR_BASELINE:
        for r in range(N_REPEATS):
            accum_metrics(L, r, lin_pred[:, r], Y, OFFSET_MM, IDEAL_TOLERANCE_MM)
    with np.errstate(all="ignore"):
        fold_spread = np.nanmean(np.nanstd(fold_mae, axis=0, ddof=1))

    # --- 5. 代表性重複 ---
    rep = int(np.argmin(np.abs(M["mae"] - np.median(M["mae"]))))
    Y_pred = ann_pred[:, rep]
    print(f"\n逐筆結果採用第 {rep + 1} 次重複(MAE={M['mae'][rep]:.3f}，最接近中位數)。")

    Y_pred_offset = Y_pred - OFFSET_MM
    residuals = Y_pred - Y
    off_residuals = Y_pred_offset - Y

    # --- 6. 組輸出表 ---
    results = pd.DataFrame({
        "檔名": ids, "Train_or_Test": "test", "像素長度_px": X, "實際長度_mm": Y,
        "模型預測_mm": Y_pred, "誤差Bias_mm": residuals, "絕對誤差_mm": np.abs(residuals),
        "校正後預測_mm": Y_pred_offset, "校正後誤差Bias_mm": off_residuals,
        "校正後絕對誤差_mm": np.abs(off_residuals), "摺別": fold_id,
    })
    if has_demo:
        results["年齡"] = age
        results["性別"] = np.where(sex == 1, "M", "F")

    # 'A_Test set表現' 與 'C_臨床安全性評估_Test set校正後' 是 C2 的抓列關鍵字，勿改
    metric_names = [
        f"A_Test set表現 (out-of-fold, n={n}, ANN, {N_REPEATS}次重複平均, 輸入={feat_desc})",
        f"B_重複間標準差 (同一份資料重跑{N_REPEATS}次的波動)",
        "--- 分界線 ---",
        f"C_臨床安全性評估_Test set校正後 (offset=-{OFFSET_MM:.2f}mm)",
        f"D_線性迴歸對照組_未校正 (同一組摺, n={n})",
        f"E_線性迴歸對照組_校正後 (offset=-{OFFSET_MM:.2f}mm)",
    ]
    nan, hl = np.nan, RUN_LINEAR_BASELINE
    col = lambda a, b, c, d, e: [a, b, nan, c, d, e]
    metrics = pd.DataFrame({
        "評估範圍_與_嚴格程度": metric_names,
        "MAE_mm": mround(col(np.mean(M["mae"]), sd(M["mae"]), np.mean(M["offMae"]),
                             mean_or(L["mae"], hl), mean_or(L["offMae"], hl)), 3),
        "RMSE_mm": mround(col(np.mean(M["rmse"]), sd(M["rmse"]), np.mean(M["offRmse"]),
                              mean_or(L["rmse"], hl), mean_or(L["offRmse"], hl)), 3),
        # R / r^2：offset 不影響 → C、E 列與 A、D 列同值
        "R_Pearson": mround(col(np.mean(M["r"]), sd(M["r"]), np.mean(M["r"]),
                                mean_or(L["r"], hl), mean_or(L["r"], hl)), 3),
        "R_Square": mround(col(np.mean(M["r2"]), sd(M["r2"]), np.mean(M["r2"]),
                               mean_or(L["r2"], hl), mean_or(L["r2"], hl)), 3),
        "Mean_Bias_mm": mround(col(np.mean(M["bias"]), sd(M["bias"]), np.mean(M["offBias"]),
                                   mean_or(L["bias"], hl), mean_or(L["offBias"], hl)), 3),
        "理想比率_pct": mround([nan, sd(M["ideal"]), nan, np.mean(M["ideal"]),
                                nan, mean_or(L["ideal"], hl)], 1),
    })

    settings = pd.DataFrame({
        "項目": ["量測方法", "輸入特徵", "特徵標籤", "n", "ANN參數數", "樣本每參數", "腳本版本"],
        "值": [METHOD, feat_desc, FEAT_TAG, str(n), str(n_param), f"{n / n_param:.2f}", "C1_v6_py"],
    })

    # --- 全資料線性迴歸：只為解讀效應大小，不是效能 ---
    coef = None
    if has_demo:
        try:
            import statsmodels.api as sm
        except ImportError:
            print("⚠️ 沒裝 statsmodels，略過「線性係數_全資料」(pip install statsmodels)。")
        else:
            tbl = pd.DataFrame({"X": X, "age": age, "sexM": sex})
            preds = ["X"] + (["age"] if USE_AGE else []) + (["sexM"] if USE_SEX else [])
            fit = sm.OLS(Y, sm.add_constant(tbl[preds], has_constant="add")).fit()
            coef = pd.DataFrame({
                "項": ["(Intercept)" if i == "const" else i for i in fit.params.index],
                "係數": mround(fit.params.values, 4), "SE": mround(fit.bse.values, 4),
                "t": mround(fit.tvalues.values, 3), "p值": mround(fit.pvalues.values, 4),
            })

    # --- 7. 匯出 ---
    out = f"預測結果與評估指標_kfold_{METHOD}{FEAT_TAG}.xlsx"
    if os.path.isfile(out):
        os.remove(out)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        results.to_excel(w, sheet_name="所有牙齒預測結果", index=False)
        metrics.to_excel(w, sheet_name="模型評估指標", index=False)
        settings.to_excel(w, sheet_name="模型設定", index=False)
        if coef is not None:
            coef.to_excel(w, sheet_name="線性係數_全資料", index=False)

    print(f"\n✅ 已匯出：{out}\n")
    print("=== 評估指標 ===")
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.unicode.east_asian_width", True):
        print(metrics.to_string(index=False))

    # --- 8. 報告用區塊(校正後) ---
    print("\n=== 臨床評估(校正後，報告用) ===")
    print(f"量測方法：{METHOD} | 輸入：{feat_desc} | offset = -{OFFSET_MM:.2f} mm | "
          f"理想容忍 = ±{IDEAL_TOLERANCE_MM:.2f} mm")
    print(f"[Out-of-fold, n={n}, {N_REPEATS}次重複]")
    print(f"  校正後 MAE  : {np.mean(M['offMae']):.3f} ± {sd(M['offMae']):.3f} mm")
    print(f"  校正後 RMSE : {np.mean(M['offRmse']):.3f} ± {sd(M['offRmse']):.3f} mm")
    print(f"  校正後 Bias : {np.mean(M['offBias']):.3f} ± {sd(M['offBias']):.3f} mm   (正 = 整體偏長)")
    print(f"  R (Pearson) : {np.mean(M['r']):.3f} ± {sd(M['r']):.3f}      (offset 不影響，與未校正同值)")
    print(f"  R^2         : {np.mean(M['r2']):.3f} ± {sd(M['r2']):.3f}      (= R 的平方，同樣不受 offset 影響)")
    print(f"  理想比率    : {np.mean(M['ideal']):.1f} ± {sd(M['ideal']):.1f} %")
    print(f"  摺間 MAE 標準差 : {fold_spread:.3f} mm")

    if RUN_LINEAR_BASELINE:
        lm, am = np.mean(L["offMae"]), np.mean(M["offMae"])
        msg = "線性不輸 ANN" if lm <= am else f"ANN 勝出 {lm - am:.3f} mm"
        print(f"\n[線性迴歸對照組，校正後] MAE {lm:.3f} mm vs ANN {am:.3f} mm  →  {msg}")

    if coef is not None:
        print("\n=== 全資料線性係數(解讀用，非效能) ===")
        print(coef.to_string(index=False))
        print("  age 係數 = 每多 1 歲長度變化(mm)；sexM 係數 = 男比女多幾 mm(控制其他變數後)。")
        print(f"  p 值只是參考：n={n}、未校正多重比較。")

    print("\n👉 對照建議：同一個 METHOD 依序跑")
    print("   (USE_AGE,USE_SEX) = (F,F) → (T,F) → (F,T) → (T,T)，比校正後 MAE。")
    print("👉 C2 的 METHOD / USE_AGE / USE_SEX 要設成跟這次一樣。")


if __name__ == "__main__":
    if RUN_NOW:
        main()
    else:
        print(__doc__)
        print(f"待執行設定：METHOD={METHOD}, USE_AGE={USE_AGE}, USE_SEX={USE_SEX}, "
              f"H={HIDDEN_SIZE}, {N_REPEATS} 次重複, seed={BASE_SEED}")
        print(f"輸入：{input_filename()}\n輸出：{output_filename()}")
