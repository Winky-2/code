"""C1/C2 共用：特徵標籤、性別解析、檔名規則、MATLAB fitnet+trainbr 的 numpy 實作

只依賴 numpy / pandas，不需要 MATLAB 或深度學習框架。

trainbr 對應說明(不確定處已標明)：
  - 網路：1 層隱藏 tansig(=tanh) + purelin 輸出，同 fitnet(H)
  - 前處理：removeconstantrows + mapminmax 到 [-1,1](輸入與輸出都做)，同 fitnet 預設
  - 初始化：隱藏層 Nguyen-Widrow，輸出層 U(-1,1)，近似 MATLAB initnw
  - 訓練：Levenberg-Marquardt + MacKay 貝氏正則化(alpha/beta/gamma 更新)，perf=mse(見 fit 內註解)，
          mu=0.005, mu_dec=0.1, mu_inc=10, mu_max=1e10, min_grad=1e-7(MATLAB 預設)
  - divideFcn='dividetrain'：全部訓練資料都拿來訓練，沒有 validation 早停
  ⚠️ 亂數流與 MATLAB 不同，逐筆預測不會與 MATLAB 位元相同；
     要比的是 10 次重複平均的指標是否落在 MATLAB 的 ± 標準差內。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ==================== 檔名 / 標籤規則(C1、C2 必須一致) ====================
def feat_tag(use_age: bool, use_sex: bool) -> str:
    """輸出檔名的特徵後綴；都關 = ''(跟舊版同名)"""
    parts = []
    if use_age:
        parts.append("年齡")
    if use_sex:
        parts.append("性別")
    return "_" + "".join(parts) if parts else ""


def hidden_tag(hidden) -> str:
    """等同 MATLAB strrep(mat2str(HIDDEN_SIZE),' ','-')：5 → '5'，[3 3] → '[3-3]'"""
    h = list(np.atleast_1d(hidden))
    if len(h) == 1:
        return f"{int(h[0])}"
    return "[" + "-".join(str(int(v)) for v in h) + "]"


# ==================== 數值工具 ====================
def to_num(x) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)


def parse_sex(v) -> np.ndarray:
    """M/男/Male → 1，F/女/Female → 0，其他 → NaN"""
    t = np.array([str(x).strip().upper() if pd.notna(x) else "" for x in v])
    s = np.full(t.size, np.nan)
    s[np.isin(t, ["M", "MALE", "男"])] = 1
    s[np.isin(t, ["F", "FEMALE", "女"])] = 0
    return s


def mround(x, d: int):
    """MATLAB round(x,d)：四捨五入遠離 0(numpy 的 round 是銀行家捨入，會有 0.0005 差異)"""
    x = np.asarray(x, dtype=float)
    f = 10.0 ** d
    return np.sign(x) * np.floor(np.abs(x) * f + 0.5) / f


# ==================== fitnet + trainbr ====================
class FitnetBR:
    """單隱藏層 fitnet(H, 'trainbr') 的 numpy 版"""

    def __init__(self, hidden: int, epochs: int = 200, rng: np.random.Generator | None = None,
                 mu: float = 0.005, mu_dec: float = 0.1, mu_inc: float = 10.0,
                 mu_max: float = 1e10, min_grad: float = 1e-7):
        if np.ndim(hidden) != 0 and len(np.atleast_1d(hidden)) != 1:
            raise ValueError("目前只實作單一隱藏層(HIDDEN_SIZE 需為純量)。")
        self.H = int(np.atleast_1d(hidden)[0])
        self.epochs = epochs
        self.rng = rng if rng is not None else np.random.default_rng()
        self.mu0, self.mu_dec, self.mu_inc, self.mu_max = mu, mu_dec, mu_inc, mu_max
        self.min_grad = min_grad

    # ---- mapminmax ----
    @staticmethod
    def _mm_fit(a):
        lo, hi = a.min(0), a.max(0)
        return lo, np.where(hi > lo, hi - lo, 1.0)

    @staticmethod
    def _mm(a, lo, rg):
        return 2.0 * (a - lo) / rg - 1.0

    # ---- 參數向量 ↔ 權重 ----
    def _unpack(self, w):
        H, p = self.H, self.p
        i = H * p
        return w[:i].reshape(H, p), w[i:i + H], w[i + H:i + 2 * H], w[i + 2 * H]

    def _forward(self, w, X):
        W1, b1, v, c = self._unpack(w)
        h = np.tanh(X @ W1.T + b1)
        return h @ v + c, h, v

    def _jacobian(self, X, h, v):
        n = X.shape[0]
        d = v * (1.0 - h ** 2)                                   # (n, H)
        dW1 = (d[:, :, None] * X[:, None, :]).reshape(n, -1)     # 與 W1.reshape(H,p) 順序一致
        return np.hstack([dW1, d, h, np.ones((n, 1))])

    def _init(self):
        H, p = self.H, self.p
        beta = 0.7 * H ** (1.0 / p)
        W1 = self.rng.uniform(-1, 1, (H, p))
        W1 = beta * W1 / np.linalg.norm(W1, axis=1, keepdims=True)
        b1 = beta * np.linspace(-1, 1, H) * np.sign(W1[:, 0]) if H > 1 else self.rng.uniform(-beta, beta, H)
        v = self.rng.uniform(-1, 1, H)
        c = self.rng.uniform(-1, 1)
        return np.concatenate([W1.ravel(), b1, v, [c]])

    # ---- 訓練 ----
    def fit(self, X, y):
        X = np.asarray(X, float).reshape(len(y), -1)
        y = np.asarray(y, float).ravel()
        self.keep = X.max(0) > X.min(0)                          # removeconstantrows
        Xk = X[:, self.keep]
        self.p = Xk.shape[1]
        self.xlo, self.xrg = self._mm_fit(Xk)
        self.ylo, self.yrg = self._mm_fit(y[:, None])
        Xn = self._mm(Xk, self.xlo, self.xrg)
        t = self._mm(y[:, None], self.ylo, self.yrg).ravel()

        w = self._init()
        N, nE = w.size, t.size
        I = np.eye(N)
        out, h, v = self._forward(w, Xn)
        e = out - t
        mse, ssw = e @ e / nE, w @ w
        # 同 MATLAB trainbr initializeTraining：迴圈前就以 gamma=N 設 alpha/beta。
        # perf 用 mse(fitnet 預設 performFcn)，LM 步用未正規化的 J'J、J'e
        #  → 資料項權重相當於 sse 版的 nE 倍。⚠️ 這點依 MATLAB 行為推定，未逐行核對原始碼；
        #     純 sse 版(教科書 MacKay)在本資料會塌縮成常數預測，已實測排除。
        mu = self.mu0
        alpha = 1.0 if ssw == 0 else N / (2.0 * ssw)
        beta = 1.0 if mse == 0 else (nE - N) / (2.0 * mse)
        if beta <= 0:
            beta = 1.0
        F = beta * mse + alpha * ssw

        for _ in range(self.epochs):
            J = self._jacobian(Xn, h, v)
            JJ, Je = J.T @ J, J.T @ e
            if np.linalg.norm(2.0 * (beta * Je + alpha * w)) < self.min_grad:
                break
            stepped = False
            while mu <= self.mu_max:
                dw = -np.linalg.solve(beta * JJ + (mu + alpha) * I, beta * Je + alpha * w)
                w2 = w + dw
                out2, h2, v2 = self._forward(w2, Xn)
                e2 = out2 - t
                mse2, ssw2 = e2 @ e2 / nE, w2 @ w2
                if beta * mse2 + alpha * ssw2 < F:
                    w, h, v, e, mse, ssw = w2, h2, v2, e2, mse2, ssw2
                    mu *= self.mu_dec
                    stepped = True
                    break
                mu *= self.mu_inc
            if not stepped:
                break
            # MacKay 超參數更新
            gamma = N - alpha * np.trace(np.linalg.inv(beta * JJ + alpha * I))
            gamma = float(np.clip(gamma, 1e-12, nE - 1e-12))
            alpha = gamma / (2.0 * max(ssw, 1e-300))
            beta = (nE - gamma) / (2.0 * max(mse, 1e-300))
            F = beta * mse + alpha * ssw
        self.gamma_ = gamma if self.epochs > 0 else float(N)   # 有效參數數(診斷用)

        self.w = w
        return self

    def predict(self, X):
        X = np.asarray(X, float).reshape(np.shape(X)[0], -1)[:, self.keep]
        out, _, _ = self._forward(self.w, self._mm(X, self.xlo, self.xrg))
        return (out + 1.0) / 2.0 * self.yrg[0] + self.ylo[0]
