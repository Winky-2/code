#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gpr_core.py — 與 MATLAB fitrgp(FitMethod/PredictMethod = 'exact')相同的 GPR 數學
===============================================================
模型(對應 C1_pxToMm_GPR.m)：
    X 先用訓練集 mean/std 標準化(手動，fitrgp 'Standardize' false)
    y = h(x)ᵀβ + f(x) + ε，  f ~ GP(0, k)，  ε ~ N(0, σ²)
    h(x) = [1, x](basis 'linear') | [1](constant) | 無(none)
    k    = σf² exp(−r²/2)，r = ‖(x−x')/σL‖            (squaredexponential / ard…)
         = σf²(1+√5r+5r²/3)exp(−√5r)                   (matern52 / ardmatern52)
預測：
    mean = h*ᵀβ + k*ᵀ (K+σ²I)⁻¹ (y − Hβ)
    sd   = sqrt(σf² − k*ᵀ(K+σ²I)⁻¹k* + σ²)              ← fitrgp predict 的 ysd(含雜訊)
"""
from __future__ import annotations

import numpy as np

KERNELS = ("squaredexponential", "ardsquaredexponential", "matern52", "ardmatern52")
BASES = ("linear", "constant", "none")


def basis(X, kind):
    n = X.shape[0]
    if kind == "linear":
        return np.column_stack([np.ones(n), X])
    if kind == "constant":
        return np.ones((n, 1))
    if kind == "none":
        return np.zeros((n, 0))
    raise ValueError(f"basis 只能是 {BASES}，目前是 {kind!r}")


def kernel(A, B, kind, theta):
    """theta 與 MATLAB KernelParameters 同順序：[σL(或 σL1..σLd); σf]。"""
    theta = np.ravel(theta).astype(np.float64)
    ls, sf = theta[:-1], theta[-1]
    if not kind.startswith("ard"):
        ls = np.full(A.shape[1], ls[0])
    D = (A[:, None, :] - B[None, :, :]) / ls
    r2 = np.sum(D * D, axis=2)
    if "squaredexponential" in kind:
        return sf ** 2 * np.exp(-0.5 * r2)
    if "matern52" in kind:
        r = np.sqrt(np.maximum(r2, 0))
        return sf ** 2 * (1 + np.sqrt(5) * r + 5 * r2 / 3) * np.exp(-np.sqrt(5) * r)
    raise ValueError(f"kernel 只能是 {KERNELS}，目前是 {kind!r}")


def _chol(K):
    jitter = 0.0
    for _ in range(6):
        try:
            return np.linalg.cholesky(K + jitter * np.eye(len(K)))
        except np.linalg.LinAlgError:
            jitter = 1e-10 if jitter == 0 else jitter * 100
    raise np.linalg.LinAlgError("核矩陣非正定")


def _solve(L, b):
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def gls_beta(Xn, y, kind, theta, sigma, basis_kind):
    """給定超參數，用 GLS 求 β(fitrgp 的 profile likelihood 做法)。回傳 (β, L, 負對數概似)。"""
    K = kernel(Xn, Xn, kind, theta) + sigma ** 2 * np.eye(len(y))
    L = _chol(K)
    H = basis(Xn, basis_kind)
    if H.shape[1]:
        KiH = _solve(L, H)
        beta = np.linalg.solve(H.T @ KiH, KiH.T @ y)
    else:
        beta = np.zeros(0)
    r = y - H @ beta
    nll = 0.5 * r @ _solve(L, r) + np.log(np.diag(L)).sum() + 0.5 * len(y) * np.log(2 * np.pi)
    return beta, L, float(nll)


def fit(Xn, y, kind="squaredexponential", basis_kind="linear", n_starts=1, seed=42):
    """最大化邊際概似求 (theta, σ, β)。初始值與界限比照 fitrgp 預設。
    n_starts=1 = 跟 fitrgp 一樣只從預設初始值出發；多起點常會找到「σL 極小、把訓練點
    硬背下來」的解(概似較高但新影像預測會跳動)，所以不建議加大。"""
    from scipy.optimize import minimize

    d = Xn.shape[1]
    n_ls = d if kind.startswith("ard") else 1
    sy = float(np.std(y, ddof=1))
    sig_lo = 1e-2 * sy                                  # fitrgp SigmaLowerBound 預設
    x0 = np.r_[np.full(n_ls, np.log(np.mean(np.std(Xn, axis=0, ddof=1)))),
               np.log(sy / np.sqrt(2)), np.log(sy / np.sqrt(2))]

    def unpack(z):
        theta = np.exp(z[:-1])
        return theta, sig_lo + np.exp(z[-1])

    def obj(z):
        theta, s = unpack(z)
        try:
            return gls_beta(Xn, y, kind, theta, s, basis_kind)[2]
        except np.linalg.LinAlgError:
            return 1e10

    rng = np.random.default_rng(seed)
    starts = [x0] + [x0 + rng.normal(0, 1.0, x0.shape) for _ in range(n_starts - 1)]
    bounds = [(-9, 9)] * len(x0)
    best = min((minimize(obj, s, method="L-BFGS-B", bounds=bounds) for s in starts), key=lambda r: r.fun)
    theta, sigma = unpack(best.x)
    beta, _, nll = gls_beta(Xn, y, kind, theta, sigma, basis_kind)
    return {"theta": theta, "sigma": float(sigma), "beta": beta, "nll": nll}


class Predictor:
    """由 (標準化後訓練 X, y, 超參數, β) 建立；跟 MATLAB 匯出的 JSON 同欄位。"""

    def __init__(self, Xn, y, kind, basis_kind, theta, sigma, beta, mu, sg):
        self.Xn = np.atleast_2d(np.asarray(Xn, dtype=np.float64))
        if self.Xn.shape[0] == 1 and len(np.ravel(y)) > 1:
            self.Xn = self.Xn.T                         # MATLAB 單欄會被攤平
        self.y = np.ravel(y).astype(np.float64)
        self.kind, self.basis_kind = kind, basis_kind
        self.theta = np.ravel(theta).astype(np.float64)
        self.sigma = float(np.ravel(sigma)[0])
        self.beta = np.ravel(beta).astype(np.float64)
        self.mu, self.sg = np.ravel(mu).astype(np.float64), np.ravel(sg).astype(np.float64)
        K = kernel(self.Xn, self.Xn, kind, self.theta) + self.sigma ** 2 * np.eye(len(self.y))
        self.L = _chol(K)
        self.alpha = _solve(self.L, self.y - basis(self.Xn, basis_kind) @ self.beta)

    def predict(self, X):
        """X：原始尺度(未標準化)，shape (m, d)。回傳 (mean, sd)。"""
        Xn = (np.atleast_2d(np.asarray(X, dtype=np.float64)) - self.mu) / self.sg
        ks = kernel(Xn, self.Xn, self.kind, self.theta)
        mean = basis(Xn, self.basis_kind) @ self.beta + ks @ self.alpha
        v = np.linalg.solve(self.L, ks.T)
        var = self.theta[-1] ** 2 - np.sum(v * v, axis=0) + self.sigma ** 2
        return mean, np.sqrt(np.maximum(var, 0))
