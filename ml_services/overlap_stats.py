"""重叠前瞻窗样本的方差修正与功效评估（2026-10-13）。

**为什么需要**：本项目的预测文件里，每一行标签是「该日之后 20 个交易日的收益」。
相邻两行的前瞻窗重叠 19/20 天 ⇒ 763 个交易日**不是** 763 个独立观测。
但它们也不是 39 个独立观测（那是把信息扔掉）。中间的正确处理是方差修正。

**三种口径，从最严到最松**
1. `naive`      —— 直接 t 检验（假设日间独立）。**最不可信**：会高估显著性。
2. `hac`         —— Newey–West，滞后阶数取前瞻期 −1 = 19。
3. `block_boot`  —— 移动块 bootstrap，块长 = 前瞻期 20 天，重抽整块以保留块内相关。

**本模块的硬约束**：所有函数都**不做**任何假设检验的"修正后显著性"以外的推断，
只返回 `(点估计, 有效样本量, 标准误)`。是否显著由调用方按预登记判据决定。

**为什么这能让 D27 提前**：D27 原先要求「≥12 个未污染新折」，理由是新折之间不重叠。
若 HAC / block bootstrap 能在**不引入重叠假设**的前提下恢复部分自由度，
则可用「重叠样本 + 方差修正」作为 A 级的替代路径 —— 但这是**方法变更**，
必须先验证其修正的可靠性（新条 D30），不能直接拿来给 D27 判案。
"""

import numpy as np


def naive_t(x):
    """朴素 t：假设观测独立。返回 (mean, n_eff, se, t)。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.nan, n, np.nan, np.nan
    se = x.std(ddof=1) / np.sqrt(n)
    return float(x.mean()), n, float(se), float(x.mean() / se) if se else np.nan


def hac_se(x, lags):
    """Newey–West 自相关一致标准误。

    lags 取 前瞻期−1（默认 19）：重叠 k 天的前瞻窗，使 lag-k 自相关非零，
    故 Bartlett 核保留到 lags。
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < lags + 2:
        return np.nan, n
    u = x - x.mean()
    g0 = float((u * u).sum() / n)
    var = g0
    for k in range(1, lags + 1):
        wk = 1.0 - k / (lags + 1.0)
        gk = float((u[k:] * u[:-k]).sum() / n)
        var += 2.0 * wk * gk
    var = max(var, 1e-18)
    return float(np.sqrt(var / n)), n


def hac_t(x, lags=19):
    se, n = hac_se(x, lags)
    if not se or not np.isfinite(se):
        return np.nan, n, np.nan, np.nan
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()), n, float(se), float(x.mean() / se)


def block_bootstrap_se(x, block, n_boot=2000, seed=0):
    """移动块 bootstrap 标准误：重抽「连续 block 天」整块，保留块内相关。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 * block:
        return np.nan, n
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max + 1, size=n_blocks)
        samp = np.concatenate([x[s:s + block] for s in starts])[:n]
        means[i] = samp.mean()
    return float(means.std(ddof=1))


def power_report(x, lags=None, block=None, n_boot=2000):
    """一次性给出三种口径的 (mean, n_eff, se, t)，供对照。

    ⚠️ `n_eff` 只是一个**参考**，不同口径的 n_eff 定义不同：
       - naive: 原始样本量（假设独立）
       - hac:   仍是原始样本量，但 se 已含长程方差
       - block: 原始样本量，se 由整块重抽得到
       真正可比的是 **se** 与 **t**，不是 n_eff。
    """
    if lags is None:
        lags = 19
    if block is None:
        block = 20
    out = {}
    m, n, se, t = naive_t(x)
    out['naive'] = {'mean': m, 'n_eff': n, 'se': se, 't': t}
    m, n, se, t = hac_t(x, lags)
    out['hac'] = {'mean': m, 'n_eff': n, 'se': se, 't': t}
    se = block_bootstrap_se(x, block, n_boot=n_boot)
    se = float(se) if np.isscalar(se) or np.ndim(se) == 0 else np.nan
    if np.isfinite(se) and se > 0:
        out['block'] = {'mean': float(np.nanmean(x)), 'n_eff': int(x.size),
                        'se': se, 't': float(np.nanmean(x) / se)}
    else:
        out['block'] = {'mean': float(np.nanmean(x)), 'n_eff': int(x.size),
                        'se': np.nan, 't': np.nan}
    return out