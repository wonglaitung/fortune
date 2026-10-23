#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
月度护栏：对最新 Walk-forward 预测一键复核（DECISIONS D2）

对给定/最新 prediction_analysis.csv 输出：
  - 20d TopK 组合（raw 打分，D73）：净IR / 95%CI / P(IR>0.5) / 累计 / 换手
  - PBO（CSCV）与 DSR（Deflated Sharpe）
  - 信号超额 lift（胜率 − 无条件买入基准）
判定（docs/DECISIONS.md D2）：
  - 通过：净IR≥0.7 且 PBO<0.5 且 DSR≥0.95 → 可升级
  - 保留：未达升级但 IR>0 → 低配辅助
  - 停用：IR≤0

用法：
  python3 ml_services/monthly_guardrail.py                          # 自动找最新 20d 预测
  python3 ml_services/monthly_guardrail.py --horizon 20 --pred <csv> --output output/mg.md
"""

import os
import sys
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.portfolio_backtest import load_panel, backtest, bootstrap, _stats, COST, selection_ic
from ml_services.eval_overfit import cscv_pbo, deflated_sharpe

def min_detectable_ir(n, horizon):
    """本窗口（n 个非重叠期）最小可探测「超额 IR」(α=0.05, 功率80% ⇒ z≈2.8)。

    MDE_IR = 2.8·√(252/(h·n))（std 在 IR 里约掉）；低于它的真效应**测不出**（lessons 三.75）。
    """
    return 2.8 * np.sqrt(252.0 / (horizon * n)) if n else float('nan')


def latest_pred(horizon, market='hk'):
    """最新 prediction_analysis.csv；默认港股（排除 A股目录，避免误取）"""
    if str(market).lower().startswith('a'):
        # 覆盖 catboost 与 lightgbm（P4.1 A/B）两种目录
        cands = glob.glob(f"output/*_a_stock_*_{horizon}d/prediction_analysis.csv")
    else:
        cands = [f for f in glob.glob(f"output/*_catboost_{horizon}d/prediction_analysis.csv")
                 if '_a_stock_' not in f]
        if not cands:
            cands = glob.glob(f"output/walk_forward_catboost_{horizon}d_*.csv")
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def signal_lift(df):
    """信号胜率 − 无条件买入基准胜率（净收益>0.5%）"""
    df = df.copy()
    for a, b in [('Predict_Prob', 'prob'), ('Actual_Return', 'ret'), ('Dynamic_Threshold', 'dyn')]:
        # 旧A股CSV缺 Dynamic_Threshold → 阈值回退 0.5（与 backtest_eval 一致）
        df[b] = pd.to_numeric(df[a], errors='coerce') if a in df.columns else np.nan
    df['dyn'] = df['dyn'].fillna(0.5)
    df = df.dropna(subset=['ret', 'prob'])
    trade = df['prob'] >= df['dyn']
    win = df['ret'] > COST
    base = win.mean()
    wr = win[trade].mean() if trade.sum() else np.nan
    return wr, base, (wr - base) if wr is not None else np.nan


def win_rate_paired_test(df, cost=COST, topk=10):
    """胜率 lift 的**逐日配对**检验，**固定 TopK 口径**。

    ⚠️ 2026-10-13（statistic redesign）：原先按模型 UP/DOWN 分组取「判涨组」，
    而**每日判涨人数在 5~53 之间剧烈波动** ⇒ 胜率的分母不稳 ⇒ lift 的方差随日而异
    （实测：判涨人数少的日子 lift sd 0.0999，人多的日子 0.0403，**比值 2.48**）。
    异方差下任何标准误修正（HAC/块bootstrap）都是在给一个病态统计量打补丁，
    且它们会把「可预测的方差变化」也算进序列相关去修正 ⇒ se 被放大 2.2 倍。

    **改为固定取每日概率最高的 `topk` 只**：分母固定 ⇒ 方差稳定 ⇒ 统计量良定义。
    判定链也相应只引用本口径（`top10-raw`），不再引用 UP/DOWN 口径。

    仍并列三种标准误口径（naive / HAC / 块bootstrap），以 HAC 与块bootstrap 为准（三.69）。
    """
    from ml_services.overlap_stats import hac_t, block_bootstrap_se
    # 为什么必须另算（lessons 三.64 规则 1）：全局胜率 lift 会把「哪几天模型偏乐观」与
    # 「哪几天大盘在涨」混在一起。逐日去均值后才是纯选股口径。
    # 与收益口径必须**同时报**：实测出现过胜率 lift t=+5.08（p<0.0001）而收益 lift t=−0.37（p=0.71），
    # 只报前者会把「统计显著」误读成「有 edge」——命中率提升但收益不变是极常见的形态。
    if 'Actual_Return' not in df.columns:
        return None
    d = df.copy()
    if 'ret' in d.columns:                  # load_panel 已备好 ret/prob
        ret_s = d['ret']
    else:
        ret_s = pd.to_numeric(d['Actual_Return'], errors='coerce')
    d['ret'] = ret_s
    d['prob'] = pd.to_numeric(d['Predict_Prob'], errors='coerce')
    d = d.dropna(subset=['ret', 'prob'])
    dc = 'date' if 'date' in d.columns else 'Date'
    # 信号 = 每日概率最高的 topk 只（**固定分母**）；基准 = 同日全池。
    # ⚠️ 基准若也过滤成信号组，差值按构造恒为 0 —— 早期版本踩过这个静默错误。
    sig = pd.concat([g.nlargest(topk, 'prob') for _, g in d.groupby(dc)])
    per_day = []
    for dt, x in d.groupby(dc):
        sx = sig.loc[sig[dc] == dt]
        if len(sx) < 3 or len(x) < 10:
            continue
        per_day.append((sx['ret'] > cost).mean() - (x['ret'] > cost).mean())
    if len(per_day) < 20:
        return None
    arr = np.asarray(per_day, dtype=float)
    t, p = ttest_1samp(arr, 0.0)
    # ⚠️ 2026-10-13：标签是「后 20 个交易日」，相邻天的前瞻窗重叠 19/20 天 ⇒
    #    **朴素 t 的标准误被低估约 2.2 倍**，会把不显著读成显著。
    #    此处并列三种口径，报出的显著性以 HAC/块 bootstrap 为准（lessons 三.69）。
    m, n_h, se_hac, t_hac = hac_t(arr, lags=19)
    se_blk = block_bootstrap_se(arr, block=20, n_boot=2000)
    t_blk = (float(arr.mean() / se_blk)
             if np.isfinite(se_blk) and se_blk > 0 else np.nan)
    return {'lift': float(arr.mean()), 't': float(t), 'p': float(p), 'n': int(arr.size),
            't_hac': float(t_hac), 'se_hac': float(se_hac),
            't_block': float(t_blk), 'se_block': float(se_blk)}


def equal_weight_baseline(bt):
    """等权基准 —— **直接复用 `portfolio_backtest.backtest()` 内部的 `bt['bench']`**。

    ⚠️ 这里曾经写错过（lessons 三.67）：早先版本自己用全部**日频** `Actual_Return` 的横截面均值当基准，
    而 backtest 的调仓日是 `dates[::horizon]`（**非重叠**）、且其 `bench` 也是在这批调仓日上算的。
    两套口径混用 ⇒ 与模型组合的对比整体失真。
    ⇒ **凡是项目里已经有的基准，不许自己再造一个**。

    口径对齐后 D26 实测（39 个非重叠调仓期，top10-neutral）：
      等权 bench      +1.719%/期
      组合 毛 gross   +1.516%/期   毛−等权 −0.203pp  95%CI [−1.10%, +0.70%]  p=0.66 **不显著**
      组合 净 net     +1.163%/期   净−等权 −0.556pp  95%CI [−1.46%, +0.35%]  p=0.24 **不显著**
      成本拖累        0.353%/期
    ⇒ 正确表述是「**测不出超额**」，**不是**「打平」（CI 同时容纳差很多与略好，见三.67）。
    """
    if bt is None or 'bench' not in bt.columns:
        return None
    b = bt['bench'].dropna()
    if len(b) < 20:
        return None
    t, p = ttest_1samp(b.values, 0.0)
    return {'mean': float(b.mean()), 't': float(t), 'p': float(p), 'n': int(len(b))}


def vs_baseline(series, bench, label):
    """与等权基准的**配对**差额 + 95%CI。

    必须带 CI：只报「−0.20%/期」会被读成「打平」，
    而 CI [−1.10%, +0.70%] 的正确读法是「**测不出**」（lessons 三.67）。
    """
    d = np.asarray(series, dtype=float) - np.asarray(bench, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return None
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    t, p = ttest_1samp(d, 0.0)
    return {'label': label, 'diff': float(m), 'ci_lo': float(m - 1.96 * se),
            'ci_hi': float(m + 1.96 * se), 'p': float(p), 'n': int(len(d))}


def pbo_null_band(M, n_perm=10, seed=0):
    """PBO 的零假设分布：把时间轴整行重排后重算 PBO，得到「完全没有时间可预测性」时的取值范围。

    为什么必须有这个（lessons 三.63）：CSCV 在样本内挑夏普最高的配置、看它样本外排第几。
    当几个配置彼此不可分时，样本内 argmax 就是抛硬币 ⇒ PBO 必然趋近 1。
    实测 T=39×6 时零假设带为 0.73–0.94，真实数据 0.871 落在带内 ⇒ 与「无信息」不可区分。
    单看一个 PBO 数字会把它误读成「过拟合」，故此函数把噪声带一并算出。

    ⚠️ 必须**整行重排**，不能逐列独立打乱：
    逐列打乱会把「同一期各配置面对同一个市场」这个真实相关性也毁掉，
    六个配置变成互不相干 ⇒ CSCV 的 IS argmax 更随机 ⇒ PBO 被系统性推到 1，
    零假设带会假性收窄到 [0.97, 1.00]，反而掩盖真实数据的读数。
    整行重排保留跨配置相关性、只破坏时间可预测性，才是这里该问的零假设。
    """
    rng = np.random.default_rng(seed)
    vals = []
    X = M.values
    for _ in range(n_perm):
        vals.append(cscv_pbo(X[rng.permutation(X.shape[0]), :])[0])
    v = np.asarray(vals)
    return float(np.percentile(v, 5)), float(np.percentile(v, 95)), len(v)


def config_spread(M):
    """各配置全期夏普的极差。极差过小 ⇒ 配置互不可分 ⇒ PBO 必然高（lessons 三.63 准则 2）。"""
    sr = (M.mean() / M.std(ddof=1))
    return float(sr.max() - sr.min()), sr


def run(horizon, pred_csv, topk, cost, out_md, market='hk', null_perm=200, dropout=0):
    df = load_panel(pred_csv, market=market)
    bt = backtest(df, horizon, topk, cost=cost, dropout=dropout)
    # ⓪ 同一性闸第⑤问：能力指标必须用**部署打分口径**（raw prob，D73）
    cap = selection_ic(df)
    st = _stats(bt['top_net'].values, horizon)
    # ⛔ 「净IR」是组合**绝对**收益（含 beta），不是超额。三.68：必须同时报超额口径。
    st_bench = _stats(bt['bench'].values, horizon)
    st_excess = _stats(bt['top_net'].values - bt['bench'].values, horizon)
    # 功效：本窗口（n 个非重叠期）最小可探测「超额 IR」
    mde_ir = min_detectable_ir(st_excess['n'], horizon)
    c = bootstrap(bt['top_net'].values, horizon)
    turnover = float(bt['turnover'].mean())
    wr, base, lift = signal_lift(df)
    # D27 D 级 + E 级：胜率配对检验 与 等权全池基准 必须与主表并列
    wr_test = win_rate_paired_test(df, cost=cost)
    ew = equal_weight_baseline(bt)          # 复用 backtest 内部 bench，口径对齐
    vb_gross = vs_baseline(bt['top_gross'], bt['bench'], '毛 gross') if ew else None
    vb_net = vs_baseline(bt['top_net'], bt['bench'], '净 net') if ew else None

    from ml_services.eval_overfit import build_matrix
    M = build_matrix(horizon, pred_csv, market=market)
    pbo, _ = cscv_pbo(M.values)
    mu = M.mean(); sd = M.std(ddof=1); sr = mu / sd
    best = sr.idxmax()
    from scipy.stats import skew, kurtosis
    dsr = deflated_sharpe(sr[best], len(M), M.shape[1], float(sr.var(ddof=1)),
                          float(skew(M[best].values)), float(kurtosis(M[best].values, fisher=False)))

    spread, _ = config_spread(M)
    null_lo = null_hi = None
    if null_perm and null_perm > 0:
        null_lo, null_hi, _ = pbo_null_band(M, n_perm=null_perm)

    if null_lo is not None:
        in_band = null_lo <= pbo <= null_hi
        if in_band:
            pbo_read = (f"PBO {pbo:.2f} 落在零假设带 [{null_lo:.2f}, {null_hi:.2f}] 内 "
                        f"⇒ 与「无信息」不可区分，**不可读作过拟合**")
        elif pbo < null_lo:
            pbo_read = (f"PBO {pbo:.2f} **低于**零假设带 [{null_lo:.2f}, {null_hi:.2f}] 下界 "
                        f"⇒ 本窗口的偶然，不可采信（lessons 三.29）")
        else:
            pbo_read = (f"PBO {pbo:.2f} **高于**零假设带 [{null_lo:.2f}, {null_hi:.2f}] 上界 "
                        f"⇒ 该过拟合读数在本尺寸下才站得住")
    else:
        pbo_read = f"PBO {pbo:.2f}（未算零假设带，--null-perm=0）"

    passed = st['ir'] >= 0.7 and pbo < 0.5 and (dsr is not None and dsr >= 0.95)
    if st['ir'] <= 0:
        verdict = "🔴 停用（IR≤0）"
    elif passed:
        verdict = "🟢 通过 → 可升级"
    else:
        verdict = "🟡 保留低配（未达升级门槛）"

    # D27 E 级：模型组合 vs 等权全池。**强制净+毛双口径 + 配对 CI + 成本占比**
    # （lessons 三.65 成本归因；lessons 三.67 未检出不等于零，CI 必须随行）
    alpha_verdict = None
    cost_drag = turnover * cost
    if ew is not None and vb_gross is not None and vb_net is not None:
        strat, gross = st['mean'], st['mean'] + cost_drag
        net_rel = (strat / ew['mean'] - 1) if ew['mean'] else float('nan')
        gross_rel = (gross / ew['mean'] - 1) if ew['mean'] else float('nan')
        share = (cost_drag / (ew['mean'] - strat) * 100) if (ew['mean'] - strat) > 0 else float('nan')
        lines = [
            f"等权基准 **bt['bench']**（同 {ew['n']} 个非重叠调仓日）**{ew['mean']*100:+.2f}%/期**",
            f"模型组合 **净 {strat*100:+.2f}%/期**（{net_rel*100:+.0f}%）　"
            f"**毛 {gross*100:+.2f}%/期**（{gross_rel*100:+.0f}%）",
            f"配对差额 vs 等权（同调仓日）：毛 gross {vb_gross['diff']*100:+.3f}%/期　"
            f"95%CI [{vb_gross['ci_lo']*100:+.3f}%, {vb_gross['ci_hi']*100:+.3f}%]　p={vb_gross['p']:.3f} → "
            + ('显著' if vb_gross['p'] < 0.05 else '**不显著 ⇒ 读作「测不出超额」，不是「打平」**'),
            f"　　（同上）净 net　 {vb_net['diff']*100:+.3f}%/期　"
            f"95%CI [{vb_net['ci_lo']*100:+.3f}%, {vb_net['ci_hi']*100:+.3f}%]　p={vb_net['p']:.3f} → "
            + ('显著' if vb_net['p'] < 0.05 else '不显著'),
            f"交易成本拖累 **{cost_drag*100:.3f}%/期**（单边换手 {turnover*100:.0f}% × 双边成本 {cost*100:.1f}%）"
            + (f"，占净-毛总差距 **{share:.0f}%**" if np.isfinite(share) else ''),
            f"⚠️ **降换手的上限**：即便换手降到 0，差距也只能从 {vb_net['diff']*100:+.3f}% 收窄到 "
            f"{vb_gross['diff']*100:+.3f}%/期 —— **它救不了策略，主要矛盾是毛口径本身没有可测超额**。",
            "⚠️ `backtest()` 已内置 `dropout`（TopK-Dropout 降换手），护栏当前传 `dropout=0` "
            "⇒ 机制已存在只差传参，**属执行动作，不是新主线**。",
        ]
        alpha_verdict = "\n".join("- " + x for x in lines)

    # D27 D 级：胜率与收益必须同时显著，否则不得声称有 edge
    dual = None
    if wr_test is not None:
        wr_sig = (abs(wr_test['t_hac']) > 1.96) if np.isfinite(wr_test['t_hac']) else (wr_test['p'] < 0.05)
        ret_sig = st['ir'] > 0 and c['ir_lo'] > 0
        dual = (wr_sig, ret_sig)
        if wr_sig and not ret_sig:
            dual_note = ("⚠️ **胜率显著、收益不显著** —— 命中率提升不等于有 edge；"
                         "只报胜率会把统计显著误读成可交易（lessons 三.64 规则 1）。")
        elif ret_sig and not wr_sig:
            dual_note = "收益显著但胜率不显著 —— 分层排序未改善命中率，注意收益可能来自 beta。"
        elif wr_sig and ret_sig:
            dual_note = "胜率与收益**双口径**均显著 —— 两边一致，可信度较高。"
        else:
            dual_note = "胜率与收益**双口径**均不显著 —— 无可展示的选股能力。"

    L = []
    L.append(f"# 月度护栏复核（{horizon}d）\n")
    L.append(f"- 日期: {datetime.now():%Y-%m-%d %H:%M:%S}")
    L.append(f"- 数据: `{pred_csv}`")
    L.append(f"- TopK={topk}（raw 打分），成本 {cost:.3f}\n")
    L.append("| 指标 | 值 | 门槛 |")
    L.append("|------|----|------|")
    L.append(f"| 净IR（**绝对口径，含 beta**） | {st['ir']:.2f} [{c['ir_lo']:.2f},{c['ir_hi']:.2f}] | — 见超额行 |")
    L.append(f"| **等权基准 IR**（无模型） | {st_bench['ir']:.2f} | 参照系 |")
    L.append(f"| **超额 IR**（− 基准） | **{st_excess['ir']:.2f}** | ≥0.7 且 CI 下界>0 更佳 |")
    L.append(f"| **功效/MDE**（α=.05 功率80%） | 超额 IR 最小可探测 **{mde_ir:.2f}** | n={st_excess['n']} 期 |")
    L.append(f"| P(IR>0.5) | {c['p_ir']*100:.0f}% | — |")
    L.append(f"| 净均收益/期 | {st['mean']*100:+.2f}% | >0 |")
    L.append(f"| 累计净收益 | {st['cum']*100:+.1f}% | — |")
    L.append(f"| 换手 | {turnover*100:.0f}% | 越低越好 |")
    L.append(f"| **PBO** | **{pbo:.2f}** | <0.5 |")
    L.append(f"| **DSR**（最优 {best}） | **{dsr:.3f}** | ≥0.95 |")
    L.append(f"| 信号胜率 / 基准胜率 | {wr*100:.1f}% / {base*100:.1f}% | — |")
    L.append(f"| **超额 lift** | **{lift*100:+.1f}pp** | >0 |")
    L.append(f"| **等权基准**（无模型，同调仓日） | **{ew['mean']*100:+.2f}%/期** | 见下方 alpha 判定（带 CI） |" if ew is not None else "")
    L.append("")
    L.append(f"## 判定：{verdict}")
    L.append("")
    L.append("> 判定规则：净IR≥0.7 且 PBO<0.5 且 DSR≥0.95 → 升级；IR>0 但未达 → 保留低配；IR≤0 → 停用。")
    L.append("> （docs/DECISIONS.md D2）")
    L.append(">")
    L.append("> ⚠️ **该判定式用的是「绝对 IR」，含 beta —— 三.68。** 等权全池不做任何选择也有 "
             f"IR {st_bench['ir']:.2f}；策略要「赢」必须超过这个数，而不是超过 0。")
    L.append(f"> 实测 D26：策略绝对 IR {st['ir']:.2f} **低于**基准 {st_bench['ir']:.2f}，"
             f"超额 IR {st_excess['ir']:.2f}。\n")
    L.append(f"> ⚠️ **功效门槛**：本窗口 n={st_excess['n']} 期 ⇒ 最小可探测超额 IR ≈ **{mde_ir:.2f}**；"
             f"**低于它的真 edge 测不出** ⇒ **「不显著」≠「没有」**（lessons 三.75）。\n")
    L.append("### 相对等权全池的判定（D27 E 级，强制）")
    L.append("")
    L.append(alpha_verdict if alpha_verdict else '- 等权基准不可用（样本不足）')
    L.append("")
    L.append("### 胜率 vs 收益 双口径（D27 D 级，强制同报）")
    L.append("")
    if wr_test is not None:
        sig_corr = (np.isfinite(wr_test.get('t_hac', np.nan))
                    and abs(wr_test['t_hac']) > 1.96)
        L.append(f"- 胜率 lift（逐日配对，**固定 Top{topk} 口径**）**{wr_test['lift']*100:+.2f}pp**，n={wr_test['n']} 日")
        L.append(f"　　朴素 t={wr_test['t']:+.2f}（p={wr_test['p']:.4f}）→ "
                 f"{'显著' if wr_test['p']<0.05 else '不显著'}"
                 f"　|　**HAC(19) t={wr_test['t_hac']:+.2f}**　|　块bootstrap(20) t={wr_test['t_block']:+.2f}"
                 f"　⇒ 以修正口径为准：**{'显著' if sig_corr else '不显著'}**")
        L.append("　　（① 标签为「后 20 个交易日」，相邻日前瞻窗重叠 ⇒ 朴素 t 会高估显著性（三.69）")
        L.append("　　② **原 UP/DOWN 口径已废弃**：每日判涨人数 5~53 剧烈波动 ⇒ lift 方差随日而异（异方差），"
                 "标准误修正失效；现固定取 TopK 只，分母固定")
    if 'c' in dir():
        L.append(f"- 收益 lift（折聚类）**{lift*100:+.1f}pp**，CI [{c['ir_lo']:.2f},{c['ir_hi']:.2f}] → "
                 f"{'显著' if c['ir_lo']>0 else '不显著'}")
    L.append(f"- {dual_note}" if dual_note else "")
    L.append("")
    L.append("### 选股能力（部署口径 IC —— ⓪ 同一性闸第⑤问，lessons 三.73）")
    L.append("")
    from ml_services.overlap_stats import hac_t as _hac_t
    _m, _n, _se, _t = _hac_t(cap['ic_series'], lags=horizon - 1)
    L.append("- 部署打分 = **raw prob**（`selection_score`，D73）；"
             f"下列 IC 全部用**同一**打分逐日计算")
    L.append(f"- **部署口径 IC（逐日 Spearman，n={cap['n_days']} 日）**："
             f"均值 **{cap['ic']:+.4f}** / 中位 {cap['ic_med']:+.4f}　"
             f"HAC({horizon-1}) t={_t:+.2f} → **{'显著' if abs(_t) > 1.96 else '不显著'}**")
    L.append(f"- 对照①同一逐日口径、换用**原始 prob**：{cap['ic_raw']:+.4f}（打分变换之差）")
    L.append(f"- 对照②同一「逐日」、换**线性**相关：{cap['ic_lin']:+.4f}"
             f"（rank 与线性可**异号** ⇒ 该量在噪声水平时本身就统计量依赖）")
    L.append(f"- 对照③**全表跨日池化** Spearman：{cap['ic_pooled']:+.4f}"
             f"（含日间共同层 ⇒ 天然偏高，**禁用于能力评估**，三.68）")
    L.append("- ⚠️ 这四个数**不是同一个量**；能力评估只能读**部署口径那一行**，"
             "引用任何 IC 必须标注口径（D29 D 级）。")
    L.append("")
    L.append(f"### PBO 读法（lessons 三.63，{len(M)} 期 × {M.shape[1]} 配置）")
    L.append("")
    L.append(f"- {pbo_read}")
    L.append(f"- ⚠️ 零假设带是 **seed=0 的单次抽样**（{null_perm} 次重排）。边界的具体数值不可当精确阈值；"
             f"稳健的是「在带内 / 带外」这个**分类**（D26 实测跨 seed 与跨 n_perm 均落在带内）。")
    L.append(f"- 六个配置全期夏普极差 **{spread:.3f}**"
             + ("（过小 ⇒ 配置互不可分 ⇒ PBO 必然偏高，先看别的指标）" if spread < 0.2 else ""))
    L.append("- **PBO 必须与零假设带同报**；单看一个数字会把噪声读成过拟合，"
             "也会把本窗口的偶然读成改善。")

    md = "\n".join(L)
    print(md)
    if out_md:
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"✅ 报告已保存: {out_md}")
    return passed


def main():
    ap = argparse.ArgumentParser(description='月度护栏（净IR/PBO/DSR/lift）')
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--topk', type=int, default=10)
    ap.add_argument('--cost', type=float, default=COST)
    ap.add_argument('--pred', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    ap.add_argument('--market', type=str, default='hk', choices=['hk', 'a'],
                    help='市场：hk=港股（默认），a=A股（A_STOCK_REFORM_PLAN P2.1）')
    ap.add_argument('--null-perm', type=int, default=200,
                    help='PBO 零假设带的重排次数；0=不算（会退回「单看一个 PBO 数字」的坏用法）。'
                         '默认 200：5/95 分位由样本顺序统计量决定，10 次时该分位只由第 1 与第 10 个'
                         '观测决定，边际极不稳定（实测 D26 的下界跨 seed 在 0.71–0.83 间跳）。')
    ap.add_argument('--dropout', type=int, default=0,
                    help='TopK-Dropout：每期只剔除分数最低的 N 只、保留其余旧持仓再补新（降换手）。'
                         '0=全换（默认）。')
    ap.add_argument('--sweep', action='store_true',
                    help='换手率扫描：对 dropout∈{0,2,4,6,8} 各跑一次，输出'
                         '「换手 / 净IR / 毛IR / 超额IR / 成本拖累」对照表，量化降换手的上限。')
    ap.add_argument('--sensitivity', action='store_true',
                    help='相邻参数敏感性（默认应跑）：topk∈{-5,0,+5} × dropout∈{0,2}，报净/超额 IR，'
                         '看**超额 IR 符号是否跨参一致**；翻转=不稳健（lessons 三.79）。')
    args = ap.parse_args()

    pred = args.pred or latest_pred(args.horizon, market=args.market)
    if not pred:
        print("❌ 未找到 prediction_analysis.csv，请用 --pred 指定")
        sys.exit(1)
    suffix = '_a' if args.market == 'a' else ''

    if args.sweep:
        from ml_services.portfolio_backtest import load_panel as _load  # 复用同一加载
        df = _load(pred, market=args.market)
        print(f"\n# 换手率扫描（horizon={args.horizon}d, topk={args.topk}, cost={args.cost}）")
        print(f"{'dropout':>7} | {'换手':>6} | {'净IR':>7} | {'毛IR':>7} | {'超额IR':>7} | {'成本拖累/期':>10}")
        print("-" * 60)
        for d in (0, 2, 4, 6, 8):
            bt = backtest(df, args.horizon, args.topk,
                          cost=args.cost, dropout=d)
            tn = float(bt['turnover'].mean())
            st_net = _stats(bt['top_net'].values, args.horizon)
            st_gross = _stats(bt['top_gross'].values, args.horizon)
            st_exc = _stats(bt['top_net'].values - bt['bench'].values, args.horizon)
            cd = tn * args.cost
            print(f"{d:>7} | {tn*100:>5.0f}% | {st_net['ir']:>7.2f} | {st_gross['ir']:>7.2f} "
                  f"| {st_exc['ir']:>7.2f} | {cd*100:>9.3f}%")
        print("\n解读：降换手只收复「成本拖累」那一行；若毛IR≈0（无信号），净IR 永为正也救不了 edge。")
        return

    if args.sensitivity:
        from ml_services.portfolio_backtest import load_panel as _load  # 复用同一加载
        df = _load(pred, market=args.market)
        print(f"\n# 相邻参数敏感性（horizon={args.horizon}d, cost={args.cost}）")
        print(f"{'topk':>5}{'dropout':>9} | {'净IR':>7}{'超额IR':>9}{'换手':>7}")
        print("-" * 44)
        signs = []
        for tk in sorted({max(3, args.topk - 5), args.topk, args.topk + 5}):
            for dp in (0, 2):
                bt = backtest(df, args.horizon, tk, cost=args.cost, dropout=dp)
                st_net = _stats(bt['top_net'].values, args.horizon)
                st_exc = _stats(bt['top_net'].values - bt['bench'].values, args.horizon)
                signs.append(np.sign(st_exc['ir']))
                print(f"{tk:>5}{dp:>9} | {st_net['ir']:>7.2f}{st_exc['ir']:>9.2f}{bt['turnover'].mean()*100:>6.0f}%")
        consistent = len(set(signs)) == 1
        print("-" * 44)
        print(f"判读：超额 IR 符号 **{'一致（较稳）' if consistent else '跨参翻转 ⇒ 不稳健'}**"
              f"（出现 {sorted({int(s) for s in signs})}）—— 符号翻转 = 「取最好一轮」陷阱（lessons 三.79）。")
        return

    out = (args.output
           or f"output/monthly_guardrail_{args.horizon}d_{datetime.now():%Y%m%d}{suffix}.md")
    run(args.horizon, pred, args.topk, args.cost, out, market=args.market,
        null_perm=args.null_perm, dropout=args.dropout)


if __name__ == '__main__':
    main()