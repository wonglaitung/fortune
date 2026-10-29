#!/usr/bin/env python3
"""M1 预注册实验：A-H 溢价均值回归（时序）

严格按 ml_services/prereg_M1.md 执行，不得中途改判据。
产出 output/ah_premium/ah_m1_*.csv
"""
import glob
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
DATA = os.path.join(REPO, "data", "ah_premium")
OUT = os.path.join(REPO, "output", "ah_premium")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, REPO)
from ml_services.eval_overfit import cscv_pbo, deflated_sharpe  # noqa: E402

from ml_services.costs import COST_ROUND_TRIP
HORIZON = 21          # 月频 → 年化 sqrt(252/21)=sqrt(12)
N_TRIALS = 6          # 配置族数（预注册 §4）

# 长窗 10 对 = 主判据；短窗 3 对（A 股 2021/22 才 IPO）= 附录
LONG = {"601398", "600036", "601939", "601318", "601628",
        "601088", "601601", "601288", "601633", "002594"}
SHORT = {"601728", "600941", "600938"}


def load_pairs():
    out = {}
    for f in sorted(glob.glob(os.path.join(DATA, "ah_*.csv"))):
        ac = os.path.basename(f).split("_")[1]
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        out[ac] = df
    return out


def month_end_index(df, win_months=12):
    """返回月末调仓日（每个自然月最后一个共同交易日），并给出滚动窗统计（仅用 t-1 及更早）"""
    s = df["premium_pit"]
    me = s.groupby([s.index.year, s.index.month]).apply(lambda x: x.index[-1])
    me = me[~me.duplicated(keep="last")].sort_values()
    return me


def build_signal(df, win=12, normalize=True):
    """z_t = (p_t - mean(p_{t-win..t-1}))/std(...)；滚动窗严格只用 t-1 及更早（PIT）"""
    me = month_end_index(df)
    p = df["premium_pit"].reindex(me)
    if normalize:
        mu = p.shift(1).rolling(win, min_periods=max(6, win // 2)).mean()
        sd = p.shift(1).rolling(win, min_periods=max(6, win // 2)).std()
    else:
        mu = pd.Series(p.mean(), index=p.index)
        sd = pd.Series(p.std(), index=p.index)
    z = (p - mu) / sd.replace(0, np.nan)
    return me, z


def fwd_return(df, me):
    """下一月 H 腿收益：close(t) → close(next month-end)"""
    h = df["H_hkd"]
    base = h.reindex(me[:-1].values)
    nxt = h.reindex(me[1:].values)
    fwd = (nxt.values / base.values) - 1.0
    return pd.Series(fwd, index=me[:-1])


def run_config(pairs, codes, win=12, normalize=True, long_only=True):
    """返回逐月组合收益序列（等权跨 pairs）"""
    port = []
    for t in sorted(set().union(*[set(build_signal(pairs[c])[0]) for c in codes])):
        ws, fs = [], []
        for c in sorted(codes):  # 固定遍历序 → 求和顺序确定 → 消除末位漂移
            df = pairs[c]
            me, z = build_signal(df, win, normalize)
            if t not in z.index or not np.isfinite(z.loc[t]):
                continue
            fr = fwd_return(df, me)
            if t not in fr.index or not np.isfinite(fr.loc[t]):
                continue
            s = -np.sign(z.loc[t])
            w = max(0.0, s) if long_only else s      # 锁定可交易性：多头倾斜为准
            ws.append(w)
            fs.append(fr.loc[t])
        if ws:
            port.append((t, np.dot(ws, fs) / max(sum(ws), 1e-9), np.mean(fs),
                         float(np.mean(ws))))
    return pd.DataFrame(port, columns=["date", "ret_bh", "ret_ew_h", "exposure"]).set_index("date")


def stats(x, horizon=HORIZON, cost=0.0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return dict(n=len(x), mean=np.nan, ir=np.nan, win=np.nan, cum=np.nan)
    m = x.mean() - cost
    sd = x.std(ddof=1)
    ir = m / sd * np.sqrt(252.0 / horizon) if sd > 0 else np.nan
    return dict(n=len(x), mean=float(m), ir=float(ir),
                win=float((x > 0).mean()), cum=float(np.prod(1 + x - cost) - 1))


def main():
    pairs = load_pairs()
    print(f"loaded pairs: {len(pairs)}  (long={len(LONG)}, short={len(SHORT)})")

    # ---- 配置族（时序口径：窗长 × 标准化）----
    configs = {}
    for win in (6, 12, 24):
        for nz in (True, False):
            r = run_config(pairs, LONG, win=win, normalize=nz)
            if len(r):
                configs[(win, nz)] = r["ret_bh"]
    M = pd.DataFrame(configs).dropna()
    print(f"config matrix M: {M.shape}")

    # ---- 主判据：预注册 12 月窗 + z标准化 + 多头倾斜 ----
    R = run_config(pairs, LONG, win=12, normalize=True)
    bh = run_config(pairs, LONG, win=12, normalize=True, long_only=False)
    bench = R["ret_ew_h"]        # 等权买入持有 H
    raw = R["ret_bh"]
    turn = R["exposure"].diff().abs().fillna(R["exposure"].iloc[0] * 0.5)
    net = raw - 0.5 * COST_ROUND_TRIP * turn      # 成本按换手（|Δw| 是单边，D97）

    s = stats(raw)
    sn = stats(net.values, cost=0.0)
    sb = stats(bench)
    diff = (net - bench).dropna()
    tstat = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) > 2 else np.nan

    from scipy import stats as st
    pval = float(2 * (1 - st.t.cdf(abs(tstat), len(diff) - 1))) if np.isfinite(tstat) else np.nan

    sr_net = float(net.mean() / net.std(ddof=1) * np.sqrt(252 / HORIZON))
    pbo, _ = cscv_pbo(M.values)
    dsr = deflated_sharpe(sr_net, len(net), N_TRIALS, net.var(ddof=1), float(st.skew(net)), float(st.kurtosis(net)))

    # ---- A 闸 ① 绝对值异常：信号与前瞻收益不得逼近 1 ----
    zs, fs = [], []
    for c in sorted(LONG):
        me, z = build_signal(pairs[c], 12, True)
        zs.append(z.rename(c))
        fs.append(fwd_return(pairs[c], me).rename(c))
    Zp, Fp = pd.concat(zs, axis=1), pd.concat(fs, axis=1)
    common = Zp.index.intersection(Fp.index)
    Zp, Fp = Zp.loc[common], Fp.loc[common]
    ok = Zp.notna() & Fp.notna()
    cc = float(np.corrcoef(Zp.to_numpy()[ok], Fp.to_numpy()[ok])[0, 1])
    per_pair = {c: float(np.corrcoef(Zp[c][ok[c]], Fp[c][ok[c]])[0, 1])
                for c in sorted(LONG) if ok[c].sum() > 30}

    res = dict(
        n_periods=int(len(net)), n_pairs=len(LONG),
        gross_ir=s["ir"], net_ir=sn["ir"], net_mean=sn["mean"],
        net_win=sn["win"], net_cum=sn["cum"],
        bench_ir=sb["ir"], bench_mean=sb["mean"],
        lift_mean=float(diff.mean()), lift_t=tstat, lift_p=pval,
        pbo=float(pbo), dsr=float(dsr), sr_net=sr_net,
        signal_fwd_corr=cc, per_pair_corr=per_pair,
        avg_exposure=float(R["exposure"].mean()),
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))

    R.round(12).to_csv(os.path.join(OUT, "ah_m1_monthly_long10.csv"))
    M.to_csv(os.path.join(OUT, "ah_m1_config_matrix.csv"))
    with open(os.path.join(OUT, "ah_m1_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    # 短窗附录
    Rs = run_config(pairs, SHORT, win=12, normalize=True)
    if len(Rs):
        turns = Rs["exposure"].diff().abs().fillna(Rs["exposure"].iloc[0] * 0.5)
        nets_s = Rs["ret_bh"] - 0.5 * COST_ROUND_TRIP * turns
        ss = stats(nets_s.values)
        print(f"\n[附录-短窗3对] n={ss['n']} netIR={ss['ir']:.3f} mean={ss['mean']:.5f} "
              f"cum={ss['cum']:.3f}")
        Rs.to_csv(os.path.join(OUT, "ah_m1_monthly_short3.csv"))


if __name__ == "__main__":
    main()