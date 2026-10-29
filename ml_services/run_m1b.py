#!/usr/bin/env python3
"""M1b 预注册实验：A-H 溢价动量（独立窗 2022+ 短窗 3 对）

严格按 ml_services/prereg_M1b.md 执行。判据跑前定稿，一发制。
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

warnings = __import__("warnings")
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BASE)
OUT = os.path.join(REPO, "output", "ah_premium")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, REPO)
sys.path.insert(0, BASE)
from ml_services.eval_overfit import cscv_pbo, deflated_sharpe  # noqa: E402
from run_m1 import COST_ROUND_TRIP, HORIZON, N_TRIALS, load_pairs, build_signal, fwd_return  # noqa: E402

# 独立窗：短窗 3 对（A 股 2021/22 才 IPO）
SHORT = ["600938", "600941", "601728"]  # 油 / 移 / 电


def stats(x, horizon=HORIZON):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return dict(n=len(x), mean=np.nan, ir=np.nan, win=np.nan, cum=np.nan)
    m = x.mean()
    sd = x.std(ddof=1)
    return dict(n=len(x), mean=float(m), ir=float(m / sd * np.sqrt(252.0 / horizon)),
                win=float((x > 0).mean()), cum=float(np.prod(1 + x) - 1))


def momentum_config(pairs, codes, win=12, normalize=True):
    """动量：w=1 if z>0 else 0（溢价高→多 H）"""
    port = []
    for t in sorted(set().union(*[set(build_signal(pairs[c], win, normalize)[0]) for c in codes])):
        ws, fs = [], []
        for c in sorted(codes):
            df = pairs[c]
            me, z = build_signal(df, win, normalize)
            if t not in z.index or not np.isfinite(z.loc[t]):
                continue
            fr = fwd_return(df, me)
            if t not in fr.index or not np.isfinite(fr.loc[t]):
                continue
            w = 1.0 if z.loc[t] > 0 else 0.0
            ws.append(w)
            fs.append(fr.loc[t])
        if ws:
            port.append((t, np.dot(ws, fs) / max(sum(ws), 1e-9),
                         np.mean(fs), float(np.mean(ws))))
    return pd.DataFrame(port, columns=["date", "ret_bh", "ret_ew_h", "exposure"]).set_index("date")


def main():
    pairs = load_pairs()

    # ---- 配置族（PBO/DSR 多重试验）----
    cfg = {}
    for win in (6, 12, 24):
        for nz in (True, False):
            r = momentum_config(pairs, SHORT, win=win, normalize=nz)
            if len(r):
                cfg[(win, nz)] = r["ret_bh"]
    M = pd.DataFrame(cfg).dropna()
    print(f"[M1b] 独立窗短窗3对 | config M: {M.shape}")

    # ---- 主判据：预注册 12 月窗 + z + 动量 ----
    R = momentum_config(pairs, SHORT, win=12, normalize=True)
    bench = R["ret_ew_h"]
    raw = R["ret_bh"]
    turn = R["exposure"].diff().abs().fillna(R["exposure"].iloc[0] * 0.5)
    net = raw - 0.5 * COST_ROUND_TRIP * turn

    sn = stats(net.values)
    sb = stats(bench)
    diff = (net - bench).dropna()
    tstat = float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if len(diff) > 2 else np.nan
    pval = float(2 * (1 - st.t.cdf(abs(tstat), len(diff) - 1))) if np.isfinite(tstat) else np.nan

    sr_net = float(net.mean() / net.std(ddof=1) * np.sqrt(252 / HORIZON))
    pbo, _ = cscv_pbo(M.values)
    dsr = deflated_sharpe(sr_net, len(net), N_TRIALS, net.var(ddof=1),
                          float(st.skew(net)), float(st.kurtosis(net)))

    # ---- A 闸①：信号/前瞻相关（动量方向应 >0）----
    zs, fs = [], []
    for c in SHORT:
        me, z = build_signal(pairs[c], 12, True)
        zs.append(z.rename(c))
        fs.append(fwd_return(pairs[c], me).rename(c))
    Zp, Fp = pd.concat(zs, axis=1), pd.concat(fs, axis=1)
    common = Zp.index.intersection(Fp.index)
    Zp, Fp = Zp.loc[common], Fp.loc[common]
    ok = Zp.notna() & Fp.notna()
    cc = float(np.corrcoef(Zp.to_numpy()[ok], Fp.to_numpy()[ok])[0, 1])
    per_pair = {c: float(np.corrcoef(Zp[c][ok[c]], Fp[c][ok[c]])[0, 1])
                for c in SHORT if ok[c].sum() > 30}

    res = dict(n_periods=int(len(net)), n_pairs=len(SHORT),
               net_ir=sn["ir"], net_mean=sn["mean"], net_cum=sn["cum"], net_win=sn["win"],
               bench_ir=sb["ir"], bench_mean=sb["mean"],
               lift_mean=float(diff.mean()), lift_t=tstat, lift_p=pval,
               pbo=float(pbo), dsr=float(dsr), sr_net=sr_net,
               signal_fwd_corr=cc, per_pair_corr=per_pair,
               avg_exposure=float(R["exposure"].mean()))
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))

    R.round(12).to_csv(os.path.join(OUT, "ah_m1b_monthly_short3.csv"))
    with open(os.path.join(OUT, "ah_m1b_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()