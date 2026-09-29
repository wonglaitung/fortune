#!/usr/bin/env python3
"""Watcher：三路 walk-forward 全部 DONE 后，自动跑评估链（backtest_eval + monthly_guardrail + portfolio_backtest）。"""
import os, glob, time, subprocess, datetime

BASE = "/data/fortune"
OUT = os.path.join(BASE, "output")
LOGS = {
    "20": os.path.join(OUT, "wf20d_fix.log"),
    "5": os.path.join(OUT, "wf5d_fix.log"),
    "1": os.path.join(OUT, "wf1d_fix.log"),
}
DONE = {"20": "DONE_20D", "5": "DONE_5D", "1": "DONE_1D"}
EVAL_LOG = os.path.join(OUT, "wf_eval_fix.log")
POLL = 120
MAX_WAIT = 8 * 3600


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(EVAL_LOG, "a", errors="ignore") as f:
        f.write(line + "\n")


def all_done():
    for k, path in LOGS.items():
        if not os.path.exists(path):
            return False
        with open(path, errors="ignore") as f:
            if DONE[k] not in f.read():
                return False
    return True


def latest_dir(horizon):
    pats = glob.glob(os.path.join(OUT, f"*_catboost_{horizon}d"))
    if not pats:
        return None
    return max(pats, key=os.path.getmtime)


def run(cmd):
    log("RUN: " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=BASE, capture_output=True, text=True, timeout=1800)
        out = r.stdout + "\n" + r.stderr
        with open(EVAL_LOG, "a", errors="ignore") as f:
            f.write(out + "\n" + "=" * 60 + "\n")
        log(f"exit={r.returncode}")
        return r.returncode == 0
    except Exception as e:
        log(f"EXC: {e}")
        return False


def main():
    log("watcher start")
    waited = 0
    while not all_done() and waited < MAX_WAIT:
        time.sleep(POLL)
        waited += POLL
        if waited % 600 == 0:
            log(f"waiting... {waited//60} min elapsed")
    if not all_done():
        log("TIMEOUT: 三路未全部完成")
        return

    log("三路全部完成，开始评估链")
    for h in ["20", "5", "1"]:
        d = latest_dir(h)
        log(f"[{h}d] output dir = {d}")
        if not d:
            log(f"[{h}d] 未找到输出目录，跳过")
            continue
        csv = os.path.join(d, "prediction_analysis.csv")
        if not os.path.exists(csv):
            log(f"[{h}d] 缺少 prediction_analysis.csv，跳过")
            continue
        run(["python3", "ml_services/backtest_eval.py",
             "--input", csv, "--horizon", h, "--output", os.path.join(d, "backtest_eval_fix.md")])
        run(["python3", "ml_services/monthly_guardrail.py",
             "--horizon", h, "--pred", csv])
        if h in ("5", "20"):
            run(["python3", "ml_services/portfolio_backtest.py",
                 "--horizon", int(h), "--topk", "10", "--pred", csv])
    log("评估链完成")


if __name__ == "__main__":
    main()
