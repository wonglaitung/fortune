#!/usr/bin/env python3
"""Walk-forward 重跑进度监控：每 INTERVAL 秒解析三个日志，写入 wf_monitor.log。"""
import os, time, re, datetime

LOG_DIR = "/data/fortune/output"
LOGS = {
    "20d": os.path.join(LOG_DIR, "wf20d_fix.log"),
    "5d": os.path.join(LOG_DIR, "wf5d_fix.log"),
    "1d": os.path.join(LOG_DIR, "wf1d_fix.log"),
}
MON = os.path.join(LOG_DIR, "wf_monitor.log")
INTERVAL = 900  # 15 分钟


def parse_progress(path):
    if not os.path.exists(path):
        return "无日志"
    try:
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return "读失败"
    text = "".join(lines[-400:])
    # 找 fold 进度：常见 "Fold 3/86" 或 "折叠" 或 "fold 3"
    m = re.findall(r"[Ff]old\s*(\d+)\s*/\s*(\d+)", text)
    fold = f"{m[-1][0]}/{m[-1][1]}" if m else "?"
    # 错误检测
    err = "ERROR" in text or "Traceback" in text or "error:" in text
    # 最后一行
    last = lines[-1].strip() if lines else ""
    return f"fold={fold} err={err} | {last[:120]}"


def main():
    while True:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"=== {ts} ==="]
        for k, p in LOGS.items():
            parts.append(f"[{k}] {parse_progress(p)}")
        # 检查是否全部完成
        done = all(
            os.path.exists(p) and (
                "DONE_20D" in open(p, errors="ignore").read() if os.path.exists(p) else False
            )
            for k, p in LOGS.items()
        )
        with open(MON, "a", errors="ignore") as f:
            f.write("\n".join(parts) + "\n")
        if done:
            with open(MON, "a", errors="ignore") as f:
                f.write(f"=== ALL DONE @ {ts} ===\n")
            break
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
