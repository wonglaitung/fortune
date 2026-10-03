#!/usr/bin/env python3
"""回测产物入库：提交 prediction_analysis.csv + 同步 GATE_SNAPSHOT（分位门槛数据源）。

由 walk_forward_validation.py 在验证成功后自动调用，也可手动执行：

    python3 scripts/commit_backtest_result.py output/20260925_044407_catboost_20d [--no-push]

职责：
1. 用新回测 CSV 计算分位门槛（bear=P92 / weak=P90）
2. 同步 ml_services/market_regime.py 的 GATE_SNAPSHOT / GATE_SNAPSHOT_AS_OF 常量
3. git add 本目录 CSV + market_regime.py；将仓库中其它港股 20d 回测 CSV
   移出跟踪（工作树保留，控制仓库体积——只保留"最新"语义）
4. commit（[skip ci]）并 push（--no-push 跳过）

任何失败只打印 WARNING、exit 0，绝不影响回测结果。
"""
import argparse
import os
import re
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

SNAPSHOT_FILE = os.path.join(BASE_DIR, 'ml_services', 'market_regime.py')
SNAPSHOT_RE = re.compile(r'^GATE_SNAPSHOT = \{.*?\}$', re.M)
AS_OF_RE = re.compile(r"^GATE_SNAPSHOT_AS_OF = '[^']*'", re.M)
# 仅港股 20d 目录（要求紧邻 output/<时间戳>_catboost_20d/，排除 a_stock 与嵌套层级）
HK20D_RE = re.compile(r'^output/\d{8}_\d{6}_catboost_20d/prediction_analysis\.csv$')

# 作废目录黑名单（2026-10-03 新增）
# ------------------------------------------------------------------
# 事故记录：实验「港股训练窗 36→12」跑完后，其产物目录被本脚本自动入库，
# 替换掉了可信基线。但该实验已按**预注册判据判为「排除」**（同测试期对齐后
# 12月窗更差，净IR 0.42→1.06/PBO 0.64→0.09 实为样本混淆，见 lessons 三.40）。
# 而该 CSV 正是**分位门槛的数据源**（GATE_SOURCE_CSV 默认取最新港股 CSV），
# 一旦入库，会成为后续所有港股回测的门槛基准 → 污染传递。
#
# 本脚本按「目录」自动入库，无法知道某轮是正式基线还是作废实验，
# 故必须靠显式黑名单。任何实验产物入库前，先把目录名登记到这里。
VOIDED_DIRS = {
    '20261003_214205_catboost_20d',   # 实验1 训练窗36→12，已判排除
    # Top30 降噪实验（--top-k 30）：单变量对照显示 PBO 0.41→0.54 越线，
    # 路径已关闭（lessons 三.37）。**它不是基线**，勿作门槛源。
    '20261002_173214_catboost_20d',
}


def is_voided(rel_dir: str) -> bool:
    d = os.path.basename(os.path.normpath(rel_dir))
    return d in VOIDED_DIRS or os.path.normpath(rel_dir) in VOIDED_DIRS


def update_snapshot_source(source: str, gates: dict, as_of: str) -> str:
    """把 gates/as_of 写入 market_regime.py 源文本的两个常量（纯函数，可单测）。"""
    snap_line = 'GATE_SNAPSHOT = ' + repr(gates)
    as_of_line = f"GATE_SNAPSHOT_AS_OF = '{as_of}'"
    if not SNAPSHOT_RE.search(source):
        raise ValueError('GATE_SNAPSHOT 定义未找到')
    if not AS_OF_RE.search(source):
        raise ValueError('GATE_SNAPSHOT_AS_OF 定义未找到')
    source = SNAPSHOT_RE.sub(lambda m: snap_line, source, count=1)
    source = AS_OF_RE.sub(lambda m: as_of_line, source, count=1)
    return source


def select_csv_to_untrack(current_csv: str, tracked_files: list) -> list:
    """仓库中需移出跟踪的旧港股 20d CSV（排除当前这份）。

    Args:
        current_csv: 当前回测 CSV 的**仓库相对路径**（与 tracked_files 同口径）
        tracked_files: git ls-files 输出（仓库相对路径）
    """
    cur = os.path.normpath(current_csv)
    return [f for f in tracked_files
            if HK20D_RE.search(f.replace(os.sep, '/'))
            and os.path.normpath(f) != cur]


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True, **kw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('detail_dir', help='回测输出目录（含 prediction_analysis.csv）')
    parser.add_argument('--no-push', action='store_true', help='只 commit 不 push')
    args = parser.parse_args()

    try:
        if is_voided(args.detail_dir):
            print(f"WARNING: 回测入库跳过，**{args.detail_dir} 已在作废黑名单中**")
            print("         作废原因见 VOIDED_DIRS 注释与 lessons.md；")
            print("         若该实验已翻案，请先从黑名单移除再入库。")
            return 0

        csv_path = os.path.join(BASE_DIR, args.detail_dir, 'prediction_analysis.csv')
        if not os.path.exists(csv_path):
            print(f"WARNING: 回测入库跳过，未找到 {csv_path}")
            return 0

        rel_csv = os.path.relpath(csv_path, BASE_DIR)
        # 仅港股时间戳目录（排除 *_a_stock_catboost_20d）才算 20d 分位语义；
        # 5d/1d/A股 目录只提交 CSV，不更新 GATE_SNAPSHOT
        is_20d = bool(HK20D_RE.match(rel_csv.replace(os.sep, '/')))

        gates = None
        if is_20d:
            import pandas as pd
            from ml_services.market_regime import compute_gate_thresholds
            df = pd.read_csv(csv_path, usecols=['Date'])
            as_of = str(df['Date'].max())[:10]
            gates = compute_gate_thresholds(as_of=as_of, source_csv=csv_path)
            print(f"分位门槛（as_of={as_of}）: {gates}")

            with open(SNAPSHOT_FILE, 'r', encoding='utf-8') as f:
                old_src = f.read()
            new_src = update_snapshot_source(old_src, gates, as_of)
            if new_src != old_src:
                with open(SNAPSHOT_FILE, 'w', encoding='utf-8') as f:
                    f.write(new_src)
                print(f"GATE_SNAPSHOT 已同步 -> {SNAPSHOT_FILE}")

        r = _run(['git', 'ls-files', 'output/'])
        if r.returncode != 0:
            raise RuntimeError(f'git ls-files 失败: {r.stderr}')
        tracked = r.stdout.split()
        untrack = select_csv_to_untrack(rel_csv, tracked) if is_20d else []

        add_cmd = ['git', 'add', rel_csv]
        if is_20d:
            add_cmd.append(os.path.relpath(SNAPSHOT_FILE, BASE_DIR))
        _run(add_cmd)
        for f in untrack:
            _run(['git', 'rm', '--cached', '-q', f])
        if untrack:
            print(f"移出跟踪（工作树保留）: {len(untrack)} 个旧 CSV")

        dirname = os.path.basename(os.path.dirname(csv_path))
        if gates:
            msg = (f"[skip ci] 回测入库: {dirname} "
                   f"+ GATE_SNAPSHOT 同步 bear={gates['bear']:.4f} weak={gates['weak']:.4f}")
        else:
            msg = f"[skip ci] 回测入库: {dirname}"
        # 注意：不可用 `git commit -- <paths>`（会从工作树取内容，把 rm --cached
        # 取消跟踪的旧 CSV 重新提交）；只能提交 index。若用户预先 stage 了其它
        # 文件会被一并提交，这里仅告警。
        commit_paths = set([rel_csv] + untrack +
                           ([os.path.relpath(SNAPSHOT_FILE, BASE_DIR)] if is_20d else []))
        staged = _run(['git', 'diff', '--cached', '--name-only']).stdout.split()
        others = [p for p in staged if p not in commit_paths]
        if others:
            print(f"WARNING: 以下已暂存文件将随本次提交一并入库: {others}")
        r = _run(['git', 'commit', '-m', msg])
        if r.returncode != 0 and 'nothing to commit' not in (r.stdout + r.stderr):
            raise RuntimeError(f'git commit 失败: {r.stderr}')
        print(r.stdout.strip() or r.stderr.strip())

        if not args.no_push:
            r = _run(['git', 'push'], timeout=300)
            if r.returncode != 0:
                # 远程有新提交：rebase 拉取后重试一次（autostash 保护本地 data 修改）
                print("push 被拒，自动 pull --rebase 重试...")
                rp = _run(['git', 'pull', '--rebase', '--autostash'], timeout=300)
                if rp.returncode != 0:
                    raise RuntimeError(f'pull --rebase 失败: {rp.stderr}')
                r = _run(['git', 'push'], timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f'git push 失败: {r.stderr}')
            print("已推送")
        return 0
    except Exception as e:
        print(f"WARNING: 回测入库失败（不影响回测结果）: {e}")
        return 0


if __name__ == '__main__':
    sys.exit(main())
