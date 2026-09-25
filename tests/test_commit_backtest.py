# -*- coding: utf-8 -*-
"""回测自动入库工具测试：
- update_snapshot_source: GATE_SNAPSHOT/AS_OF 正则替换（纯函数）
- select_csv_to_untrack: 只清旧港股 20d CSV（排除 a_stock/嵌套/5d/当前）
- 端到端：临时 git 仓库中工具完成 commit 且不 push
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import commit_backtest_result as cbr


SRC = '''GATE_QUANTILES = {'bear': 0.92, 'weak': 0.90}
# 注释行
GATE_SNAPSHOT = {'bear': 0.111, 'weak': 0.222}
GATE_SNAPSHOT_AS_OF = '2026-01-01'
GATE_FALLBACK = {'bear': 0.70, 'weak': 0.65}
'''


def test_update_snapshot_source():
    out = cbr.update_snapshot_source(SRC, {'bear': 0.6923, 'weak': 0.6667}, '2026-09-25')
    assert "GATE_SNAPSHOT = {'bear': 0.6923, 'weak': 0.6667}" in out
    assert "GATE_SNAPSHOT_AS_OF = '2026-09-25'" in out
    assert "GATE_FALLBACK = {'bear': 0.70, 'weak': 0.65}" in out  # 其它行不动
    assert 'GATE_QUANTILES' in out


def test_update_snapshot_source_missing_constant():
    with pytest.raises(ValueError):
        cbr.update_snapshot_source('x = 1\n', {'bear': 0.5}, '2026-01-01')


def test_select_csv_to_untrack():
    tracked = [
        'output/20260524_220824_catboost_20d/prediction_analysis.csv',   # 旧港股20d → 清
        'output/20260925_044407_catboost_20d/prediction_analysis.csv',   # 当前 → 留
        'output/20260722_181211_a_stock_catboost_20d/prediction_analysis.csv',  # A股 → 留
        'output/20260922_212530_catboost_5d/prediction_analysis.csv',    # 5d → 留
        'output/walk_forward_x/20260524_003643_catboost_20d/prediction_analysis.csv',  # 嵌套 → 留
        'output/any.md',
    ]
    cur = 'output/20260925_044407_catboost_20d/prediction_analysis.csv'
    out = cbr.select_csv_to_untrack(cur, tracked)
    assert out == ['output/20260524_220824_catboost_20d/prediction_analysis.csv']


def test_end_to_end_tmp_git_repo(tmp_path):
    """临时仓库：工具完成 CSV+market_regime 提交、清旧 20d CSV、不 push"""
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    (repo / 'ml_services').mkdir()
    # 拷贝工具与 market_regime
    here = Path(__file__).resolve().parent.parent
    (repo / 'scripts' / 'commit_backtest_result.py').write_text(
        (here / 'scripts' / 'commit_backtest_result.py').read_text(encoding='utf-8'),
        encoding='utf-8')
    (repo / 'ml_services' / '__init__.py').write_text('')
    (repo / 'ml_services' / 'market_regime.py').write_text(
        (here / 'ml_services' / 'market_regime.py').read_text(encoding='utf-8'),
        encoding='utf-8')
    # 回测目录（当前 + 旧港股20d）——250 行保证样本量过 GATE_MIN_SAMPLES=200
    cur = repo / 'output' / '20260926_120000_catboost_20d'
    cur.mkdir(parents=True)
    rows = ['Date,Predict_Prob']
    import pandas as pd
    for d in pd.bdate_range('2026-01-01', periods=250):
        rows.append(f'{d:%Y-%m-%d},0.5')
    (cur / 'prediction_analysis.csv').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    old = repo / 'output' / '20260524_220824_catboost_20d'
    old.mkdir(parents=True)
    (old / 'prediction_analysis.csv').write_text('Date,Predict_Prob\n', encoding='utf-8')

    def git(*a):
        return subprocess.run(['git', *a], cwd=repo, capture_output=True, text=True, check=True)

    git('init', '-q')
    git('config', 'user.email', 't@t'); git('config', 'user.name', 't')
    git('add', '-A')
    git('commit', '-qm', 'init')

    r = subprocess.run(
        [sys.executable, str(repo / 'scripts' / 'commit_backtest_result.py'),
         'output/20260926_120000_catboost_20d', '--no-push'],
        cwd=repo, capture_output=True, text=True, timeout=120)
    print(r.stdout, r.stderr)
    assert r.returncode == 0

    # 当前 CSV 已入库，旧港股 20d CSV 移出跟踪但文件仍在
    ls = subprocess.run(['git', 'ls-files'], cwd=repo, capture_output=True, text=True)
    assert '20260926_120000_catboost_20d/prediction_analysis.csv' in ls.stdout
    assert '20260524_220824_catboost_20d/prediction_analysis.csv' not in ls.stdout
    assert (old / 'prediction_analysis.csv').exists()
    # market_regime 常量已同步（as_of 取 CSV 最大日期）
    mr = (repo / 'ml_services' / 'market_regime.py').read_text(encoding='utf-8')
    expected_asof = str(pd.bdate_range('2026-01-01', periods=250).max())[:10]
    assert f"GATE_SNAPSHOT_AS_OF = '{expected_asof}'" in mr
    # 提交信息
    log = subprocess.run(['git', 'log', '-1', '--pretty=%s'], cwd=repo,
                         capture_output=True, text=True)
    assert '[skip ci] 回测入库' in log.stdout

    # 5d 目录：只提交 CSV，不更新 GATE_SNAPSHOT（20d 语义）
    mr_before = (repo / 'ml_services' / 'market_regime.py').read_text(encoding='utf-8')
    d5 = repo / 'output' / '20260926_130000_catboost_5d'
    d5.mkdir(parents=True)
    (d5 / 'prediction_analysis.csv').write_text(
        'Date,Predict_Prob\n2026-09-01,0.5\n', encoding='utf-8')
    r5 = subprocess.run(
        [sys.executable, str(repo / 'scripts' / 'commit_backtest_result.py'),
         'output/20260926_130000_catboost_5d', '--no-push'],
        cwd=repo, capture_output=True, text=True, timeout=120)
    assert r5.returncode == 0
    assert (repo / 'ml_services' / 'market_regime.py').read_text(encoding='utf-8') == mr_before
    ls5 = subprocess.run(['git', 'ls-files'], cwd=repo, capture_output=True, text=True)
    assert '20260926_130000_catboost_5d/prediction_analysis.csv' in ls5.stdout
    log5 = subprocess.run(['git', 'log', '-1', '--pretty=%s'], cwd=repo,
                          capture_output=True, text=True)
    assert 'GATE_SNAPSHOT' not in log5.stdout


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
