"""性能报告生成测试：三周期免责 / 港股-only契约 / 小样本注 / 诚实摘要前置。

背景（2026-09-26）：性能报告曾给出三周期"分批建仓"类可执行建议，
其数据为重叠窗口生产历史（未 embargo），与严格验证结论镜像反转，
违反 DECISIONS D3 与 lessons 0.2 —— 本测试锁定修复后的报告契约。
2026-09-27 起报告只分析港股（A 股评估与市场分布节停用）。
"""
import pytest

from ml_services.performance_monitor import (
    assemble_report,
    generate_monthly_report,
)


def _pred(**kw):
    base = {
        'data_date': '2026-09-01',
        'stock_code': '0700.HK',
        'stock_name': '腾讯控股',
        'name': '腾讯控股',
        'sector': 'tech',
        'market': 'HK',
        'horizon': 20,
        'predicted_direction': 'up',
        'outcome': 'correct',
        'actual_return': 0.02,
    }
    base.update(kw)
    return base


def _three_horizon_history():
    """两条三周期齐全的记录（111 模式）+ 一条未到期记录。"""
    preds = []
    for d in ['2026-09-01', '2026-09-05']:
        for h in (1, 5, 20):
            preds.append(_pred(data_date=d, horizon=h, predicted_direction='up'))
    preds.append(_pred(data_date='2026-09-25', horizon=1, outcome=None))
    return {'predictions': preds}


def test_pattern_section_has_disclaimer_no_actions():
    """三周期节：有免责 + 无"建议"列 + 无具体操作词 + 列名降级。"""
    report = generate_monthly_report(_three_horizon_history())
    assert '不构成交易依据' in report
    assert '未 embargo' in report
    assert '表面准确率' in report
    assert '| 建议 |' not in report
    assert '分批建仓' not in report
    assert '谨慎减仓' not in report


def test_report_is_hk_only():
    """报告只分析港股：标题标港股、无市场分布节、混入的 A 股记录被剔除。"""
    mixed = {'predictions': [_pred(market='HK'), _pred(market='A', stock_code='600000', stock_name='浦发银行')]}
    report = generate_monthly_report(mixed)
    assert '# 预测性能报告（港股）' in report
    assert '市场分布' not in report
    assert 'A股' not in report
    assert '600000' not in report


def test_small_sample_note_present():
    """板块表下必须有小样本提示（n<30 无统计意义）。"""
    report = generate_monthly_report({'predictions': [_pred()]})
    assert '小样本提示' in report


def test_honest_summary_before_section_one():
    """D3：诚实摘要（lift/方向技能）必须前置到第一节之前，护栏段保留在末尾。"""
    report, extra_html = assemble_report(
        _three_horizon_history(), guardrail_line='## 判定：🟢 通过 → 可升级')
    idx_honest = report.find('## 诚实监控摘要')
    idx_s1 = report.find('## 一、')
    assert 0 <= idx_honest < idx_s1
    assert '方向技能' in report and '超额 lift' in report
    # 护栏段在末尾，且 html 片段包含两块
    assert report.find('## 策略护栏状态') > idx_s1
    assert '方向技能' in extra_html and '策略护栏状态' in extra_html


def _fake_guardrail_md(tmp_path, verdict="## 判定：🟢 通过 → 可升级", title="# 月度护栏复核（20d）"):
    p = tmp_path / "monthly_guardrail_test.md"
    p.write_text(
        f"{title}\n\n- 日期: 2026-09-25 04:56:58\n\n"
        "| 指标 | 值 | 门槛 |\n|------|----|------|\n"
        "| 净IR | **1.06** [-0.07,2.18] | ≥0.7 |\n"
        "| **PBO** | **0.47** | <0.5 |\n"
        "| **DSR**（最优 top5-neutral） | **0.981** | ≥0.95 |\n"
        f"\n{verdict}\n",
        encoding="utf-8")
    return str(p)


def test_guardrail_block_renders(monkeypatch, tmp_path):
    """20d 护栏块：三门槛/判定/日期齐全，🟢 带配比说明。"""
    from ml_services import performance_monitor as pm
    p = _fake_guardrail_md(tmp_path)
    monkeypatch.setattr(pm, "_guardrail_file_20d", lambda: p)
    b = pm._guardrail_block()
    assert '20d 辅助策略（行业中性 TopK）护栏' in b
    assert '🟢 通过 → 可升级' in b and '15–20%' in b
    assert '1.06' in b and '0.47' in b and '0.981' in b
    assert '2026-09-25' in b and b.count('✅') == 3


def test_guardrail_block_empty_without_20d_file(monkeypatch):
    """无 20d 护栏文件 → 返回空串（宁缺毋滥，不显示错的）。"""
    from ml_services import performance_monitor as pm
    monkeypatch.setattr(pm, "_guardrail_file_20d", lambda: None)
    assert pm._guardrail_block() == ""
    assert pm._guardrail_status() is None


# ── D3 套件：block bootstrap CI / 横截面 IC / 表格主判定（2026-09-27） ──

def test_tables_primary_lift_not_accuracy():
    """四表主判定列 = 超额lift/方向技能（加粗），绝对准确率降为"(参考)"列。"""
    from datetime import datetime, timedelta
    td = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    report = generate_monthly_report({'predictions': [
        _pred(target_date=td),
        _pred(target_date=td, predicted_direction='down',
               outcome='wrong', actual_return=-0.01),
    ]})
    assert '| 超额lift | 方向技能 | 准确率(参考) |' in report
    assert '评估以 **超额lift / 方向技能** 为准' in report
    # lift/方向技能加粗（pp 格式）
    import re
    assert re.search(r'\|\s*\*\*[+-]\d+\.\d+pp\*\*', report)
    # 绝对准确率不再作为加粗主列
    assert not re.search(r'\|\s*\*\*\d+\.\d{2}%\*\*', report)


def _big_history(n_days=20, n_stocks=6):
    """合成 ≥60 条、按日分块的历史（供 bootstrap）。"""
    preds = []
    for i in range(n_days):
        d = f'2026-08-{i + 1:02d}'
        for s in range(n_stocks):
            ret = 0.02 if (i + s) % 3 else -0.01
            preds.append(_pred(
                data_date=d,
                stock_code=f'{1000 + s}.HK', horizon=20,
                predicted_direction='up', actual_return=ret,
                outcome='correct' if ret > 0 else 'wrong'))
    return {'predictions': preds}


def test_block_bootstrap_ci_deterministic():
    """≥15 日期块且 ≥60 条 → 返回 CI dict；同 seed 两次结果一致（可复现）。"""
    from ml_services.performance_monitor import _block_bootstrap_ci
    preds = _big_history()['predictions']
    r1 = _block_bootstrap_ci(preds, n_boot=200, seed=42)
    r2 = _block_bootstrap_ci(preds, n_boot=200, seed=42)
    assert r1 is not None and r2 is not None
    assert r1['lift_ci'] == r2['lift_ci']
    lo, hi = r1['lift_ci']
    assert lo <= hi
    assert r1['n_blocks'] >= 15


def test_block_bootstrap_ci_insufficient_sample():
    """样本不足（<15 日期块或 <60 条）→ None，宁可不显示也不错显示。"""
    from ml_services.performance_monitor import _block_bootstrap_ci
    assert _block_bootstrap_ci(_three_horizon_history()['predictions']) is None


def test_cross_sectional_ic():
    """同日 ≥10 只且 prob 与收益单调正相关 → mean_ic 显著为正；截面日不足 → None。"""
    from ml_services.performance_monitor import _cross_sectional_ic
    preds = []
    for i in range(16):                      # 16 个截面日
        d = f'2026-07-{(i % 28) + 1:02d}' if i < 28 else f'2026-08-{i - 27:02d}'
        for s in range(10):                  # 每日 10 只
            prob = 0.30 + 0.04 * s
            preds.append(_pred(
                data_date=d, stock_code=f'{1000 + s}.HK',
                prediction_probability=prob,
                actual_return=prob - 0.5 + 0.01))   # 单调：prob 越高 ret 越大
    ic = _cross_sectional_ic(preds)
    assert ic is not None
    assert ic['mean_ic'] > 0.9
    assert ic['n_days'] == 16
    # 截面日不足 → None
    assert _cross_sectional_ic(preds[:20]) is None


def test_pattern_ranked_by_avg_return_not_win_rate():
    """模式排名按平均收益（D3 禁绝对胜率排名）：收益高的 111 排第一。"""
    preds = []
    for h in (1, 5, 20):
        preds.append(_pred(data_date='2026-09-01', horizon=h,
                           predicted_direction='up', outcome='correct',
                           actual_return=0.03))     # 111 模式，收益高
        preds.append(_pred(data_date='2026-09-05', horizon=h,
                           predicted_direction='down', outcome='correct',
                           actual_return=-0.02))    # 000 模式，收益低
    report = generate_monthly_report({'predictions': preds})
    i111 = report.find('| 1 | 111 |')
    i000 = report.find('| 2 | 000 |')
    assert i111 > 0 and i000 > 0 and i111 < i000


def test_honest_summary_has_ci_and_ic_lines():
    """诚实摘要必须含 block bootstrap CI 与横截面 IC 行（样本不足时给降级说明）。"""
    report, extra_html = assemble_report(_big_history())
    assert 'block bootstrap 95%CI' in report
    assert '横截面 Spearman IC' in report
    assert 'block bootstrap' in extra_html or '样本不足' in extra_html
