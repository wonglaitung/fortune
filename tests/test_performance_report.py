"""性能报告生成测试：三周期免责 / A股显式空行 / 小样本注 / 诚实摘要前置。

背景（2026-09-26）：性能报告曾给出三周期"分批建仓"类可执行建议，
其数据为重叠窗口生产历史（未 embargo），与严格验证结论镜像反转，
违反 DECISIONS D3 与 lessons 0.2 —— 本测试锁定修复后的报告契约。
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


def test_market_section_shows_empty_a_stock():
    """A 股无评估数据时必须显式输出空行，不得静默省略。"""
    report = generate_monthly_report({'predictions': [_pred(market='HK')]})
    assert 'A股' in report
    assert '无已评估预测' in report


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
