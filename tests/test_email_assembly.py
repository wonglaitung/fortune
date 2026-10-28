"""D93 邮件编排：用 mock 数据跑完整组装，锁定章节顺序。"""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comprehensive_analysis as ca  # noqa: E402

ADVICE = '''# 综合买卖建议

## 强烈买入信号（2-3只）
1. 2318.HK 中国平安
   - 推荐理由：短期买入，中期买入，ML 分数0.62
   - 操作建议：买入
   - 建议仓位：4%
   - 价格指引：
     * 建议买入价：HK$50.00
     * 止损位：HK$46.00（-8.0%）
     * 目标价：HK$55.00（+10.0%）
   - 操作时机：当前
   - 风险提示：风险

## 买入信号（3-5只）
当前无股票同时满足条件，本期不出具该档信号。

## 持有/观望
1. 0728.HK 中国电信
   - 推荐理由：观望
   - 操作建议：观望
   - 关注要点：成交量
   - 风险提示：风险

## 卖出信号（如有）
当前无股票同时满足条件，本期不出具该档信号。

## 风险控制建议
- 当前市场整体风险：中
- 建议仓位百分比：45%
- 止损位设置：-8%
- 组合调整建议：分散

---
分析日期：2026-10-28
'''


def _stock(code, name):
    return {
        'stock_code': code, 'stock_name': name, 'code': code, 'name': name,
        'analysis': {'recommendation': '买入', 'operation_advice': '买入',
                     'risk_warnings': ['风险A']},
        'position_advice': 4, 'current_price': 50.0, 'stop_loss': 46.0,
        'target_price': 55.0, 'ml_prob_5d': 50, 'ml_prob_10d': 55,
        'ml_prob_20d': 62, 'short_term_advice': '买入', 'mid_term_advice': '买入',
    }


class TestEmailAssembly(unittest.TestCase):
    def test_section_order(self):
        captured = {}

        def _capture(content, date_str):
            captured['text'] = content
            return 'out.md'

        patches = [
            patch.object(ca, 'chat_with_llm', return_value=ADVICE),
            patch.object(ca, 'extract_llm_recommendations',
                         return_value={'short_term': 'SHORT', 'medium_term': 'MID'}),
            patch.object(ca, 'extract_ml_predictions', return_value={
                'ensemble': 'ML_TEXT', 'ensemble_email': 'ML_EMAIL',
                'market_layer': 'weak', 'up_ratio': 0.35,
                'three_horizon_results': {'2318.HK': {}, '0728.HK': {}},
                'chip_data': {}, 'network_insights': {},
                'risk_reward_data': {}, 'historical_pl_data': {},
            }),
            patch.object(ca, 'get_sector_analysis', return_value=None),
            patch.object(ca, 'get_dividend_info', return_value=[]),
            patch.object(ca, 'get_hsi_analysis', return_value={
                'current_price': 20000, 'change_pct': 1.0, 'change_points': 100,
                'rsi': 50, 'ma20': 19000, 'ma50': 18000,
                'amount': 1000, 'amount_ratio': 1.0,
            }),
            patch.object(ca, 'get_hsi_email_indicators', return_value=None),
            patch.object(ca, 'get_stock_anomalies', return_value=[]),
            patch.object(ca, 'generate_anomaly_report_content', return_value='ANOMALY'),
            patch.object(ca, 'get_current_market_state', return_value=None),
            patch.object(ca, 'generate_technical_indicators_table', return_value='TABLE'),
            patch.object(ca, 'get_recent_transactions', return_value=pd.DataFrame()),
            patch.object(ca, 'format_recent_transactions', return_value='TRADES'),
            patch.object(ca, 'get_last_trading_day', return_value='2026-10-28'),
            patch.object(ca, 'save_comprehensive_report_md', side_effect=_capture),
            patch.object(ca, 'get_stock_realtime_data', return_value={}),
            patch.object(ca, 'build_stock_data_for_llm', return_value=''),
            patch.object(ca, 'extract_stock_data_with_llm',
                         side_effect=[_stock('2318.HK', '中国平安'), _stock('0728.HK', '中国电信')]),
            patch.object(ca, 'parse_recommendation_from_text', return_value={}),
            patch.object(ca, 'get_holder_advice_with_llm', return_value={}),
            patch.object(ca, 'fill_calibrated_ml_probs'),
            patch.object(ca, 'compute_hk_stock_dimensions', return_value={'x': 0}),
        ]
        for p in patches:
            p.start()
        try:
            with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
                f.write(ADVICE)
                llm_path = f.name
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as out:
                out_path = out.name
            ca.run_comprehensive_analysis(
                llm_path, None, output_filepath=out_path, send_email_flag=False,
                use_cached_predictions=True, stock_codes=['2318.HK', '0728.HK'])
        finally:
            for p in patches:
                p.stop()
            for path in (locals().get('llm_path'), locals().get('out_path')):
                if path and os.path.exists(path):
                    os.remove(path)

        text = captured['text']
        order = ['# 综合买卖建议', '## 重点个股分析', '# 信息参考',
                 '# 附录', '## 附录 A', '## 附录 B']
        positions = [text.index(x) for x in order]
        self.assertEqual(positions, sorted(positions), order)
        self.assertIn('2318.HK', text)
        self.assertIn('0728.HK', text)
        self.assertNotIn('1600.HK', text)
        self.assertNotIn('<!--', text)


if __name__ == '__main__':
    unittest.main()
