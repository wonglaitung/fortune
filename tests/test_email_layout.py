"""回归：综合分析邮件编排（D93）。

约束（1B/2B/3A）：
- 个股卡只覆盖 --stocks 指定股票；
- 个股卡在主结论之后、信息参考之前；
- 静态说明在附录（邮件末尾）；
- 个股卡内部不再用「一、二、三…」编号，避免与主报告编号冲突；
- HTML 只有一个 H1。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comprehensive_analysis as ca  # noqa: E402


def _stock(code, name):
    return {
        'stock_code': code,
        'stock_name': name,
        'code': code,
        'name': name,
        'analysis': {'recommendation': '观望'},
        'position_advice': 0,
        'current_price': 10.0,
        'stop_loss': 9.2,
        'target_price': 11.0,
        'ml_prob_5d': 50,
        'ml_prob_10d': 55,
        'ml_prob_20d': 60,
        'short_term_advice': '观望',
        'mid_term_advice': '观望',
        'risk_warnings': ['风险 A'],
    }


class TestStockCards(unittest.TestCase):
    def test_cards_have_no_chinese_numbering(self):
        html = ca.generate_stock_sections_html([_stock('2318.HK', '中国平安')])
        self.assertIn('<h3>核心指标</h3>', html)
        self.assertNotIn('<h3>一、', html)
        self.assertNotIn('<h3>十一、', html)

    def test_cards_have_no_h1(self):
        html = ca.generate_stock_sections_html([_stock('2318.HK', '中国平安')])
        self.assertNotIn('<h1', html.lower())

    def test_text_cards_cover_only_specified_codes(self):
        text = ca.generate_stock_sections_text([
            _stock('2318.HK', '中国平安'),
            _stock('1288.HK', '农业银行'),
        ])
        self.assertIn('2318.HK', text)
        self.assertIn('1288.HK', text)
        self.assertNotIn('1600.HK', text)

    def test_empty_cards(self):
        self.assertEqual(ca.generate_stock_sections_html([]), '')
        self.assertEqual(ca.generate_stock_sections_text([]), '')


class TestPlaceholderOrder(unittest.TestCase):
    def test_detailed_cards_between_main_and_reference(self):
        full = (
            'BANNER\n<!--RISK_PANEL-->\nMAIN ADVICE\n'
            '<!--DETAILED_STOCKS_HTML-->\n<!--DETAILED_STOCKS_TEXT-->\n'
            '# 信息参考\n## 一、恒生指数技术分析\n'
            '# 附录\n## 附录 B：技术指标说明\n'
        )
        text, html = ca.apply_email_placeholders(
            full, full, '<div>CARD</div>', '### CARD')
        self.assertLess(text.index('MAIN ADVICE'), text.index('CARD'))
        self.assertLess(text.index('CARD'), text.index('# 信息参考'))
        self.assertLess(text.index('# 信息参考'), text.index('# 附录'))
        self.assertNotIn('<!--', text)
        self.assertIn('<div>CARD</div>', html)
        self.assertLess(html.index('MAIN ADVICE'), html.index('CARD'))
        self.assertLess(html.index('CARD'), html.index('# 信息参考'))
        self.assertNotIn('<!--DETAILED', html)

    def test_risk_panel_placeholder_removed_from_text(self):
        text, _ = ca.apply_email_placeholders(
            'BANNER\n<!--RISK_PANEL-->\nMAIN', 'BANNER\n<div>PANEL</div>\nMAIN')
        self.assertNotIn('<!--RISK_PANEL-->', text)
        self.assertIn('风险面板', text)


if __name__ == '__main__':
    unittest.main()
