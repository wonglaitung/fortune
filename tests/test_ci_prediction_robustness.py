"""CI 回归：预测缺列与 LLM 瞬时 410 不得让整条流水线失败（D91，2026-10-28）。

- 7/30 只股票因 F1 列（Debt_Equity/OCF_Quality/Asset_Turnover）缺失被整只丢弃；
- 综合分析 LLM 调用间歇性 HTTP 410 Gone，未重试直接 exit 1。
"""
import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_services.ml_trading_model import FeatureEngineer  # noqa: E402
import llm_services.qwen_engine as qe  # noqa: E402

F1_COLUMNS = {
    'SFC_short_z', 'Net_Margin', 'ROE', 'ROA',
    'Debt_Equity', 'OCF_Quality', 'Asset_Turnover',
}


class TestF1Placeholders(unittest.TestCase):
    def test_returns_all_columns_when_sources_fail(self):
        fe = FeatureEngineer()
        idx = pd.date_range('2026-01-01', periods=3)
        df = pd.DataFrame({'Close': [1.0, 2.0, 3.0]}, index=idx)
        with patch('ml_services.ml_trading_model.resolve_flags',
                   return_value={'F1_FEATURES': '1'}), \
                patch('ml_services.ml_trading_model.build_fundamental_features',
                      return_value={}), \
                patch('ml_services.ml_trading_model.sfc_feature_frame',
                      side_effect=RuntimeError('no sfc')), \
                patch('ml_services.ml_trading_model.fundamental_feature_frame',
                      side_effect=RuntimeError('no fund')):
            out = fe.create_f1_alternative_features('0700.HK', df)
        self.assertIsNotNone(out)
        self.assertTrue(F1_COLUMNS.issubset(set(out.columns)), out.columns)


class _Resp:
    def __init__(self, status, content='ok'):
        self.status_code = status
        self.text = '{}'
        self.headers = {}
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': self._content}}]}


class TestLLMRetry(unittest.TestCase):
    def test_retries_on_410_then_succeeds(self):
        with patch.object(qe, 'api_key', 'test-key'), \
                patch.object(qe, 'log_message'), \
                patch.object(qe.time, 'sleep'), \
                patch.object(qe.requests, 'post',
                             side_effect=[_Resp(410), _Resp(200, '# ok')]) as post:
            out = qe.chat_with_llm('hi', enable_thinking=False)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(out, '# ok')


if __name__ == '__main__':
    unittest.main()
