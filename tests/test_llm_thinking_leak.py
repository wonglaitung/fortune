"""
回归测试：大模型「思考过程泄漏到邮件/报告」防护。

历史故障（2026-09-30 港股综合分析邮件）：
  opencode zen/go 代理 + deepseek-v4-flash 忽略 DashScope 专有的 enable_thinking，
  reasoning_content 与 content 共享 max_tokens 预算，长 prompt 下预算被思考吃光，
  content 为空 -> qwen_engine 旧代码把 reasoning_content 当答案返回，
  整段英文 CoT（5666 行）被原样写进邮件与 output/comprehensive_reports/。
"""
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_services.qwen_engine import strip_thinking  # noqa: E402
import comprehensive_analysis as ca  # noqa: E402


def _good_advice(date_str='2026-09-30'):
    return f"""# 综合买卖建议
分析日期：{date_str}

## 强烈买入信号（2-3只）
当前无股票同时满足'短期买入+中期买入+ML概率≥0.60'的三重确认条件，本期不出具该档信号。

## 买入信号（3-5只）
当前无股票同时满足'短期买入+中期买入+ML概率≥0.55'的三重确认条件，本期不出具该档信号。

## 持有/观望
1. 2318 平安银行
   - 操作建议：持有

## 卖出信号
1. 1138 招商银行
   - 操作建议：卖出/减仓

## 风险控制建议
建议仓位百分比：45%
"""


class TestStripThinking(unittest.TestCase):
    def test_strips_closed_think_block(self):
        self.assertEqual(
            strip_thinking('<think>We need to analyze...</think>\n\n# 综合买卖建议'),
            '# 综合买卖建议')

    def test_strips_unclosed_truncated_think_block(self):
        # max_tokens 耗尽时 <think> 未闭合，思考内容直接顶到输出末尾
        out = strip_thinking('# 综合买卖建议\n<think>We need to think about the rules')
        self.assertNotIn('<think>', out)
        self.assertNotIn('We need', out)

    def test_case_insensitive(self):
        self.assertEqual(strip_thinking('<THINK>x</THINK>ok'), 'ok')

    def test_empty_and_none_safe(self):
        self.assertEqual(strip_thinking(''), '')
        self.assertIsNone(strip_thinking(None))

    def test_content_without_think_unchanged(self):
        self.assertEqual(strip_thinking('  # 综合买卖建议  '), '# 综合买卖建议')


class TestValidateAdviceResponse(unittest.TestCase):
    def test_accepts_well_formed_advice(self):
        self.assertEqual(ca.validate_advice_response(_good_advice(), '2026-09-30'),
                         _good_advice().strip())

    def test_rejects_empty_and_none(self):
        self.assertEqual(ca.validate_advice_response('', '2026-09-30'), '')
        self.assertEqual(ca.validate_advice_response(None, '2026-09-30'), '')
        self.assertEqual(ca.validate_advice_response('   \n ', '2026-09-30'), '')

    def test_rejects_leaked_reasoning(self):
        """真实故障形态：整段英文思考，章节全无"""
        leaked = (
            "We need answer in Chinese, follow format. Need analyze based on provided info. "
            "Need note: We have only ML 20d predictions, no explicit short-term recommendations.\n"
            "Let's think about the hard constraints. Actually, we cannot invent signals.\n"
        )
        self.assertEqual(ca.validate_advice_response(leaked, '2026-09-30'), '')

    def test_rejects_reasoning_with_sections_appended(self):
        """章节齐全但正文仍是思考过程 —— 靠开头 CoT 标记拦截"""
        text = ("We need answer in Chinese. Need maybe infer the missing signals.\n"
                "Actually, the rule says only act when short and mid agree.\n\n"
                + _good_advice())
        self.assertEqual(ca.validate_advice_response(text, '2026-09-30'), '')

    def test_rejects_truncated_missing_sections(self):
        """reasoning 吃掉预算 -> content 被截断，缺尾部章节"""
        truncated = "# 综合买卖建议\n分析日期：2026-09-30\n\n## 强烈买入信号\n1. 2318 平安"
        self.assertEqual(ca.validate_advice_response(truncated, '2026-09-30'), '')

    def test_rejects_stale_date(self):
        """上一次运行的残留结果（日期不符）不得当作本次建议发出"""
        self.assertEqual(ca.validate_advice_response(_good_advice('2026-09-29'), '2026-09-30'), '')

    def test_real_leaked_artifact_rejected(self):
        """若故障产物仍在仓库，必须被拒绝"""
        path = 'output/comprehensive_reports/2026-09-30.md'
        if not os.path.exists(path):
            self.skipTest('故障产物已不存在')
        with open(path, encoding='utf-8') as f:
            self.assertEqual(ca.validate_advice_response(f.read(), '2026-09-30'), '')


class TestChatWithLLMGuards(unittest.TestCase):
    """验证请求参数与响应处理两道防线"""

    def _response(self, content, reasoning=None, finish='stop'):
        msg = {'role': 'assistant', 'content': content}
        if reasoning is not None:
            msg['reasoning_content'] = reasoning
        return {
            'choices': [{'index': 0, 'finish_reason': finish, 'message': msg}]
        }

    def _post(self):
        return patch('llm_services.qwen_engine.requests.post')

    def _call(self, enable_thinking):
        """api_key 在模块导入时读取，需直接打补丁"""
        with patch('llm_services.qwen_engine.api_key', 'test-key'), \
                patch('llm_services.qwen_engine.log_message'):
            from llm_services.qwen_engine import chat_with_llm
            return chat_with_llm('hi', enable_thinking=enable_thinking)

    def test_sends_reasoning_effort_none_when_thinking_disabled(self):
        """代理只认 reasoning_effort，enable_thinking 会被忽略"""
        with self._post() as post:
            post.return_value.status_code = 200
            post.return_value.text = '{}'
            post.return_value.json.return_value = self._response('# 综合买卖建议')
            post.return_value.raise_for_status.return_value = None
            self._call(enable_thinking=False)
            self.assertEqual(post.call_args.kwargs['json']['reasoning_effort'], 'none')

    def test_no_reasoning_effort_when_thinking_enabled(self):
        with self._post() as post:
            post.return_value.status_code = 200
            post.return_value.text = '{}'
            post.return_value.json.return_value = self._response('ok')
            post.return_value.raise_for_status.return_value = None
            self._call(enable_thinking=True)
            self.assertNotIn('reasoning_effort', post.call_args.kwargs['json'])

    def test_never_returns_reasoning_as_answer(self):
        """核心回归：content 空 + 有 reasoning -> 返回空，绝不返回思考过程"""
        with self._post() as post:
            post.return_value.status_code = 200
            post.return_value.text = '{}'
            post.return_value.json.return_value = self._response(
                '', reasoning='We need answer in Chinese. Need analyze...')
            post.return_value.raise_for_status.return_value = None
            self.assertEqual(self._call(enable_thinking=False), '')

    def test_strips_think_from_content(self):
        with self._post() as post:
            post.return_value.status_code = 200
            post.return_value.text = '{}'
            post.return_value.json.return_value = self._response(
                '<think>We need think</think># 综合买卖建议')
            post.return_value.raise_for_status.return_value = None
            self.assertEqual(self._call(enable_thinking=False), '# 综合买卖建议')


if __name__ == '__main__':
    unittest.main()
