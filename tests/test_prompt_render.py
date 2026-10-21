# -*- coding: utf-8 -*-
"""Prompt 渲染自检：防止邮件/LLM 提示词出现陈旧值、旧字段名、硬编码学习器矛盾。

覆盖回归点（2026-09-25 修复）：
1. 渲染后不得出现 '0.00%'（model_accuracy 坏值直出）
2. 渲染后不得出现 'catboost_prob'（已改名为 ml_prob_*）
3. prompt 不得硬编码 "（生产模型：LightGBM）" / "1/5天 CatBoost · 20天 LightGBM"
   （CI 实际训练学习器随脚本而变，声明必须中性）
4. 个股分析 prompt 的示例 JSON 必须使用 ml_prob_20d
"""
import ast
import re
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parents[1] / 'comprehensive_analysis.py'
SRC = SRC_PATH.read_text(encoding='utf-8')
A_SRC = (SRC_PATH.parent / 'a_stock_comprehensive_analysis.py').read_text(encoding='utf-8')
A_EMAIL_SRC = (SRC_PATH.parent / 'a_stock_email.py').read_text(encoding='utf-8')

RANK_FORBIDDEN = ['0.70', '0.65', '前约11%', '（前约11%）', 'probability_20d',
                  'P(模型方向判对)', '置信度评估', '动态阈值']
HARD_GATE_FORBIDDEN = ['最高优先级', '绝对禁止', '必须100%', '缺一不可', '三重确认',
                       '违反硬约束']

MODEL_ACCURACY = {
    'learner_20d': 'LightGBM',
    'ml_20d': {'accuracy': 0.5833, 'std': 0.0764, 'date': '2026-03-26'},
    '20d': {'accuracy': 0.5833, 'std': 0.0764, 'date': '2026-03-26'},
    '1d': {'accuracy': 0.6829, 'std': 0.0184, 'date': '2026-09-23'},
    '5d': {'accuracy': 0.6748, 'std': 0.0086, 'date': '2026-09-23'},
    'catboost': {'accuracy': 0.6101, 'std': 0.0219, 'date': '2026-09-23'},
}
ML_PREDICTIONS = {
    'ensemble': '{"ml_prob_20d": 0.54}',
    'ensemble_email': 'table',
    'short_term': '',
    'medium_term': '',
}


class Dummy:
    def __getattr__(self, k):
        return Dummy()

    def __getitem__(self, k):
        return Dummy()

    def get(self, *a, **k):
        return Dummy()

    def __call__(self, *a, **k):
        return Dummy()

    def __format__(self, spec):
        return '0.55'

    def __str__(self):
        return 'X'

    def __repr__(self):
        return 'X'

    def __add__(self, o):
        return 'X'

    def __radd__(self, o):
        return 'X'

    def __contains__(self, o):
        return False

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return True

    def __float__(self):
        return 0.55

    def __int__(self):
        return 1

    def __lt__(self, o):
        return False

    def __gt__(self, o):
        return True


class NS(dict):
    def __missing__(self, k):
        if k == 'model_accuracy':
            return MODEL_ACCURACY
        if k == 'ml_predictions':
            return ML_PREDICTIONS
        if k == '__builtins__':
            raise KeyError(k)
        return Dummy()


def _collect_prompt_nodes(tree):
    """收集 prompt 类赋值中的 f-string 节点：变量名含 PROMPT / text_email"""
    nodes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = []
        for t in targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
        if not any(('PROMPT' in n) or ('text_email' in n)
                   or n.lower() in ('prompt', 'prompt_text') for n in names):
            continue
        value = node.value
        for sub in ast.walk(value):
            if isinstance(sub, ast.JoinedStr):
                nodes.append(sub)
    return nodes


def _render(node):
    ns = NS(model_accuracy=MODEL_ACCURACY, ml_predictions=ML_PREDICTIONS)
    code = compile(ast.Expression(body=node), '<fstr>', 'eval')
    return eval(code, {'__builtins__': {}}, ns)


def _rendered_outputs():
    tree = ast.parse(SRC)
    outputs = []
    for node in _collect_prompt_nodes(tree):
        outputs.append((node.lineno, str(_render(node))))
    return outputs


def test_prompt_render_no_stale_values():
    outputs = _rendered_outputs()
    assert outputs, '未收集到任何 prompt f-string 节点'
    bad = [(ln, out) for ln, out in outputs if '0.00%' in out or 'catboost_prob' in out]
    assert not bad, f'发现陈旧值/旧字段: {[(ln, out[:120]) for ln, out in bad]}'


def test_prompt_no_hardcoded_learner_conflict():
    tree = ast.parse(SRC)
    bad_patterns = ['（生产模型：LightGBM）', '1/5天 CatBoost · 20天 LightGBM',
                    '20天：LightGBM（生产学习器）', '20天 {model_accuracy[\'learner_20d\']}']
    for node in _collect_prompt_nodes(tree):
        try:
            rendered = str(_render(node))
        except Exception:
            continue
        for pat in bad_patterns:
            assert pat not in rendered, f'line {node.lineno} 硬编码学习器与实际不符: {pat}'


def test_stock_analysis_prompt_uses_ml_prob_field():
    tree = ast.parse(SRC)
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'STOCK_ANALYSIS_PROMPT':
                    found = True
                    seg = ast.get_source_segment(SRC, node) or ''
                    assert 'ml_prob_20d' in seg, 'STOCK_ANALYSIS_PROMPT 缺少 ml_prob_20d 示例'
                    # 兜底读取允许旧字段，但 prompt 正文示例不得主用旧字段
                    m = re.search(r'"catboost_prob_20d"', seg)
                    assert not m, 'prompt 示例仍在使用 catboost_prob_20d'
    assert found, '未找到 STOCK_ANALYSIS_PROMPT 定义'


def _prompt_const_segment(src, name):
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.get_source_segment(src, node) or ''
    return ''


def test_hk_prompts_rank_only():
    for name in ('STOCK_ANALYSIS_PROMPT', 'COMPREHENSIVE_ANALYSIS_PROMPT',
                 'HOLDER_ADVICE_PROMPT'):
        seg = _prompt_const_segment(SRC, name)
        assert seg, f'未找到 {name}'
        for pat in RANK_FORBIDDEN:
            assert pat not in seg, f'{name} 仍含旧口径: {pat}'


def test_a_stock_prompt_rank_only():
    start = A_SRC.index('prompt = f"""你是一位专业的A股')
    end = A_SRC.index('请按照以下格式输出', start)
    seg = A_SRC[start:end]
    for pat in RANK_FORBIDDEN:
        assert pat not in seg, f'A股 prompt 仍含旧口径: {pat}'
    assert 'score_20d' in seg and '排序分数' in seg


def test_score_is_cross_reference_not_hard_gate():
    for name in ('COMPREHENSIVE_ANALYSIS_PROMPT', 'HOLDER_ADVICE_PROMPT'):
        seg = _prompt_const_segment(SRC, name)
        assert '交叉参考' in seg, f'{name} 未声明分数为交叉参考'
        for pat in HARD_GATE_FORBIDDEN:
            assert pat not in seg, f'{name} 仍把分数当硬门槛: {pat}'
    # 主分析 prompt 为局部 f-string
    start = SRC.index('prompt = f"""你是一位专业的投资分析师')
    end = SRC.index('请按照以下格式输出', start)
    seg = SRC[start:end]
    assert '交叉参考' in seg
    for pat in HARD_GATE_FORBIDDEN:
        assert pat not in seg, f'港股主 prompt 仍把分数当硬门槛: {pat}'
    # A股主 prompt
    a_start = A_SRC.index('prompt = f"""你是一位专业的A股')
    a_end = A_SRC.index('请按照以下格式输出', a_start)
    a_seg = A_SRC[a_start:a_end]
    assert '交叉参考' in a_seg
    for pat in HARD_GATE_FORBIDDEN:
        assert pat not in a_seg, f'A股主 prompt 仍把分数当硬门槛: {pat}'


def test_a_stock_email_prompt_cross_reference():
    start = A_EMAIL_SRC.index('market_info = f"""')
    end = A_EMAIL_SRC.index('"""', start + len('market_info = f"""')) + 3
    market_seg = A_EMAIL_SRC[start:end]
    assert '交叉参考' in market_seg
    for pat in HARD_GATE_FORBIDDEN + ['需概率≥', '校准后20日概率', '动态阈值']:
        assert pat not in market_seg, f'A股邮件市场段仍含旧硬门槛: {pat}'
    p_start = A_EMAIL_SRC.index('prompt = f"""你是一位专业的A股投资分析师')
    p_end = A_EMAIL_SRC.index('请按照以下六层分析框架', p_start)
    prompt_seg = A_EMAIL_SRC[p_start:p_end]
    for pat in HARD_GATE_FORBIDDEN + ['需概率≥', '校准后20日概率', '动态阈值']:
        assert pat not in prompt_seg, f'A股邮件 prompt 仍含旧硬门槛: {pat}'


if __name__ == '__main__':
    test_prompt_render_no_stale_values()
    test_prompt_no_hardcoded_learner_conflict()
    test_stock_analysis_prompt_uses_ml_prob_field()
    test_hk_prompts_rank_only()
    test_a_stock_prompt_rank_only()
    test_score_is_cross_reference_not_hard_gate()
    test_a_stock_email_prompt_cross_reference()
    print('ALL PASS')
