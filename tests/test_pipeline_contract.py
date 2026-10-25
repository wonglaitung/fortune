# -*- coding: utf-8 -*-
"""管线契约测试（⓪ 同一性闸 / lessons 三.60 · O3）

背景：2026-10-12 系统比对 Walk-forward 与生产两条路径，查出 5 条独立口径不一致，
其中「训练时点 = 推理时点」「社区 ID 集合」「列集合同源」「概率同源」「默认值只许一处」
原本**只写在 lessons 里**，靠人记得——同一类错误此前已复发三次（三.28 / 三.57 / 三.60）。

本文件把这些教训变成**可执行断言**。两类：

* **回归护栏**（当前通过）：固化已知正确的口径，防被人"顺手简化"掉
  （如 embargo 的 default=None 被误读为"不生效"那次更正）。
* **已知违约**（当前失败）：标 `xfail(strict=True)`。
  strict 的意义：**一旦修好就从 XFAIL 变 XPASS → 测试失败**，
  强制你回来删标记并把结论写进文档——即「唯一的自动报警器」（lessons 三.59 的思路）。

⚠️ 全部断言只读**源码文本 / AST / 已落盘产物**，不联网、不加载模型、不取数。
"""
import ast
import pickle
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_SRC = ROOT / 'ml_services' / 'ml_trading_model.py'
WF_SRC = ROOT / 'ml_services' / 'walk_forward_validation.py'
FS_SRC = ROOT / 'ml_services' / 'feature_selection.py'
RUNNER_SRC = ROOT / 'scripts' / 'run_comprehensive_analysis.sh'
CA_SRC = ROOT / 'comprehensive_analysis.py'

F1_COLS = ['SFC_short_z', 'Net_Margin', 'ROE', 'ROA', 'Debt_Equity', 'OCF_Quality', 'Asset_Turnover']
FLAG_NAMES = ['F1_FEATURES', 'F1_FORCE', 'HIBOR_FEATURES', 'HIBOR_FORCE']


def _src(path):
    return path.read_text(encoding='utf-8')


def _default_of(path, func_name, param):
    """取 AST 中某函数某关键字参数的默认值字面量；找不到返回 '<MISSING>'。"""
    tree = ast.parse(_src(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            pos = [a.arg for a in node.args.posonlyargs] + [a.arg for a in node.args.args]
            defaults = [None] * (len(pos) - len(node.args.defaults)) + list(node.args.defaults)
            for name, dflt in zip(pos, defaults):
                if name == param:
                    if dflt is None:
                        return None
                    try:
                        return ast.literal_eval(dflt)
                    except (ValueError, SyntaxError):
                        return ast.dump(dflt)
            return '<NO-SUCH-PARAM>'
    return '<NO-SUCH-FUNC>'


def _kwargs_of_calls(path, func_name):
    """返回该文件中所有 `func_name(...)` 调用的关键字参数名集合（列表）。"""
    tree = ast.parse(_src(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else (fn.id if isinstance(fn, ast.Name) else None)
            if name == func_name:
                out.append({k.arg for k in node.keywords})
    return out


# ======================================================================
# 回归护栏（当前应通过）
# ======================================================================

def test_embargo_defaults_to_horizon():
    """embargo 必须归一化为 horizon（--embargo-days 的 argparse 默认是 None，不代表不生效）。

    2026-10-12 曾因只 grep 到 `default=None` 误判「embargo 未生效」并写进结论，
    实测 fold1 训练期止于 2023-05-11（5-31 减 20 天）证明生效。此断言固化该更正。
    """
    text = _src(WF_SRC)
    assert 'embargo_days if embargo_days is not None else horizon' in text, \
        'embargo 归一化丢失：--embargo-days 默认 None 时不再回退到 horizon，会造成标签穿越'


@pytest.mark.skipif(not (ROOT / 'data' / 'feature_selection' /
                        'statistical_features_latest.txt').exists(),
                    reason='生产静态名单未生成（该文件不在库内）')
def test_production_pkl_contains_f1_and_hibor():
    """生产 pkl 必须真正着装 F1 与 HIBOR 列（K4「不得静默丢弃」的回归护栏）。"""
    import glob
    pkls = sorted(glob.glob(str(ROOT / 'data' / 'ml_trading_model_lgbm_20d.pkl')))
    if not pkls:
        pytest.skip('catboost_20d.pkl 未生成（gitignore）')
    cols = pickle.load(open(pkls[0], 'rb')).get('feature_columns') or []
    missing_f1 = [c for c in F1_COLS if c not in cols]
    assert not missing_f1, f'pkl 缺 F1 列 {missing_f1}（D23 K4：0/N 即静默丢弃）'
    assert any('HIBOR' in c for c in cols), 'pkl 缺 HIBOR 列（D24：不算即空转）'


# ======================================================================
# 已知违约（当前失败；strict xfail = 修好即报警）
# ======================================================================

def test_train_and_predict_use_same_shift():
    """训练与推理的 use_shift 必须相等——否则模型在推理时收到它训练时从未见过的分布。

    **D25（2026-10-13）已修复**：WF 与生产统一 `mode='production'`（T 收盘数据）。
    原为 xfail(strict=True)；修好即 XPASS 失败 → 强制回来改断言（本次即如此）。
    """
    import ast as _ast
    tree = _ast.parse(_src(WF_SRC))
    modes = [k.value.value for k in _ast.walk(tree) if isinstance(k, _ast.keyword)
             and k.arg == 'mode' and isinstance(k.value, _ast.Constant)]
    assert modes and all(m == 'production' for m in modes), (
        f'walk_forward_validation.py 的 prepare_data 调用必须全部显式传 mode=\'production\'，'
        f'实际出现 {set(modes)}（D25-4）'
    )
    src = _src(MODEL_SRC)
    assert "min_return_threshold=min_return_threshold,\n" in src and "mode='production')" in src, \
        'train() 的 prepare_data 调用未显式传 mode=\'production\'（D25-4）'


def test_wf_prefetch_passes_community_ids():
    """**D26（2026-10-13）已修复**（原 xfail(strict=True)）：WF 预取与每折训练均显式传
    社区集合，且走唯一入口 `resolve_community_ids()`。"""
    """WF 的**每一次** prepare_data 调用都必须显式传 community_ids。

    漏传的若是全段预取那一处，则后续每折都从 full_df 切片、preloaded 路径永不生效，
    net_comm_* 退回每股动态提取、缺列变 NaN（生产是显式 0）。
    """
    calls = _kwargs_of_calls(WF_SRC, 'prepare_data')
    assert calls, '未找到 prepare_data 调用'
    missing = [i for i, kws in enumerate(calls) if 'community_ids' not in kws]
    assert not missing, f'第 {missing} 处 prepare_data 调用未传 community_ids（共 {len(calls)} 处）'
    txt = _src(WF_SRC)
    assert 'resolve_community_ids' in txt, 'WF 未走唯一入口 resolve_community_ids()'
    # 退回动态提取的兜底路径必须显式报错，不得静默
    assert 'RuntimeError' in txt and '差异②' in txt, \
        'WF 解析失败时未显式报错（静默退回动态提取 = 差异② 复发）'


def test_wf_and_production_same_prob_caliber():
    """WF 与生产**方向判定**必须用同一概率口径（D26 后：均为**模型原始 prob**，不校准确方向）。

    历史：差异⑤ 原为「WF 跑未校准、生产跑校准后」。**D26 已解决**：生产 `apply_to_results`
    把 `probability` 保持原始 P(上涨)、校准只写 `confidence` ⇒ 与 WF（`prediction_proba[:,1]`）**同源**。
    本测试固化该口径（原 `xfail(strict)`，D45 转正）。
    """
    wf = _src(WF_SRC)
    assert "prediction_proba[:, 1]" in wf, 'WF 未用原始 predict_proba[:,1] 概率'
    dc = (ROOT / 'ml_services' / 'daily_confidence.py').read_text(encoding='utf-8')
    assert 'probability 保持原始' in dc, 'daily_confidence 未保留原始 probability（D26）'
    assert "preds[h]['probability'] =" not in dc, 'daily_confidence 覆写了 probability（D26 缺口复发）'
    assert "preds[h]['confidence'] = conf" in dc, '校准值未只写 confidence（D26）'


def test_runner_lookup_path_matches_feature_selection_output():
    """runner 查特征文件的目录必须与 feature_selection.py 的 --output-dir 一致（否则 cp 是死代码）。

    **D45（2026-10-15）已修复**：runner 原查 `output/` 而 feature_selection 写 `data/feature_selection/`。
    """
    m_out = re.search(r"--output-dir'[^\n]*default='([^']+)'", _src(FS_SRC))
    assert m_out, '未解析到 feature_selection 的 --output-dir 默认值'
    out_dir = m_out.group(1)

    m_ls = re.search(r'LATEST_FEATURE_FILE=\$\(ls -t\s+(\S+)', _src(RUNNER_SRC))
    assert m_ls, '未解析到 runner 的 ls -t 路径'
    lookup_dir = m_ls.group(1).rsplit('/', 1)[0]

    assert lookup_dir.rstrip('/') == out_dir.rstrip('/'), (
        f'runner 查 {lookup_dir}/ 而 feature_selection 写 {out_dir}/ → cp 永不执行'
    )


def test_flag_default_defined_only_once():
    """每个 flag 的默认值只许一处（run_manifest.FLAG_DEFAULTS），门控点一律走 resolve_flags()。

    **O5（2026-10-13）已修复**（原 xfail(strict=True)）：4 个真实门控点已改走
    resolve_flags()，两个 runner 只做透传、不再写 `${X:-1}`。
    """
    from ml_services.run_manifest import FLAG_DEFAULTS
    assert set(FLAG_DEFAULTS) == set(FLAG_NAMES)

    offenders = {}
    for f in ('scripts/run_walk_forward.sh', 'scripts/run_comprehensive_analysis.sh'):
        txt = (ROOT / f).read_text(encoding='utf-8')
        for flag in FLAG_NAMES:
            if re.search(rf'\$\{{{flag}:-', txt):
                offenders.setdefault(flag, []).append(f + ': shell 仍写 :- 默认')
    for f in ('ml_services/ml_trading_model.py', 'ml_services/walk_forward_validation.py',
              'ml_services/us_market_data.py'):
        txt = (ROOT / f).read_text(encoding='utf-8')
        for flag in FLAG_NAMES:
            # 只允许 run_manifest 内部的 resolve_flags 访问；门控点不得裸读 env
            for m in re.finditer(rf"environ\.get\(\s*['\"]{flag}['\"]", txt):
                offenders.setdefault(flag, []).append(f'{f}:{txt[:m.start()].count(chr(10))+1} 裸读 os.environ')
    assert not offenders, f'以下 flag 的默认值/读取点违规：{offenders}'


def test_scope_label_is_tool_enforced():
    """「适用范围：WF 内部比较 ✓ / 外推生产 ✗」须由闸门强制输出，而非靠自觉。

    **D45（2026-10-15）已修复**：presentation_gate 现在每次输出都带该标注。
    """
    targets = [ROOT / 'scripts' / 'presentation_gate.py', ROOT / 'scripts' / 'doc_gate.py',
               ROOT / '.githooks' / 'pre-commit']
    hit = [t.name for t in targets if t.exists() and '适用范围' in t.read_text(encoding='utf-8')]
    assert hit, '无任何闸门脚本检查「适用范围」标注 → 该规则不会被拦截'

# ======================================================================
# O4 运行清单与缓存隔离（2026-10-13 已落地 → 回归护栏，应通过）
# ======================================================================

def test_run_manifest_defaults_match_actual_gating_sites():
    """清单报告的开关状态必须与**真实门控点**一致 —— 清单说谎 = 比没有清单更坏。

    真实门控点（都是裸读 env，不走清单）：
      F1_FEATURES     ml_trading_model.py:1673  `!= '1'`      → 未设=关
      F1_FORCE        walk_forward_validation.py:480 `== '1'`  → 未设=关
      HIBOR_FEATURES  us_market_data.py:572     `get(...,'1')`→ 未设=**开**
      HIBOR_FORCE     ml_trading_model.py:4516  `== '1'`      → 未设=关
    """
    import os
    import importlib
    saved = {k: os.environ.pop(k, None) for k in FLAG_NAMES}
    try:
        rmf = importlib.import_module('ml_services.run_manifest')
        flags = rmf.resolve_flags()
        # D22/D23/D24：四个开关默认全 ON；空串（shell 透传未设）同样落到默认
        assert flags == {'F1_FEATURES': '1', 'F1_FORCE': '1',
                         'HIBOR_FEATURES': '1', 'HIBOR_FORCE': '1'}, \
            f'清单默认值与 D22/D23/D24 记载的默认口径不一致：{flags}'
        for k in FLAG_NAMES:
            os.environ[k] = ''
        assert rmf.resolve_flags() == flags, '空串（shell 透传）未落到默认'
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_cache_key_isolates_community_set_and_mode():
    """特征缓存键必须区分「社区集合」与「mode」——否则 WF 与生产互相覆写。

    ⚠️ D25 把 WF 与生产统一到同一 use_shift 后，时点后缀（`tclose`/`tminus1`，D27 前叫
    `noshift`/`shift`）**不再能隔离**，
    故社区集合与 mode 必须进键（O4）。
    """
    from ml_services.ml_trading_model import _get_feature_cache_key
    from ml_services.run_manifest import manifest_fingerprint

    def key(cids, mode):
        fp = manifest_fingerprint(community_ids=cids, mode=mode)
        return _get_feature_cache_key('0005', '20261012', use_shift=True,
                                      manifest_fp=fp, community_ids=cids)

    k_wf = key(None, 'backtest')                      # WF：每股动态社区
    k_prod = key(list(range(14)), 'production')       # 生产：静态∪PIT
    k_prod_backtest = key(list(range(14)), 'backtest')
    assert k_wf != k_prod, 'WF(动态社区) 与生产(静态∪PIT) 缓存键相同 → 会互相覆写'
    assert k_prod != k_prod_backtest, 'mode 不同但缓存键相同'
    assert k_prod == _get_feature_cache_key(
        '0005', '20261012', use_shift=True,
        manifest_fp=manifest_fingerprint(community_ids=list(range(14)), mode='production'),
        community_ids=list(range(14))), '同配置应稳定命中同一缓存键'


def test_cache_key_flags_change_key():
    """开关变化必须改缓存键（否则「加了特征却读到旧缓存」静默发生）。"""
    import os
    from ml_services.ml_trading_model import _get_feature_cache_key
    from ml_services.run_manifest import manifest_fingerprint

    saved = {k: os.environ.get(k) for k in FLAG_NAMES}
    try:
        os.environ.update({'F1_FEATURES': '0', 'F1_FORCE': '0',
                           'HIBOR_FEATURES': '1', 'HIBOR_FORCE': '0'})
        k0 = manifest_fingerprint(community_ids=[0, 1], mode='production')
        os.environ['HIBOR_FORCE'] = '1'
        k1 = manifest_fingerprint(community_ids=[0, 1], mode='production')
        assert k0 != k1, 'HIBOR_FORCE 变化未反映到指纹'
        a = _get_feature_cache_key('0005', '20261012', True, manifest_fp=k0, community_ids=[0, 1])
        b = _get_feature_cache_key('0005', '20261012', True, manifest_fp=k1, community_ids=[0, 1])
        assert a != b
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_manifest_roundtrip(tmp_path):
    from ml_services.run_manifest import write_manifest, load_manifest
    p = tmp_path / 'run_manifest.json'
    man = write_manifest({'mode': 'production', 'community_ids': [0, 1],
                          'use_shift': False, 'caller': 'test'}, path=str(p))
    assert p.exists()
    got = load_manifest(path=str(p))
    assert got['mode'] == 'production'
    assert got['community_ids'] == [0, 1]
    assert got['flags']['F1_FEATURES'] in ('0', '1')
    assert len(man['fingerprint']) == 12


# ======================================================================
# D25 标签后移（2026-10-13 已落地 → 回归护栏）
# ======================================================================

def test_label_entry_lag_is_single_source_and_applied():
    """入场滞后天数必须**全项目只定义一次**，且标签窗口 = [T+lag, T+lag+h]。

    D25：决策 T 日 16:30 HKT → T+1 成交 → 标签 T+1→T+1+h。
    散落魔数会静默产生「训练标签与评估标签窗口不一致」（三.48 同族）。
    """
    import pandas as pd
    from ml_services.ml_trading_model import FeatureEngineer, LABEL_ENTRY_LAG_DAYS

    # 单一事实源：ml_trading_model 定义，其它三处只 import 引用
    assert LABEL_ENTRY_LAG_DAYS == 1
    for f in ('ml_services/walk_forward_validation.py', 'scripts/rel_alpha_check.py'):
        txt = (ROOT / f).read_text(encoding='utf-8')
        assert 'LABEL_ENTRY_LAG_DAYS' in txt, f'{f} 未引用 LABEL_ENTRY_LAG_DAYS（各自写死数字）'
        assert 'shift(-horizon)' not in txt and 'shift(-self.horizon)' not in txt, \
            f'{f} 仍按旧窗口 shift(-horizon)，未加入场滞后'

    # 行为验证：h=3、lag=1 时标签 = Close[T+4]/Close[T+1]-1
    idx = pd.date_range('2026-01-01', periods=12, freq='B')
    close = [float(x) for x in range(100, 112)]
    df = pd.DataFrame({'Close': close}, index=idx)
    out = FeatureEngineer().create_label(df.copy(), horizon=3, for_backtest=True,
                                         min_return_threshold=0.0)
    h = 3
    for i in range(5):
        exp = close[i + LABEL_ENTRY_LAG_DAYS + h] / close[i + LABEL_ENTRY_LAG_DAYS] - 1
        assert abs(out['Future_Return'].iloc[i] - exp) < 1e-12, \
            f'第 {i} 行标签窗口不符 D25：实得 {out["Future_Return"].iloc[i]}，期望 {exp}'
    # 尾部 lag+h 行无标签
    assert out['Future_Return'].tail(h + LABEL_ENTRY_LAG_DAYS).isna().all()


def test_community_id_resolution_has_single_implementation():
    """社区集合解析逻辑**只允许一份**（D26：此前在 ml_trading_model.py 内复制了两份）。"""
    from ml_services.network_ids import resolve_community_ids, STATIC_NETWORK_FILE

    assert callable(resolve_community_ids)
    offenders = {}
    # ⚠️ 只覆盖**港股**路径。A 股按 D25 决策「不改」，其自带一份解析逻辑
    #    （a_stock_ml_model.py 约 1028 行）**不在本条范围**，属已知遗留项。
    #    不把 A 股纳入断言，是为了让本测试保持「全绿=港股已对齐」的可读含义；
    #    若将来给 A 股开决策，须同步把该文件加进来。
    for f in ('ml_services/ml_trading_model.py', 'ml_services/walk_forward_validation.py'):
        p = ROOT / f
        if not p.exists():
            continue
        txt = p.read_text(encoding='utf-8')
        for marker in ('all_community_ids', 'pit_comm_ids'):
            if marker in txt:
                offenders.setdefault(f, []).append(marker)
    assert not offenders, f'以下文件仍自带社区集合解析逻辑副本：{offenders}'

    # 两份来源都要并入（静态 ∪ PIT），漏任一都会让社区集合变窄
    assert resolve_community_ids(include_pit=False) != resolve_community_ids(include_pit=True), \
        'include_pit 无效果 → PIT 社区集合没被并入'
    assert 'network_features_pit.json' in (ROOT / 'ml_services' / 'network_ids.py').read_text(encoding='utf-8')


def test_wf_and_production_share_same_community_set():
    """WF 与生产解析出的社区集合必须相等（D26 的 A 级主判据之一）。"""
    from ml_services.network_ids import resolve_community_ids
    comm = resolve_community_ids()
    assert comm is not None and len(comm) > 1, '社区集合解析异常'
    # 生产 pkl 内保存的集合须与当前解析一致（否则模型与数据面已漂移）
    import pickle
    p = ROOT / 'data' / 'ml_trading_model_lgbm_20d.pkl'
    if not p.exists():
        pytest.skip('catboost_20d.pkl 未生成（gitignore）')
    pkl = pickle.load(open(p, 'rb'))
    saved = pkl.get('community_ids')
    assert saved == comm, f'生产 pkl 社区集合 {saved} != 现解析 {comm}（模型已漂移，需重训）'
