# -*- coding: utf-8 -*-
"""运行清单（O4 / lessons 三.60 O4）：一次解析、落盘、指纹化。

**为什么需要**：改动之前，每个 flag 的默认值在 shell 与 python 各写一遍
（`us_market_data.py:572` 默认 ON vs `ml_trading_model.py:4516` 默认 OFF），
且特征缓存键只含 `{code}_{last_date}_{shift}_{f1}` —— **不含 HIBOR / 社区集合 / PIT 版本**。
后果：WF（每股动态社区）与生产（静态∪PIT）会写**同一个缓存键**、互相覆写；
两次运行是否可比也无从判断。

**本模块提供**：
1. `resolve_flags()` —— 所有开关的**唯一**解析入口（默认值只在此处定义一次）。
2. `build_manifest()` / `manifest_fingerprint()` —— 运行清单与短指纹。
3. `write_manifest()` / `load_manifest()` —— 落盘 `output/run_manifest.json`。

**用法约定**：`manifest_fingerprint()` 不同 ⇒ 两次运行**不可互比**（写进 A 闸）。
"""
import hashlib
import json
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# ⚠️ 开关默认值**只在此处定义一次**（lessons 三.60 O5 / D22 / D23 / D24）。
#    四个真实门控点必须一律走 `resolve_flags()`，不得再裸读 os.environ。
#    默认值取 **ON**，与 D22/D23/D24 记载的「生产/回测默认全开」一致——
#    此前 shell 写 `${X:-1}` 而 python 门控点默认 OFF，两处默认值方向相反（违反 4 处）。
#    F1_FORCE / HIBOR_FORCE 必须与对应 FEATURES 同向，否则「算了不用」空转（D24）。
# ---------------------------------------------------------------------------
FLAG_DEFAULTS = {
    'F1_FEATURES': '1',       # D23 例外观察采用
    'F1_FORCE': '1',          # 必须与 F1_FEATURES 同向，否则「算了不用」
    'HIBOR_FEATURES': '1',    # D22 例外观察采用
    'HIBOR_FORCE': '1',       # D24；未开启时不算进列
    'EVENT_FEATURES': '0',    # P2c 消融；默认关，验证后再决定是否上线
}

# 参与缓存隔离的开关（任一变化 ⇒ 特征列集合可能变化 ⇒ 缓存键必须变）
CACHE_AFFECTING_FLAGS = ('F1_FEATURES', 'F1_FORCE', 'HIBOR_FEATURES', 'HIBOR_FORCE', 'EVENT_FEATURES')

#: 这些开关**只允许**出现在 run_manifest 的默认表里；其它文件出现即违反 O5。
GATING_FILES = ('ml_services/ml_trading_model.py',
                'ml_services/walk_forward_validation.py',
                'ml_services/us_market_data.py',
                'a_stock_ml_model.py')

MANIFEST_PATH = os.environ.get('RUN_MANIFEST_PATH') or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'output', 'run_manifest.json')


def resolve_flags():
    """解析全部开关 —— 唯一入口。返回值只含 '0'/'1'。"""
    out = {}
    for k, default in FLAG_DEFAULTS.items():
        v = os.environ.get(k)
        if v is None or v.strip() == '':
            v = default
        out[k] = '1' if str(v).strip() == '1' else '0'
    return out


def _hash_obj(obj, n=12):
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:n]


def community_fingerprint(community_ids):
    """社区 ID 集合指纹 —— WF（动态）与生产（静态∪PIT）据此隔离缓存。

    `community_ids=None` 表示每股动态提取，与任何显式集合都不同 ⇒ 独立指纹。
    """
    if community_ids is None:
        return 'dyn'
    try:
        ids = sorted({int(x) for x in community_ids})
    except (TypeError, ValueError):
        ids = sorted(community_ids)
    return 'c' + _hash_obj(ids, 8)


def build_manifest(extra=None):
    """构建运行清单。`extra` 放本次运行的其它上下文（如 mode / community_ids）。"""
    flags = resolve_flags()
    man = {
        'generated_at': datetime.now().isoformat(),
        'flags': flags,
        'env': {
            'US_MARKET_SNAPSHOT_DIR': os.environ.get('US_MARKET_SNAPSHOT_DIR') or '',
            'HK_MARKET_SNAPSHOT_DIR': os.environ.get('HK_MARKET_SNAPSHOT_DIR') or '',
            'GATE_SOURCE_CSV': os.environ.get('GATE_SOURCE_CSV') or '',
        },
    }
    if extra:
        man.update(extra)
    man['fingerprint'] = _hash_obj(
        {k: man[k] for k in ('flags', 'env') if k in man} |
        ({k: extra[k] for k in ('mode', 'community_ids') if extra and k in extra}),
        12)
    return man


def manifest_fingerprint(community_ids=None, mode=None):
    """短指纹 —— 直接进特征缓存键。

    组成：开关 + 快照 env + mode + 社区集合
    ⇒ 任一变化 ⇒ 键变化 ⇒ **WF 与生产不再互相覆写**。
    """
    man = build_manifest({'mode': mode, 'community_ids': community_ids})
    return f"{man['fingerprint']}{community_fingerprint(community_ids)}"


def write_manifest(extra=None, path=None):
    """落盘运行清单；失败只警告不阻断（清单是辅助，不是主流程依赖）。"""
    man = build_manifest(extra)
    p = path or MANIFEST_PATH
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(man, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, p)
    except Exception:
        pass
    return man


def load_manifest(path=None):
    p = path or MANIFEST_PATH
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None