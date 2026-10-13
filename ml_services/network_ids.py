# -*- coding: utf-8 -*-
"""社区 ID 集合解析 —— **全项目唯一入口**（D26）。

**为什么必须唯一**（lessons 三.60 差异② / 三.48）：
`create_market_network_interaction_features` 在 `community_ids=None` 时**退回每股动态提取**，
于是：

| 路径 | 传入 | 每只股票得到的 `net_comm_*` 列 | 缺列的值 |
|---|---|---|---|
| WF（此前不传） | None | 只有它自己的社区 | concat 后 **NaN**（实测 201/784 格） |
| 生产 train | 静态∪PIT | 全部社区 | 显式 **0** |

CatBoost 原生区分 NaN 与 0 ⇒ 两侧不是同一个函数。

而这段「静态 JSON → 并入 PIT → 得集合」的逻辑此前在
`ml_trading_model.py` 里**被复制了两份**（`train()` 与另一条路径），
各自独立演进 = 漂移温床。故抽出本模块，两处一并替换。

**使用契约**：
- `train()` 与 `walk_forward_validation` 的**全段预取**必须调用本函数
- `predict()` 与 WF 每折**测试**段**不得**调用——它们必须复用训练时保存的集合
  （pkl 里的 `community_ids` / `model.community_ids`），那才是模型稳定性来源
"""
import json
import os

STATIC_NETWORK_FILE = 'data/network_features/network_features_for_ml.json'
PIT_NETWORK_FILE = 'data/network_features/network_features_pit.json'


def resolve_community_ids(include_pit=True, static_file=None, pit_file=None, logger=None):
    """返回排序后的社区 ID 列表（静态 ∪ PIT）；取不到返回 None。

    返回 None 表示「无法解析」→ 调用方**必须**显式处理（不得静默退回动态提取），
    否则又回到 D26 修复前的状态。
    """
    static_file = static_file or STATIC_NETWORK_FILE
    pit_file = pit_file or PIT_NETWORK_FILE
    ids = set()

    if os.path.exists(static_file):
        try:
            with open(static_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for feats in data.values():
                if not isinstance(feats, dict):
                    continue
                cid = feats.get('net_community_id')
                if cid is not None and int(cid) >= 0:
                    ids.add(int(cid))
        except Exception as e:  # 解析失败不静默
            if logger:
                logger.warning(f'解析静态网络特征失败 {static_file}: {e}')

    if include_pit and os.path.exists(pit_file):
        try:
            with open(pit_file, 'r', encoding='utf-8') as f:
                pit = json.load(f)
            for by_date in pit.values():
                if not isinstance(by_date, dict):
                    continue
                for feats in by_date.values():
                    if not isinstance(feats, dict):
                        continue
                    cid = feats.get('net_community_id')
                    if cid is not None and int(cid) >= 0:
                        ids.add(int(cid))
        except Exception as e:
            if logger:
                logger.warning(f'解析 PIT 网络特征失败 {pit_file}: {e}')

    return sorted(ids) if ids else None