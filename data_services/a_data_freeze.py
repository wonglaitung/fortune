#!/usr/bin/env python3
"""A 股外部数据冻结机制（对齐 ml_services/us_market_data.py 的快照模式）

背景（2026-10-03 普查 + C闸 实测）
----------------------------------
A 股 walk-forward 有 10 个实时网络数据源、其中 7 个无冻结机制。三样本双跑
md5 全不相同（72dc042c / 4cc437ee / ec1bd17d），净IR 2.14~2.72、
PBO 0.30~0.61（跨越 0.5 门槛）、判定 🟢/🟡 翻转 → 回测不可复现。

根因链条（三要素叠加）：
  1. 缓存会过期（主力资金/融资融券各 6 小时；个股日线 7 天）
  2. 上游接口不稳定（东财 RemoteDisconnected 频发）
  3. 失败后**静默填默认值**（0 / -1 / 跳过），于是"接口通不通"直接决定特征值

单独修任何一条都不够：即使缓存永不过期，首次抓取失败仍会污染；
即使不填默认值，缓存过期后重抓仍会变。

本模块提供统一的"冻结"开关：设置 A_STOCK_SNAPSHOT_DIR 后，各数据源
一律优先读写该目录并**跳过 TTL 校验**，使 walk-forward 可复现
（对应 lessons 三.25/ 三.29 对港股做的事）。

用法
----
    export A_STOCK_SNAPSHOT_DIR=data/a_stock_snapshot
    bash scripts/run_a_stock_walk_forward.sh --horizon 20 ...

未设置该变量时行为与从前完全一致（不影响生产路径）。
"""
import os

import pandas as pd

SNAPSHOT_ENV = 'A_STOCK_SNAPSHOT_DIR'
DEFAULT_SNAPSHOT_DIR = 'data/a_stock_snapshot'


def is_frozen() -> bool:
    """是否处于冻结模式"""
    return bool(os.environ.get(SNAPSHOT_ENV))


def snapshot_dir() -> str:
    return os.environ.get(SNAPSHOT_ENV) or DEFAULT_SNAPSHOT_DIR


def frozen_path(name: str) -> str:
    return os.path.join(snapshot_dir(), f'{name}.pkl')


def load_frozen(name: str):
    """读取冻结数据。

    Returns:
        (found: bool, data)
        found=False 表示「无冻结记录」；found=True 且 data 为 None 表示
        **已冻结的失败哨兵**——即当初抓取确实失败，调用方应稳定地按
        「该源不可用」处理，且不得重试网络（否则两次运行结果可能不同）。
    """
    if not is_frozen():
        return False, None
    path = frozen_path(name)
    # 失败哨兵优先：冻结「当初抓取失败」这一事实，不再重试网络
    if os.path.exists(path + '.failed'):
        return True, None
    if not os.path.exists(path):
        return False, None
    try:
        return True, pd.read_pickle(path)
    except Exception:
        return False, None


def save_failure(name: str) -> bool:
    """写入失败哨兵：冻结「当初抓取失败」这一事实"""
    if not is_frozen():
        return False
    os.makedirs(snapshot_dir(), exist_ok=True)
    try:
        pd.DataFrame().to_pickle(frozen_path(name) + '.failed.tmp')
        os.replace(frozen_path(name) + '.failed.tmp',
                   frozen_path(name) + '.failed')
        return True
    except Exception:
        return False


def save_frozen(name: str, df) -> bool:
    """写入冻结数据；非冻结模式或写失败返回 False"""
    if not is_frozen() or df is None or getattr(df, 'empty', False):
        return False
    directory = snapshot_dir()
    os.makedirs(directory, exist_ok=True)
    try:
        # 原子写：先写临时文件再改名，避免并发/中断产生半截文件
        # （lessons 三.25：非原子写会导致不可复现）
        tmp = frozen_path(name) + '.tmp'
        df.to_pickle(tmp)
        os.replace(tmp, frozen_path(name))
        return True
    except Exception:
        return False


def resolve(source_name: str, fetch_fn, ttl_seconds: int = 0,
            log=None) -> pd.DataFrame:
    """统一取数入口：冻结优先 → TTL 缓存 → 实时抓取

    Args:
        source_name: 数据源名（用作冻结文件名）
        fetch_fn: 无参可调用对象，返回 DataFrame；失败应返回 None
        ttl_seconds: 非冻结模式下的缓存有效期（秒）；<=0 表示不缓存
        log: 可选 logger

    Returns:
        DataFrame 或 None（**不做静默填默认值** —— 由调用方显式处理并记录）
    """
    # 1) 冻结模式：只读冻结文件，绝不联网
    if is_frozen():
        found, frozen = load_frozen(source_name)
        if found:
            if frozen is None:
                if log:
                    log.info(f'[freeze] {source_name} 命中失败哨兵 → 该源按不可用处理')
                return None
            return frozen
        df = fetch_fn()
        if df is not None and not getattr(df, 'empty', False):
            save_frozen(source_name, df)
            if log:
                log.info(f'[freeze] 已写入冻结数据 {source_name}（{len(df)} 行）')
            return df
        # 抓取失败：写哨兵冻结「失败」这一事实，避免下次重试成功导致结果变化
        save_failure(source_name)
        if log:
            log.warning(
                f'[freeze] {source_name} 抓取失败 → 已写入失败哨兵，'
                f'后续运行将稳定按不可用处理（结果可复现，但该源无贡献）')
        return None

    # 2) 非冻结：保持原行为（调用方自管 TTL），此处直接抓
    return fetch_fn()