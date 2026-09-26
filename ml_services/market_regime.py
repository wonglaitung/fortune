#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场情绪过滤器 - 使用滞后数据避免前瞻性偏差

功能：
- 使用滞后1天的市场上涨比例识别极端市场环境
- 动态调整预测阈值，在极端市场时提高门槛
- 支持批量预测，O(1) 查询复杂度

使用方法：
    from ml_services.market_regime import MarketSentimentFilter

    # 初始化
    filter = MarketSentimentFilter(lookback_days=1)

    # 预计算（在 Walk-Forward 开始前调用一次）
    filter.prepare_market_schedule(returns_df)

    # 预测时获取动态阈值
    threshold, layer, up_ratio = filter.get_threshold(predict_date)
"""

import pandas as pd
import numpy as np
import glob
import os
import re
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 分位门槛（b 方案）：bear/weak 的准入门槛不再用绝对概率，而取
# 校准后概率分布的分位点——抗校准器重拟漂移，且避免 Isotonic 阶梯造成的
# 0.60/0.65 空转（两门槛通过率完全相同的问题）。
#   bear  = P92 → 前约8% 通过
#   weak  = P90 → 前约10% 通过
#   normal 保持绝对 0.50（硬约束语义：胜率>50%，与分位无关）
# ---------------------------------------------------------------------------
GATE_QUANTILES = {'bear': 0.92, 'weak': 0.90}
GATE_FALLBACK = {'bear': 0.70, 'weak': 0.65}   # 快照/CSV 全无时的最终回退（原绝对值）
GATE_MIN_SAMPLES = 200
# 分位快照：CI checkout 无本地 output/ 时用内嵌快照，保证 CI 与本地同值。
# 由 output/20260925_044407_catboost_20d（43,610条，as_of=2026-09-25）算出。
# ⚠️ walk-forward 重跑后分位会漂移 → 重跑完执行
#    `python3 -c "from ml_services.market_regime import suggest_gate_snapshot; suggest_gate_snapshot()"`
#    并更新本常量（progress.txt 记录）。
GATE_SNAPSHOT = {'bear': 0.8636363636363636, 'weak': 0.6923076923076923}
GATE_SNAPSHOT_AS_OF = '2026-07-31'

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 分位数据源：最新一次 walk-forward 回测的 20d 预测分布（43k+ 样本、含 Date 可 PIT、
# 与评估口径同源）。不用 prediction_history——自选股收市批量预测右尾过窄，
# 其前8%分位仅 0.54，会低于 0.55 买入线导致门槛失效（2026-09-25 实测）。
_GATE_QUANTILE_GLOB = os.path.join(_BASE_DIR, 'output', '*_catboost_20d',
                                   'prediction_analysis.csv')
_GATE_SOURCE_RE = re.compile(r'/\d{8}_\d{6}_catboost_20d/prediction_analysis\.csv$')
_GATE_CALIBRATOR_FILE = os.path.join(_BASE_DIR, 'data', 'calibrators', 'prob_cal_20.pkl')


def _latest_gate_source_csv() -> Optional[str]:
    """最新港股回测 CSV。正则排除 *_a_stock_catboost_20d 等非港股目录，
    保证 CI（无本地 untracked 目录）与本地选择一致。"""
    files = [f for f in glob.glob(_GATE_QUANTILE_GLOB) if _GATE_SOURCE_RE.search(f)]
    return max(files) if files else None


def _gates_fallback(reason: str) -> Dict[str, float]:
    """回退链：内嵌快照（CI/无输出目录）→ 绝对值。"""
    if GATE_SNAPSHOT:
        logger.info("分位门槛：%s，使用内嵌快照 %s (as_of=%s)",
                    reason, GATE_SNAPSHOT, GATE_SNAPSHOT_AS_OF)
        return dict(GATE_SNAPSHOT)
    logger.warning("分位门槛：%s，回退绝对阈值 %s", reason, dict(GATE_FALLBACK))
    return dict(GATE_FALLBACK)


def suggest_gate_snapshot(as_of: Optional[str] = None) -> str:
    """walk-forward 重跑后打印建议的 GATE_SNAPSHOT 字符串（人工粘贴更新）。"""
    gates = compute_gate_thresholds(as_of=as_of, use_snapshot_fallback=False)
    src = _latest_gate_source_csv()
    print(f"GATE_SNAPSHOT = {gates}")
    print(f"# 源: {src}  as_of={as_of or GATE_SNAPSHOT_AS_OF}")


def compute_gate_thresholds(as_of: Optional[str] = None,
                            source_csv: Optional[str] = None,
                            calibrator_file: Optional[str] = None,
                            quantiles: Optional[Dict[str, float]] = None,
                            min_samples: int = GATE_MIN_SAMPLES,
                            use_snapshot_fallback: bool = True) -> Dict[str, float]:
    """按 as_of（PIT，含当日）计算 bear/weak 分位门槛。

    数据流：最新 walk-forward 回测 CSV（20d）中 Date<=as_of 的 Predict_Prob
    → Isotonic 校准（prob_cal_20）→ 分位数。

    Returns:
        {layer: threshold}；样本不足或任何失败时按
        内嵌快照 GATE_SNAPSHOT → 绝对值 GATE_FALLBACK 回退链处理。
    """
    quantiles = quantiles or GATE_QUANTILES

    def _fb(reason: str) -> Dict[str, float]:
        if use_snapshot_fallback:
            return _gates_fallback(reason)
        logger.warning("分位门槛：%s，回退绝对阈值 %s", reason, dict(GATE_FALLBACK))
        return {layer: GATE_FALLBACK.get(layer, 0.50) for layer in quantiles}

    source_csv = source_csv or _latest_gate_source_csv()
    calibrator_file = calibrator_file or _GATE_CALIBRATOR_FILE
    try:
        if not (source_csv and os.path.exists(source_csv) and os.path.exists(calibrator_file)):
            return _fb("回测CSV或校准器缺失")
        df = pd.read_csv(source_csv, usecols=['Date', 'Predict_Prob'])
        df = df.dropna()
        if as_of:
            df = df[df['Date'].astype(str) <= str(as_of)[:10]]
        probs = df['Predict_Prob'].to_numpy(dtype=float)
        if probs.size < min_samples:
            return _fb(f"PIT样本 {probs.size} < {min_samples}")
        import joblib
        iso = joblib.load(calibrator_file)
        cal = np.asarray(iso.predict(probs.reshape(-1, 1)), dtype=float)
        gates = {layer: float(np.percentile(cal, q * 100.0))
                 for layer, q in quantiles.items()}
        logger.info("分位门槛（as_of=%s, n=%d, src=%s）: %s",
                    as_of or 'latest', probs.size, os.path.basename(os.path.dirname(source_csv)),
                    {k: round(v, 4) for k, v in gates.items()})
        return gates
    except Exception as e:
        return _fb(f"计算失败（{e}）")


class MarketSentimentFilter:
    """
    市场情绪过滤器 - 使用滞后数据，支持批量预测

    核心原理：
    - 市场上涨比例有强自相关性（lag=1 自相关系数约 0.93）
    - 滞后1天数据能有效识别极端市场环境（精确率80%，召回率80%）
    - 在极端市场时提高预测阈值，减少 False Positive

    阈值分层：
    - extreme_bear (<20%): 暂停交易（阈值=1.0）
    - bear (20-30%): 高置信（阈值=校准概率 P92 分位，前约8%；回退 0.70）
    - weak (30-40%): 谨慎（阈值=校准概率 P90 分位，前约10%；回退 0.65）
    - normal (>40%): 标准（阈值=0.50，硬约束语义，保持绝对值）

    bear/weak 门槛按 prediction_history 的 PIT 校准概率分布动态取分位
    （compute_gate_thresholds），抗校准器重拟漂移；样本不足回退绝对值。
    """

    DEFAULT_LAYERS = {
        'extreme_bear': (0.20, 1.0),   # <20%: 暂停交易
        'bear': (0.30, 0.70),          # 20-30%: 高置信（回退阈值）
        'weak': (0.40, 0.65),          # 30-40%: 谨慎（回退阈值）
        'normal': (1.0, 0.50),         # >40%: 标准
    }

    def __init__(
        self,
        threshold_layers: Optional[Dict[str, Tuple[float, float]]] = None,
        lookback_days: int = 1,
        default_threshold: float = 0.50,
        use_quantile_gates: bool = True
    ):
        """
        初始化市场情绪过滤器

        Args:
            threshold_layers: 阈值分层配置，格式为 {layer_name: (upper_bound, threshold)}
            lookback_days: 滞后天数（默认1天）
            default_threshold: 默认阈值（当数据缺失时使用）
            use_quantile_gates: bear/weak 门槛是否按校准概率分位动态计算（默认开启）
        """
        self.lookback_days = lookback_days
        self.default_threshold = default_threshold
        self.threshold_layers = threshold_layers or self.DEFAULT_LAYERS
        self.use_quantile_gates = use_quantile_gates

        # 预计算缓存：{date: (up_ratio, threshold, layer_name)}
        self._daily_cache: Dict[str, Tuple[float, float, str]] = {}

        # 分位门槛基准（惰性一次性加载）与 per-date 缓存
        self._gate_dates: Optional[np.ndarray] = None
        self._gate_cal: Optional[np.ndarray] = None
        self._gate_base_ok = False
        self._gate_cache: Dict[str, Dict[str, float]] = {}

        logger.info(f"初始化 MarketSentimentFilter")
        logger.info(f"  滞后天数: {lookback_days}")
        logger.info(f"  默认阈值: {default_threshold}")
        logger.info(f"  阈值分层: {self.threshold_layers}")
        logger.info(f"  分位门槛: {use_quantile_gates and GATE_QUANTILES or '关闭'}")

    def _prepare_gate_base(self) -> None:
        """一次性加载回测 20d 校准概率基准（Date 升序），供 per-date PIT 分位。"""
        if self._gate_dates is not None:
            return
        try:
            source_csv = _latest_gate_source_csv()
            if not (source_csv and os.path.exists(_GATE_CALIBRATOR_FILE)):
                raise FileNotFoundError("回测 CSV 或 prob_cal_20 缺失")
            df = pd.read_csv(source_csv, usecols=['Date', 'Predict_Prob']).dropna()
            if len(df) < GATE_MIN_SAMPLES:
                raise ValueError(f"回测记录 {len(df)} < {GATE_MIN_SAMPLES}")
            import joblib
            probs = df['Predict_Prob'].to_numpy(dtype=float)
            dates = df['Date'].astype(str).str[:10].to_numpy()
            iso = joblib.load(_GATE_CALIBRATOR_FILE)
            cal = np.asarray(iso.predict(probs.reshape(-1, 1)), dtype=float)
            order = np.argsort(dates, kind='stable')
            self._gate_dates = dates[order]
            self._gate_cal = cal[order]
            self._gate_base_ok = True
            logger.info(f"分位门槛基准已加载: {len(dates)} 条回测记录 ({source_csv})")
        except Exception as e:
            logger.warning(f"分位门槛基准加载失败（{e}），bear/weak 回退绝对阈值")
            self._gate_base_ok = False

    def _gates_for(self, date_str: str) -> Dict[str, float]:
        """PIT 分位门槛：只用 data_date <= date_str 的记录。空 dict = 用回退值。"""
        if not self.use_quantile_gates:
            return {}
        cached = self._gate_cache.get(date_str)
        if cached is not None:
            return cached
        gates: Dict[str, float] = {}
        self._prepare_gate_base()
        if self._gate_base_ok:
            idx = int(np.searchsorted(self._gate_dates, date_str, side='right'))
            if idx >= GATE_MIN_SAMPLES:
                sample = self._gate_cal[:idx]
                gates = {layer: float(np.percentile(sample, q * 100.0))
                         for layer, q in GATE_QUANTILES.items()}
        if not gates:
            # 基准不可用（CI 无 output/）或 PIT 样本不足 → 内嵌快照 → 绝对值
            # （_prepare_gate_base 失败时已 warning，此处不再刷 4470 条日志）
            gates = dict(GATE_SNAPSHOT) if GATE_SNAPSHOT else dict(GATE_FALLBACK)
        self._gate_cache[date_str] = gates
        return gates

    def prepare_market_schedule(
        self,
        returns_df: pd.DataFrame,
        date_col: str = 'Date',
        ret_col: str = 'Return_1d'
    ) -> None:
        """
        预计算所有交易日的上涨比例与阈值

        在 Walk-Forward 开始前调用一次，避免在 predict 中重复查询全量数据

        Args:
            returns_df: 收益率数据，包含日期和收益率列
            date_col: 日期列名
            ret_col: 收益率列名
        """
        logger.info(f"开始预计算市场情绪...")

        # 确保日期列是 datetime 类型
        if not pd.api.types.is_datetime64_any_dtype(returns_df[date_col]):
            returns_df = returns_df.copy()
            returns_df[date_col] = pd.to_datetime(returns_df[date_col])

        # 1. 按日期分组计算上涨比例
        daily_stats = returns_df.groupby(date_col)[ret_col].apply(
            lambda x: (x > 0).mean()
        ).sort_index()

        logger.info(f"  计算了 {len(daily_stats)} 个交易日的上涨比例")

        # 2. 滞后 shift
        lagged_up_ratio = daily_stats.shift(self.lookback_days)

        # 3. 生成每日阈值映射
        self._daily_cache.clear()

        for date, up_ratio in lagged_up_ratio.items():
            date_str = date.strftime('%Y-%m-%d')

            if pd.isna(up_ratio):
                self._daily_cache[date_str] = (0.5, self.default_threshold, 'unknown')
                continue

            # 根据阈值分层确定阈值
            threshold = self.default_threshold
            layer_name = 'normal'

            for layer, (upper_bound, thresh) in self.threshold_layers.items():
                if up_ratio < upper_bound:
                    threshold = thresh
                    layer_name = layer
                    break

            # bear/weak 门槛分位化（PIT）：失败/样本不足时保留上面的回退绝对值
            gates = self._gates_for(date_str)
            if layer_name in gates:
                threshold = gates[layer_name]

            self._daily_cache[date_str] = (float(up_ratio), threshold, layer_name)

        logger.info(f"  预计算完成，覆盖 {len(self._daily_cache)} 个交易日")

        # 统计各层级分布
        layer_counts = {}
        for _, (_, _, layer) in self._daily_cache.items():
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        logger.info(f"  层级分布: {layer_counts}")

    def get_threshold(self, predict_date: str) -> Tuple[float, str, float]:
        """
        预测时调用：O(1) 复杂度获取当日动态阈值

        Args:
            predict_date: 预测日期（格式：YYYY-MM-DD 或 datetime）

        Returns:
            Tuple[float, str, float]: (阈值, 层级名称, 滞后上涨比例)
        """
        # 处理日期格式
        if isinstance(predict_date, str):
            date_str = predict_date[:10]  # 取前10个字符（YYYY-MM-DD）
        else:
            date_str = pd.to_datetime(predict_date).strftime('%Y-%m-%d')

        if date_str not in self._daily_cache:
            logger.warning(f"日期 {date_str} 无市场情绪缓存，使用默认阈值 {self.default_threshold}")
            return self.default_threshold, 'fallback', 0.5

        up_ratio, threshold, layer = self._daily_cache[date_str]
        return threshold, layer, up_ratio

    def apply_filter(
        self,
        predictions_df: pd.DataFrame,
        date_col: str = 'Date',
        prob_col: str = 'Predict_Prob',
        direction_col: str = 'Predict_Direction'
    ) -> pd.DataFrame:
        """
        对预测信号应用市场情绪过滤

        Args:
            predictions_df: 预测结果 DataFrame
            date_col: 日期列名
            prob_col: 预测概率列名
            direction_col: 预测方向列名

        Returns:
            pd.DataFrame: 过滤后的预测结果，新增以下列：
                - market_up_ratio_lag1: 滞后1天上涨比例
                - dynamic_threshold: 动态阈值
                - market_layer: 市场层级
                - filtered_signal: 过滤后信号（0/1）
        """
        if not self._daily_cache:
            raise ValueError("请先调用 prepare_market_schedule() 预计算市场情绪")

        results = []

        for _, row in predictions_df.iterrows():
            date_str = row[date_col]
            if isinstance(date_str, str):
                date_str = date_str[:10]
            else:
                date_str = pd.to_datetime(date_str).strftime('%Y-%m-%d')

            threshold, layer, up_ratio = self.get_threshold(date_str)

            # 判断是否保留信号
            prob = row[prob_col]
            direction = row[direction_col]

            # 只有预测上涨且概率超过阈值才保留
            should_keep = (direction == 'UP' or direction == 1) and (prob >= threshold)

            results.append({
                **row.to_dict(),
                'market_up_ratio_lag1': up_ratio,
                'dynamic_threshold': threshold,
                'market_layer': layer,
                'filtered_signal': int(should_keep)
            })

        return pd.DataFrame(results)

    def get_filter_stats(self) -> Dict[str, int]:
        """
        获取过滤统计信息

        Returns:
            Dict[str, int]: 各层级的交易日数量
        """
        layer_counts = {}
        for _, (_, _, layer) in self._daily_cache.items():
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        return layer_counts


def create_market_filter_from_stock_data(
    stock_data: pd.DataFrame,
    date_col: str = 'Date',
    close_col: str = 'Close',
    lookback_days: int = 1
) -> MarketSentimentFilter:
    """
    从股票数据创建市场情绪过滤器

    Args:
        stock_data: 股票数据 DataFrame，包含日期和收盘价
        date_col: 日期列名
        close_col: 收盘价列名
        lookback_days: 滞后天数

    Returns:
        MarketSentimentFilter: 市场情绪过滤器实例
    """
    # 计算收益率
    returns_df = stock_data[[date_col, close_col]].copy()
    returns_df['Return_1d'] = returns_df.groupby(date_col)[close_col].pct_change()

    # 创建过滤器
    market_filter = MarketSentimentFilter(lookback_days=lookback_days)
    market_filter.prepare_market_schedule(returns_df, date_col=date_col, ret_col='Return_1d')

    return market_filter
