#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场状态检测模块（Hidden Markov Model）

使用 HMM 识别港股市场的 3 种状态：
- 状态 0：低波动震荡（Sideways/Consolidation）
- 状态 1：上涨趋势（Bull/Uptrend）
- 状态 2：下跌趋势（Bear/Downtrend）

输出特征：
- Market_Regime: 当前市场状态（0/1/2）
- Regime_Prob_0/1/2: 各状态的概率
- Regime_Duration: 当前状态已持续的交易日数
- Regime_Transition_Prob: 状态转换概率（从当前状态转换的概率）

新增特征（2026-04-27 Tier 1 增强）：
- Regime_Switch_Prob_5d: 5天内转换到不同状态的概率
- Regime_Expected_Duration: 当前状态期望剩余持续时间
- Regime_Momentum: 状态概率 5 日变化（增强/减弱）
- Regime_Vol_Interaction: 高波动+高转换=动荡期

依赖：hmmlearn 库（pip install hmmlearn）
"""

import os
import sys
import warnings
import pickle
import pandas as pd
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings('ignore')

# 模型缓存
CACHE_DIR = 'data/regime_cache'
MODEL_FILE = os.path.join(CACHE_DIR, 'hmm_regime_model.pkl')


class RegimeDetector:
    """基于 HMM 的市场状态检测器"""

    # 状态标签（训练后根据特征均值排序确定）
    REGIME_LABELS = {
        0: '震荡',   # 低波动、收益率接近 0
        1: '上涨',   # 正收益、中等波动
        2: '下跌',   # 负收益、高波动
    }

    def __init__(self, n_states=3, lookback=252):
        """
        参数:
        - n_states: 隐状态数量
        - lookback: 用于训练的回看窗口（交易日），默认1年
        """
        self.n_states = n_states
        self.lookback = lookback
        self.model = None
        self._state_mapping = None  # 原始状态到语义状态的映射

    def _prepare_observations(self, df):
        """
        准备 HMM 观测序列

        使用 3 个观测变量：
        1. 20日收益率（趋势方向）
        2. 20日波动率（波动水平）
        3. 成交量变化率（资金活跃度）
        """
        # 计算原始特征
        returns = df['Close'].pct_change()
        vol_20d = returns.rolling(window=20).std()
        return_20d = returns.rolling(window=20).sum()
        volume_change = df['Volume'].pct_change().rolling(window=5).mean()

        # 组合观测矩阵
        obs = pd.DataFrame({
            'return_20d': return_20d,
            'vol_20d': vol_20d,
            'volume_change': volume_change,
        }, index=df.index)

        # 标准化（Z-Score，避免极端值影响）
        for col in obs.columns:
            mean = obs[col].rolling(window=self.lookback, min_periods=60).mean()
            std = obs[col].rolling(window=self.lookback, min_periods=60).std()
            obs[col] = (obs[col] - mean) / (std + 1e-10)

        return obs

    def fit(self, df):
        """
        训练 HMM 模型

        参数:
        - df: 包含 Close 和 Volume 列的 DataFrame

        返回:
        - self
        """
        from hmmlearn.hmm import GaussianHMM

        obs = self._prepare_observations(df)
        # 去除 NaN
        obs_clean = obs.dropna()

        if len(obs_clean) < 100:
            raise ValueError(f"训练数据不足（{len(obs_clean)} 条），需要至少 100 条")

        # 训练 HMM
        # 多次随机初始化取最优
        best_model = None
        best_score = -np.inf

        for seed in [42, 123, 456]:
            try:
                model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type='full',
                    n_iter=200,
                    random_state=seed,
                    tol=0.01,
                )
                model.fit(obs_clean.values)
                score = model.score(obs_clean.values)
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception:
                continue

        if best_model is None:
            raise ValueError("HMM 训练失败，所有随机种子均不收敛")

        self.model = best_model

        # 确定状态语义映射
        self._determine_state_mapping(obs_clean)

        # 缓存模型
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'state_mapping': self._state_mapping,
            }, f)

        print(f"  ✅ HMM 模型训练完成（log-likelihood={best_score:.2f}）")

        return self

    def _determine_state_mapping(self, obs_clean):
        """
        根据 HMM 状态的均值特征，确定每个状态的语义含义

        约定：
        - 收益率最高 + 波动率中等 → 上涨（1）
        - 收益率最低 + 波动率偏高 → 下跌（2）
        - 收益率接近0 + 波动率低 → 震荡（0）
        """
        means = self.model.means_  # shape: (n_states, n_features)
        # 第0列 = return_20d 均值，第1列 = vol_20d 均值

        # 按收益率均值排序
        sorted_by_return = np.argsort(means[:, 0])

        # 最低收益 → 下跌(2)
        # 最高收益 → 上涨(1)
        # 中间 → 震荡(0)
        self._state_mapping = {}
        self._state_mapping[sorted_by_return[0]] = 2  # 最低收益 → 下跌
        self._state_mapping[sorted_by_return[-1]] = 1  # 最高收益 → 上涨

        # 中间状态
        if self.n_states == 3:
            self._state_mapping[sorted_by_return[1]] = 0  # 中间 → 震荡
        else:
            # 多于3个状态时，按波动率分
            remaining = [s for s in range(self.n_states) if s not in self._state_mapping]
            for s in remaining:
                self._state_mapping[s] = 0  # 统一归为震荡

    def predict(self, df):
        """
        预测市场状态

        参数:
        - df: 包含 Close 和 Volume 列的 DataFrame

        返回:
        - DataFrame: 包含状态预测结果
        """
        if self.model is None:
            # 尝试加载缓存模型
            self._load_model()
            if self.model is None:
                raise ValueError("模型未训练且无缓存")

        obs = self._prepare_observations(df)
        obs_clean = obs.dropna()

        if len(obs_clean) == 0:
            return pd.DataFrame()

        # 预测状态序列
        states_raw = self.model.predict(obs_clean.values)

        # 计算状态概率
        post_probs = self.model.predict_proba(obs_clean.values)

        # 映射到语义状态
        states_mapped = np.array([self._state_mapping[s] for s in states_raw])

        # 构建结果 DataFrame
        result = pd.DataFrame(index=obs_clean.index)
        result['Market_Regime'] = states_mapped
        for i in range(self.n_states):
            mapped_state = self._state_mapping[i]
            result[f'Regime_Prob_{mapped_state}'] = post_probs[:, i]

        # 重新排列概率列顺序（0=震荡, 1=上涨, 2=下跌）
        prob_cols = {}
        for i in range(self.n_states):
            mapped_state = self._state_mapping[i]
            if f'Regime_Prob_{mapped_state}' not in prob_cols:
                prob_cols[f'Regime_Prob_{mapped_state}'] = post_probs[:, i]
            else:
                # 多个原始状态映射到同一语义状态，累加概率
                prob_cols[f'Regime_Prob_{mapped_state}'] += post_probs[:, i]

        for col, values in prob_cols.items():
            result[col] = values

        # 计算状态持续时间
        result['Regime_Duration'] = self._calculate_duration(states_mapped)

        # 状态转换概率（从当前状态转换到其他状态的总概率）
        trans_mat = self.model.transmat_
        result['Regime_Transition_Prob'] = 0.0
        for i, state in enumerate(states_raw):
            mapped = self._state_mapping[state]
            # 留在当前状态的概率
            stay_prob = trans_mat[state, state]
            result.iloc[i, result.columns.get_loc('Regime_Transition_Prob')] = 1 - stay_prob

        # ========== 新增 Tier 1 特征（2026-04-27）==========
        # 1. Regime_Switch_Prob_5d: 5天内转换到不同状态的概率
        # 使用 T^5 转移矩阵计算
        trans_mat_5d = np.linalg.matrix_power(trans_mat, 5)
        result['Regime_Switch_Prob_5d'] = 0.0
        for i, state in enumerate(states_raw):
            # 5天后留在当前状态的概率
            stay_prob_5d = trans_mat_5d[state, state]
            result.iloc[i, result.columns.get_loc('Regime_Switch_Prob_5d')] = 1 - stay_prob_5d

        # 2. Regime_Expected_Duration: 当前状态期望剩余持续时间
        # 期望持续时间 = 1 / (1 - T[i,i])
        result['Regime_Expected_Duration'] = 0.0
        for i, state in enumerate(states_raw):
            stay_prob = trans_mat[state, state]
            if stay_prob < 1.0:
                expected_duration = 1.0 / (1.0 - stay_prob)
            else:
                expected_duration = 100.0  # 上限
            result.iloc[i, result.columns.get_loc('Regime_Expected_Duration')] = expected_duration

        return result

    def _calculate_duration(self, states):
        """计算状态持续时间（当前状态已持续的天数）"""
        duration = np.zeros(len(states))
        current_duration = 1

        for i in range(len(states)):
            if i == 0:
                duration[i] = 1
            elif states[i] == states[i - 1]:
                current_duration += 1
                duration[i] = current_duration
            else:
                current_duration = 1
                duration[i] = 1

        return duration

    def _load_model(self):
        """从缓存加载模型"""
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    cache = pickle.load(f)
                self.model = cache['model']
                self._state_mapping = cache['state_mapping']
                return True
            except Exception:
                return False
        return False

    def calculate_features(self, df):
        """
        计算市场状态特征（集成接口）

        参数:
        - df: 包含 Close 和 Volume 列的 DataFrame

        返回:
        - DataFrame: 添加了状态特征的 DataFrame
        """
        print("  🔄 计算市场状态特征（HMM）...")

        try:
            # 尝试加载缓存模型
            if self.model is None:
                self._load_model()

            # 如果没有缓存或缓存模型过期，重新训练
            if self.model is None:
                print("  📊 训练 HMM 模型...")
                self.fit(df)

            # 预测状态
            regime_df = self.predict(df)

            # 合并到主 DataFrame
            for col in self.get_feature_names():
                df[col] = np.nan

            common_idx = df.index.intersection(regime_df.index)
            for col in regime_df.columns:
                if col in df.columns:
                    df.loc[common_idx, col] = regime_df.loc[common_idx, col].values

            # ⚠️ 数据泄漏防护：HMM 预测的是基于截至 t-1 的信息
            # 但 predict 使用了全部数据，所以需要 shift(1)
            for col in ['Market_Regime', 'Regime_Prob_0', 'Regime_Prob_1',
                       'Regime_Prob_2', 'Regime_Duration', 'Regime_Transition_Prob',
                       'Regime_Switch_Prob_5d', 'Regime_Expected_Duration']:
                if col in df.columns:
                    df[col] = df[col].shift(1)

            # ========== 新增 Tier 1 特征计算 ==========
            # 3. Regime_Momentum: 状态概率 5 日变化
            # 上涨状态概率的变化（增强/减弱）
            if 'Regime_Prob_1' in df.columns:
                df['Regime_Momentum'] = df['Regime_Prob_1'].diff(5)
                # 已通过 shift(1) 处理的 Regime_Prob_1，diff(5) 不会泄漏

            # 4. Regime_Vol_Interaction: 高波动+高转换=动荡期
            # 需要 GARCH_Conditional_Vol（GARCH 模型先计算）
            if 'GARCH_Conditional_Vol' in df.columns and 'Regime_Transition_Prob' in df.columns:
                df['Regime_Vol_Interaction'] = (
                    df['GARCH_Conditional_Vol'] * df['Regime_Transition_Prob']
                )
            else:
                # 回退：使用 Volatility_20d
                if 'Volatility_20d' in df.columns and 'Regime_Transition_Prob' in df.columns:
                    df['Regime_Vol_Interaction'] = (
                        df['Volatility_20d'] * df['Regime_Transition_Prob']
                    )
                else:
                    df['Regime_Vol_Interaction'] = 0.0

            feature_count = len([c for c in df.columns if c in self.get_feature_names()])
            print(f"  ✅ 市场状态特征计算完成（{feature_count} 个特征）")

        except Exception as e:
            print(f"  ⚠️ HMM 市场状态检测失败: {e}，使用默认值")
            for col in self.get_feature_names():
                if col == 'Market_Regime':
                    df[col] = 0  # 默认震荡
                elif col == 'Regime_Prob_0':
                    df[col] = 0.5  # 默认50%概率震荡
                elif col.startswith('Regime_Prob_'):
                    df[col] = 0.25
                elif col == 'Regime_Expected_Duration':
                    df[col] = 10.0  # 默认期望持续时间
                else:
                    df[col] = 0.0

        return df

    @staticmethod
    def get_feature_names():
        """返回所有市场状态特征名"""
        return [
            'Market_Regime',
            'Regime_Prob_0',  # 震荡概率
            'Regime_Prob_1',  # 上涨概率
            'Regime_Prob_2',  # 下跌概率
            'Regime_Duration',
            'Regime_Transition_Prob',
            # 新增 Tier 1 特征（2026-04-27）
            'Regime_Switch_Prob_5d',      # 5天内转换概率
            'Regime_Expected_Duration',   # 期望剩余持续时间
            'Regime_Momentum',            # 状态概率变化
            'Regime_Vol_Interaction',     # 波动率与转换概率交互
        ]


# 市场状态特征配置
REGIME_FEATURE_CONFIG = {
    'regime_features': RegimeDetector.get_feature_names()
}
