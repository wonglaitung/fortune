"""
行为金融因子模块

实现基于行为金融学理论的因子：
1. 凸显性因子 (Salience Theory Factor, STR) - 国信证券研究
2. 球队硬币因子 (Team Coin Factor) - 方正证券研究

参考文献：
- 国信证券：《凸显性因子：行为金融学在A股的创新应用》
- 方正证券：《球队硬币因子：基于体育博彩理论的选股因子》
- Hugo2046/QuantsPlaybook GitHub 项目
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)


class BehavioralFactorCalculator:
    """
    行为金融因子计算器

    凸显性因子基于 Kahneman-Tversky 前景理论：
    - 投资者对极端收益/损失给予过多关注（凸显效应）
    - 凸显性高的股票容易被过度关注，产生反转机会

    球队硬币因子基于体育博彩理论：
    - 市场存在"主队偏好"和"热门偏好"
    - 当市场过度偏向一方时，反向操作有利可图
    """

    @staticmethod
    def get_feature_names() -> List[str]:
        """获取所有行为金融因子名称"""
        return [
            # 凸显性因子
            'STR_5d',           # 5日凸显性
            'STR_20d',          # 20日凸显性
            'STR_Intensity',    # 凸显强度
            'STR_Signal',       # 凸显信号（综合）
            # 球队硬币因子
            'Team_Coin_Sentiment',    # 市场情绪偏向
            'Team_Coin_Reversal',     # 反转信号
            'Team_Coin_Intensity',    # 情绪强度
        ]

    @staticmethod
    def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有行为金融因子

        Args:
            df: 股票数据，必须包含 Open, High, Low, Close, Volume 列

        Returns:
            添加了行为金融因子的 DataFrame
        """
        df = df.copy()

        # 计算凸显性因子
        df = BehavioralFactorCalculator.calculate_salience_factor(df)

        # 计算球队硬币因子
        df = BehavioralFactorCalculator.calculate_team_coin_factor(df)

        return df

    @staticmethod
    def calculate_salience_factor(df: pd.DataFrame,
                                   windows: List[int] = [5, 20]) -> pd.DataFrame:
        """
        计算凸显性因子 (Salience Theory Factor, STR)

        理论基础（Kahneman-Tversky 前景理论）：
        - 投资者对极端结果给予更多关注（凸显效应）
        - 凸显性 = 最大收益的感知权重 - 最小收益的感知权重
        - 凸显性高的股票容易被过度关注，未来收益倾向于反转

        计算方法（国信证券）：
        1. 计算窗口期内每日收益
        2. 找出最大收益和最小收益
        3. 计算凸显性指数

        Args:
            df: 股票数据
            windows: 计算窗口列表

        Returns:
            添加了凸显性因子的 DataFrame
        """
        df = df.copy()

        # 计算日收益率
        df['Daily_Return'] = df['Close'].pct_change()

        for window in windows:
            # 滚动窗口内的最大收益和最小收益
            df[f'Max_Return_{window}d'] = df['Daily_Return'].rolling(window=window).max()
            df[f'Min_Return_{window}d'] = df['Daily_Return'].rolling(window=window).min()

            # 滚动窗口内的平均收益（绝对值）
            df[f'Avg_Abs_Return_{window}d'] = df['Daily_Return'].abs().rolling(window=window).mean()

            # 凸显性指数 = (最大收益 - 最小收益) / 平均绝对收益
            # 反映收益分布的极端程度
            denominator = df[f'Avg_Abs_Return_{window}d'].replace(0, np.nan)
            df[f'STR_{window}d'] = (df[f'Max_Return_{window}d'] - df[f'Min_Return_{window}d']) / denominator

            # 使用 shift(1) 避免数据泄漏
            df[f'STR_{window}d'] = df[f'STR_{window}d'].shift(1)

        # 凸显强度 = 短期凸显性 / 长期凸显性
        # 值 > 1 表示短期极端性增强
        if 5 in windows and 20 in windows:
            df['STR_Intensity'] = (df['STR_5d'] / df['STR_20d'].replace(0, np.nan)).shift(1)

        # 凸显信号（综合）
        # 结合不同窗口的凸显性，生成交易信号
        # 高凸显性 -> 可能过度关注 -> 反转机会
        df['STR_Signal'] = (
            df['STR_5d'].rank(pct=True) * 0.6 +  # 短期权重更高
            df['STR_20d'].rank(pct=True) * 0.4
        ).shift(1)

        # 清理中间变量
        cols_to_drop = ['Daily_Return']
        for window in windows:
            cols_to_drop.extend([
                f'Max_Return_{window}d',
                f'Min_Return_{window}d',
                f'Avg_Abs_Return_{window}d'
            ])

        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        logger.info(f"凸显性因子计算完成: STR_5d, STR_20d, STR_Intensity, STR_Signal")

        return df

    @staticmethod
    def calculate_team_coin_factor(df: pd.DataFrame,
                                    window: int = 20) -> pd.DataFrame:
        """
        计算球队硬币因子 (Team Coin Factor)

        理论基础（体育博彩理论）：
        - 投资者存在"主队偏好"，倾向于买入熟悉的股票
        - 当市场情绪过度偏向一方时，反向操作有利可图
        - 类似于体育博彩中"热门球队被高估"现象

        计算方法（方正证券）：
        1. 计算上涨天数占比（市场情绪偏向）
        2. 当上涨占比过高时，市场过度乐观，看空
        3. 当上涨占比过低时，市场过度悲观，看多

        Args:
            df: 股票数据
            window: 计算窗口

        Returns:
            添加了球队硬币因子的 DataFrame
        """
        df = df.copy()

        # 计算日收益方向：上涨为1，下跌为0
        df['Up_Day'] = (df['Close'] > df['Close'].shift(1)).astype(int)

        # 滚动窗口内上涨天数占比（市场情绪偏向）
        # 值接近1 = 过度乐观，值接近0 = 过度悲观
        df['Up_Ratio'] = df['Up_Day'].rolling(window=window).mean()

        # 使用 shift(1) 避免数据泄漏
        df['Up_Ratio'] = df['Up_Ratio'].shift(1)

        # 市场情绪偏向：偏离中性（0.5）的程度
        # 正值 = 过度乐观，负值 = 过度悲观
        df['Team_Coin_Sentiment'] = (df['Up_Ratio'] - 0.5) * 2  # 归一化到 [-1, 1]

        # 反转信号：当情绪极端时，反转信号强
        # 情绪 > 0.65（过度乐观）-> 反转信号为负
        # 情绪 < 0.35（过度悲观）-> 反转信号为正
        df['Team_Coin_Reversal'] = np.where(
            df['Up_Ratio'] > 0.65, -1,  # 过度乐观，看空
            np.where(
                df['Up_Ratio'] < 0.35, 1,  # 过度悲观，看多
                0  # 中性区间
            )
        )

        # 情绪强度：极端程度
        # 值越大，市场情绪越极端，反转机会越大
        df['Team_Coin_Intensity'] = np.abs(df['Team_Coin_Sentiment'])

        # 清理中间变量
        df = df.drop(columns=['Up_Day', 'Up_Ratio'])

        logger.info(f"球队硬币因子计算完成: Team_Coin_Sentiment, Team_Coin_Reversal, Team_Coin_Intensity")

        return df


# ==================== 辅助函数 ====================

def calculate_behavioral_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：计算所有行为金融因子

    Args:
        df: 股票数据

    Returns:
        添加了行为金融因子的 DataFrame
    """
    return BehavioralFactorCalculator.calculate_all_factors(df)


# ==================== 测试代码 ====================

if __name__ == '__main__':
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    test_df = pd.DataFrame({
        'Open': 100 + np.cumsum(np.random.randn(100) * 0.02),
        'High': 100 + np.cumsum(np.random.randn(100) * 0.02) + 0.5,
        'Low': 100 + np.cumsum(np.random.randn(100) * 0.02) - 0.5,
        'Close': 100 + np.cumsum(np.random.randn(100) * 0.02),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # 计算因子
    result = BehavioralFactorCalculator.calculate_all_factors(test_df)

    print("=" * 60)
    print("行为金融因子计算结果")
    print("=" * 60)

    # 显示因子统计
    for col in BehavioralFactorCalculator.get_feature_names():
        if col in result.columns:
            print(f"\n{col}:")
            print(f"  均值: {result[col].mean():.4f}")
            print(f"  标准差: {result[col].std():.4f}")
            print(f"  最小值: {result[col].min():.4f}")
            print(f"  最大值: {result[col].max():.4f}")

    print("\n" + "=" * 60)
    print("测试完成")
