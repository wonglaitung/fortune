#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股日历效应特征计算模块

提供港股特有的日历效应特征：
- 星期效应（Day of Week）
- 月份效应（Month of Year）
- 期权到期日效应（每月第四个周三）
- 月初/月末效应
- 季末窗口粉饰效应
- 长假前后效应（春节、国庆）
- 台风季效应
- 当月剩余交易日

数据来源：AKShare 交易日历
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 港股主要假期（每年需更新，此处定义固定规则 + 近年具体日期）
# 春节通常在1月下旬到2月中旬
# 国庆节：10月1-7日
# 圣诞节：12月25-26日
# 复活节：每年不同，春季3-4月

# 港股期权到期日：每月第四个周三（最后一个交易日前的周三）


class CalendarFeatureCalculator:
    """港股日历效应特征计算器"""

    # 港股特有假期定义（近3年的具体日期，远期用规则近似）
    HK_HOLIDAYS_2024 = [
        '2024-01-01',  # 元旦
        '2024-02-10', '2024-02-12', '2024-02-13',  # 春节
        '2024-03-29', '2024-03-30',  # 复活节
        '2024-04-04',  # 清明节
        '2024-05-01',  # 劳动节
        '2024-05-15',  # 佛诞
        '2024-06-10',  # 端午节
        '2024-07-01',  # 回归日
        '2024-09-18',  # 中秋节翌日
        '2024-10-01',  # 国庆节
        '2024-10-11',  # 重阳节
        '2024-12-25', '2024-12-26',  # 圣诞节
    ]

    HK_HOLIDAYS_2025 = [
        '2025-01-01',  # 元旦
        '2025-01-29', '2025-01-30', '2025-01-31',  # 春节
        '2025-04-04',  # 清明节
        '2025-04-18', '2025-04-19',  # 复活节
        '2025-05-01',  # 劳动节
        '2025-05-05',  # 佛诞
        '2025-05-31',  # 端午节
        '2025-07-01',  # 回归日
        '2025-10-01',  # 国庆节
        '2025-10-07',  # 重阳节
        '2025-10-29',  # 中秋节翌日
        '2025-12-25', '2025-12-26',  # 圣诞节
    ]

    HK_HOLIDAYS_2026 = [
        '2026-01-01',  # 元旦
        '2026-02-17', '2026-02-18', '2026-02-19',  # 春节
        '2026-04-03',  # 清明节
        '2026-04-05', '2026-04-06',  # 复活节
        '2026-05-01',  # 劳动节
        '2026-05-24',  # 佛诞
        '2026-06-19',  # 端午节
        '2026-07-01',  # 回归日
        '2026-10-01',  # 国庆节
        '2026-10-25',  # 重阳节
        '2026-10-06',  # 中秋节翌日
        '2026-12-25', '2026-12-26',  # 圣诞节
    ]

    def __init__(self):
        self._trading_days = None
        self._holidays = None

    def _get_holidays(self):
        """获取合并的假期列表"""
        if self._holidays is None:
            all_holidays = (
                self.HK_HOLIDAYS_2024 +
                self.HK_HOLIDAYS_2025 +
                self.HK_HOLIDAYS_2026
            )
            self._holidays = pd.to_datetime(all_holidays)
        return self._holidays

    def _get_trading_days(self):
        """获取交易日历（使用 AKShare）"""
        if self._trading_days is not None:
            return self._trading_days

        try:
            import akshare as ak
            # 获取A股交易日历（港股交易日基本与A股重合，主要差异在港股特有假期）
            df = ak.tool_trade_date_hist_sina()
            self._trading_days = pd.to_datetime(df['trade_date'])
        except Exception as e:
            print(f"  ⚠️ AKShare交易日历获取失败: {e}，使用简单估算")
            # 回退：生成简单的工作日日历
            dates = pd.date_range('2020-01-01', '2026-12-31', freq='B')
            self._trading_days = dates

        return self._trading_days

    def _get_options_expiry_dates(self, year, month):
        """
        计算港股期权到期日（每月第四个周三）

        港股恒指期权到期日通常是当月倒数第二个营业日
        简化规则：当月第四个周三
        """
        # 找到当月第一个周三
        first_day = datetime(year, month, 1)
        # 0=Monday, 1=Tuesday, 2=Wednesday...
        first_wed_offset = (2 - first_day.weekday()) % 7
        first_wed = first_day + timedelta(days=first_wed_offset)

        # 第四个周三 = 第一个周三 + 21天
        fourth_wed = first_wed + timedelta(days=21)

        # 确保仍在当月
        if fourth_wed.month != month:
            fourth_wed = first_wed + timedelta(days=14)

        return fourth_wed

    def calculate_features(self, df):
        """
        计算日历效应特征

        参数:
        - df: 包含日期索引的 DataFrame（必须是 datetime 索引）

        返回:
        - DataFrame: 添加了日历特征的 DataFrame
        """
        print("  📅 计算日历效应特征...")

        # 确保索引是日期类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 移除时区信息（如果有的话）
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        dates = df.index

        # ========== 1. 星期效应 ==========
        # 周一=0, 周二=1, ..., 周五=4
        df['Day_of_Week'] = dates.dayofweek
        # 周一和周五效应（二元标志）
        df['Is_Monday'] = (dates.dayofweek == 0).astype(int)
        df['Is_Friday'] = (dates.dayofweek == 4).astype(int)

        # ========== 2. 月份效应 ==========
        df['Month_of_Year'] = dates.month
        # 1月效应标志
        df['Is_January'] = (dates.month == 1).astype(int)
        # 10月效应标志（历史统计港股10月表现较强）
        df['Is_October'] = (dates.month == 10).astype(int)
        # Q4窗口粉饰效应（10-12月）
        df['Is_Q4'] = (dates.month >= 10).astype(int)

        # ========== 3. 月初/月末效应 ==========
        # 判断月初（前3个交易日）和月末（后3个交易日）
        df['Day_of_Month'] = dates.day
        # 月初标志：当月前3个交易日
        df['Is_Month_Start'] = (df['Day_of_Month'] <= 5).astype(int)
        # 月末标志：当月后3个交易日
        df['Is_Month_End'] = (df['Day_of_Month'] >= 26).astype(int)

        # ========== 4. 季末窗口粉饰效应 ==========
        # 季末月份（3/6/9/12月）的最后几天
        is_quarter_month = dates.month.isin([3, 6, 9, 12])
        df['Is_Qtr_End'] = (is_quarter_month & (df['Day_of_Month'] >= 26)).astype(int)

        # ========== 5. 期权到期日效应 ==========
        # 计算距离当月期权到期日的天数
        trading_days = self._get_trading_days()
        expiry_dates = []
        for year in range(dates.min().year, dates.max().year + 2):
            for month in range(1, 13):
                expiry = self._get_options_expiry_dates(year, month)
                expiry_dates.append(expiry)

        expiry_series = pd.Series(expiry_dates)
        # 对每个交易日，找到最近的期权到期日
        days_to_expiry = []
        for date in dates:
            # 找到当前或之后的到期日
            future_expiries = expiry_series[expiry_series >= pd.Timestamp(date)]
            if len(future_expiries) > 0:
                nearest = future_expiries.iloc[0]
                days_to = (nearest - pd.Timestamp(date)).days
            else:
                days_to = 30  # 默认值
            days_to_expiry.append(days_to)

        df['Days_to_Options_Expiry'] = days_to_expiry
        # 期权到期周标志（到期日前3天内）
        df['Is_Options_Expiry_Week'] = (df['Days_to_Options_Expiry'] <= 3).astype(int)

        # ========== 6. 长假前后效应 ==========
        holidays = self._get_holidays()
        days_to_holiday = []
        for date in dates:
            # 找到最近的假期
            future_holidays = holidays[holidays >= pd.Timestamp(date)]
            if len(future_holidays) > 0:
                days_to = (future_holidays[0] - pd.Timestamp(date)).days
            else:
                days_to = 30
            days_to_holiday.append(days_to)

        df['Days_to_Holiday'] = days_to_holiday
        # 长假前3天标志（资金可能调仓）
        df['Is_Pre_Holiday'] = (df['Days_to_Holiday'] <= 3).astype(int)
        # 假期后首日标志
        # 找假期结束后最近交易日
        past_holidays = []
        for date in dates:
            # 找最近的过去假期
            recent_holidays = holidays[holidays <= pd.Timestamp(date)]
            if len(recent_holidays) > 0:
                days_since = (pd.Timestamp(date) - recent_holidays[-1]).days
            else:
                days_since = 30
            past_holidays.append(days_since)

        df['Is_Post_Holiday'] = (pd.Series(past_holidays, index=df.index) <= 1).astype(int)

        # ========== 7. 台风季效应 ==========
        # 港股台风季：6-10月，可能影响交易
        df['Is_Typhoon_Season'] = (dates.month.isin([6, 7, 8, 9, 10])).astype(int)

        # ========== 8. 当月剩余交易日 ==========
        # 估算当月剩余交易日（简单用工作日近似）
        remaining_days = []
        for date in dates:
            month_end = pd.Timestamp(date.year, date.month, 1) + pd.offsets.MonthEnd(0)
            remaining = np.busday_count(date.date(), month_end.date())
            remaining_days.append(max(0, remaining))

        df['Remaining_Trading_Days'] = remaining_days
        # 月末资金紧张标志（剩余<=3天）
        df['Is_Month_End_Rush'] = (df['Remaining_Trading_Days'] <= 3).astype(int)

        # ========== 9. 周期性特征（正弦/余弦编码）==========
        # 将月份转为连续的周期特征，避免离散编码的跳跃
        month_angle = 2 * np.pi * dates.month / 12
        df['Month_Sin'] = np.sin(month_angle)
        df['Month_Cos'] = np.cos(month_angle)

        # 星期几的周期编码
        dow_angle = 2 * np.pi * dates.dayofweek / 5
        df['DOW_Sin'] = np.sin(dow_angle)
        df['DOW_Cos'] = np.cos(dow_angle)

        # 清理临时列
        df.drop(columns=['Day_of_Month'], inplace=True, errors='ignore')

        feature_count = len([c for c in df.columns if c in self.get_feature_names()])
        print(f"  ✅ 日历效应特征计算完成（{feature_count} 个特征）")

        return df

    @staticmethod
    def get_feature_names():
        """返回所有日历效应特征名"""
        return [
            # 星期效应
            'Day_of_Week', 'Is_Monday', 'Is_Friday',
            # 月份效应
            'Month_of_Year', 'Is_January', 'Is_October', 'Is_Q4',
            # 月初/月末效应
            'Is_Month_Start', 'Is_Month_End',
            # 季末效应
            'Is_Qtr_End',
            # 期权到期效应
            'Days_to_Options_Expiry', 'Is_Options_Expiry_Week',
            # 假期效应
            'Days_to_Holiday', 'Is_Pre_Holiday', 'Is_Post_Holiday',
            # 台风季
            'Is_Typhoon_Season',
            # 剩余交易日
            'Remaining_Trading_Days', 'Is_Month_End_Rush',
            # 周期性编码
            'Month_Sin', 'Month_Cos', 'DOW_Sin', 'DOW_Cos',
        ]


# 日历效应特征配置（用于 FEATURE_CONFIG）
CALENDAR_FEATURE_CONFIG = {
    'calendar_features': CalendarFeatureCalculator.get_feature_names()
}
