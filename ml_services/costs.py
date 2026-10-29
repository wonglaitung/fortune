# -*- coding: utf-8 -*-
"""交易成本口径（唯一来源，D97）。

- COST_ROUND_TRIP = 一次完整买入+卖出的总成本（相对成交额）
- 单边费率 = 往返 / 2

两种扣法：
1) 持仓替换（等权 TopK）：成本 = 单边换手 × 往返
2) 仓位变化（择时/权重）：成本 = 0.5 × Σ|Δw| × 往返
   （|Δw|=1 表示全买或全卖；买+卖合计 1.0，对应往返一次）
"""
COST_ROUND_TRIP = 0.005
COST_PER_SIDE = COST_ROUND_TRIP / 2


def cost_from_turnover(turnover_one_way, round_trip=COST_ROUND_TRIP):
    return float(turnover_one_way) * float(round_trip)


def cost_from_weight_change(abs_delta_sum, round_trip=COST_ROUND_TRIP):
    return 0.5 * float(abs_delta_sum) * float(round_trip)


def cost_label(round_trip=COST_ROUND_TRIP):
    return f"往返 {round_trip:.3%}（单边 {round_trip / 2:.3%}）"


from dataclasses import dataclass


@dataclass
class SideCostModel:
    """单边成本模型（D99）。

    per_side = 印花税 + 佣金率 + 交易所/监管费 + 半价差 + 冲击
    """
    broker_rate: float = 0.001      # 零售 0.10%
    min_fee: float = 50.0           # 每笔最低佣金（HK$）
    impact_k: float = 1.0
    stamp: float = 0.001            # 印花税（单边 0.1%）
    hkex: float = 0.0000565
    sfc: float = 0.000027
    afrc: float = 0.0000015
    ccass: float = 0.00002

    def per_side(self, trade_value, adv, sigma, spread, is_etf=False):
        tv = max(float(trade_value), 1.0)
        comm = max(self.broker_rate * tv, self.min_fee) / tv
        ccass = max(self.ccass * tv, 2.0) / tv
        levies = self.hkex + self.sfc + self.afrc + ccass
        stamp = 0.0 if is_etf else self.stamp
        half_spread = min(max(float(spread), 0.0), 0.2) / 2.0
        part = tv / max(float(adv), 1.0)
        impact = min(self.impact_k * float(sigma) * (part ** 0.5), 0.05) if sigma == sigma else 0.0
        return {
            'stamp': stamp, 'commission': comm, 'levies': levies,
            'half_spread': half_spread, 'impact': impact,
            'total': stamp + comm + levies + half_spread + impact,
        }


def hk_tick_size(price):
    """港交所股票最小报价单位（按价格档）。"""
    p = float(price)
    for upper, tick in ((0.25, 0.001), (0.5, 0.005), (10, 0.01), (20, 0.02),
                        (100, 0.05), (200, 0.1), (500, 0.2), (1000, 0.5),
                        (2000, 1.0), (5000, 2.0)):
        if p < upper:
            return tick
    return 5.0


def tick_spread(price, floor=0.0002, cap=0.02):
    """报价价差代理：max(tick/price, 2bp)，上限 2%。"""
    p = float(price)
    if not (p > 0):
        return floor
    return min(max(hk_tick_size(p) / p, floor), cap)
