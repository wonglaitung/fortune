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
