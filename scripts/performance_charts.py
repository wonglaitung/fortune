#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测性能报告可视化图表生成器 (Performance Report Charts)

为港股「预测性能报告」邮件生成可视化图表，风格与 A股雷达图保持一致：
matplotlib 渲染 PNG → CID 内嵌进 HTML 邮件（send_email_with_images）。

图表集（D3 口径：方向技能 / 超额lift 为主，绝对准确率/胜率不作判定依据）：
  1. 三周期整体性能雷达（small multiples：1天/5天/20天 各一张，5 维度）
  2. 各周期时间窗口超额lift / 方向技能柱状图（small multiples，0 基准虚线）
  3. 板块性能雷达网格（每个板块一张，按综合分三色上色）
  4. 三周期模式平均收益水平柱状图（8 种模式，按平均收益排序，胜率仅参考标注）
  5. 个股综合分排名条形图 + Top N 雷达网格（综合分 = 方向技能×3·超额lift×3·其余×1）

设计说明（遵循 dataviz 规范）：
  - 周期为有序类别，用单色相顺序色带（浅蓝→深蓝 = 短周期→长周期），
    且采用 small multiples 形式，每个子图单系列，身份由标题而非颜色承担。
  - 三色系统（绿/橙/红）保留为「状态色」，只在带数值标签处使用，颜色不作为唯一编码。
  - 文本统一使用墨色（#333/#666/#999），不套用系列色。
  - 静态邮件图片，无交互层；明细表格由性能报告正文提供。
"""

import io
import re

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入即复用 A股雷达图的字体（WenQuanYi Micro Hei，带回退）与颜色常量
from scripts.stock_radar import (  # noqa: E402
    _get_color,
    COLOR_GREEN,
    COLOR_ORANGE,
    COLOR_RED,
    COLOR_GRID,
)

# ── 配置 ──
# 周期颜色：顺序蓝色色带，浅→深 = 1天→20天（明度单调，符合顺序编码）
HORIZON_COLORS = {1: '#7cb4f8', 5: '#3b82f6', 20: '#1d4ed8'}
HORIZON_NAMES = {1: '1天', 5: '5天', 20: '20天'}
HORIZONS = [1, 5, 20]

# 墨色（文本专用，不使用系列色）
INK = '#333333'
INK_MUTED = '#666666'
INK_FAINT = '#999999'

# 雷达 5 维度（均为 D3 口径：方向技能 / 超额 lift 为主，不含绝对准确率/胜率）
# 注：绝对准确率/胜率受行情主导（AGENTS D3 禁用其为评估/排名依据），故不作雷达轴；
#     样本量 n 是可信度上下文，也不作轴，仅在图下方文字标注。
PERF_DIMENSIONS = ['方向技能', '超额lift', '平均收益', '夏普比率', '买入平均收益']

# 综合分加权（D3 双主指标主导）：方向技能×3、超额lift×3，其余质量指标×1。
# 仅用于「综合分」这一标量（排名条 / Top10 选取 / 状态三色 / 雷达标题综合分）；
# 雷达 5 轴多边形本身不加权（极坐标对单轴加权会扭曲形状，5 维仍等权呈现）。
PERF_DIM_WEIGHTS = {
    '方向技能': 3.0,
    '超额lift': 3.0,
    '平均收益': 1.0,
    '夏普比率': 1.0,
    '买入平均收益': 1.0,
}


def composite_score(dimensions):
    """5 维度按 PERF_DIM_WEIGHTS 加权平均 → 综合分（0-100）。

    与等权均值不同：方向技能/超额lift 权重最高，使综合分/排名向"基准扣除后的
    真实能力"倾斜（D3 口径）。雷达多边形形状不受影响。
    """
    wsum = 0.0
    wtot = 0.0
    for k, w in PERF_DIM_WEIGHTS.items():
        wsum += w * dimensions.get(k, 0)
        wtot += w
    return wsum / wtot if wtot > 0 else 0.0

# 归一化常量（在图下方 caption 中向读者说明）
# 雷达 5 维全部采用**组内自适应尺度**：bound = max(下限, 组内最大|值|)。
# 0 始终=雷达中心 50（中性：无技能/零收益/零夏普），仅幅度缩放、语义不变；
# 使各板块/个股间微弱差异在雷达上可分辨，差异大时不截断。
# 下限防止全零/全同组除零或贴中心。
RETURN_FLOOR = 0.02      # 平均收益 / 买入平均收益 轴下限：±2%
SKILL_FLOOR = 0.02       # 方向技能 轴下限：±2pp
LIFT_FLOOR = 0.02        # 超额 lift 轴下限：±2pp
SHARPE_FLOOR = 0.20      # 单期夏普比率 轴下限：±0.2
# 注：calculate_metrics 的 sharpe 是单持有期信噪比(mean/std)，真实量级 ~±1；
#     不能年化(滚动样本高度重叠、违反 i.i.d.)，故量级参考 ±1，自适应缩放仍在组内展开。

DEFAULT_BOUNDS = {
    'direction_skill': SKILL_FLOOR,
    'lift': LIFT_FLOOR,
    'avg_return': RETURN_FLOOR,
    'buy_avg_return': RETURN_FLOOR,
    'sharpe_ratio': SHARPE_FLOOR,
}

# 章节标题样式（与 A股邮件一致）
_SECTION_H2 = ('<h2 style="color: #007bff; margin-top: 30px; '
               'border-bottom: 1px solid #ddd; padding-bottom: 5px;">{title}</h2>')
_CAPTION = ('<p style="color: #666; font-size: 11px; margin: 5px 0 12px 0;">{text}</p>')


# ════════════════════════════════════════════════════════════
# 归一化：指标 → 0-100 维度分
# ════════════════════════════════════════════════════════════

def _safe_float(val, default=0.0):
    """转 float；None/NaN/非法值返回 default，避免 NaN 污染图表数值。"""
    if val is None:
        return default
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    if np.isnan(v):
        return default
    return v


def _axis_bound(metrics_list, key, floor):
    """雷达自适应轴边界：max(下限, 组内最大|指标|)。用于全部 5 维。

    居中映射 0→50 恒成立，bound 只控制幅度缩放：组内差异小时用窄界放大，
    差异大时不截断；全零/缺数据回退 floor。
    """
    vals = [abs(_safe_float(m.get(key), 0.0)) for m in metrics_list if m]
    vals = [v for v in vals if v > 0]
    return max(floor, max(vals)) if vals else floor


def _group_bounds(metrics_list):
    """对 5 维分别计算组内自适应边界（_axis_bound），返回 {指标key: bound}。"""
    return {k: _axis_bound(metrics_list, k, floor) for k, floor in DEFAULT_BOUNDS.items()}


def _radar_ci_from_raw(ci_raw, bounds):
    """bootstrap CI（原始小数 {key:(lo,hi)}）→ 归一化 {维度名:(lo,hi)}，复用组内 bounds。

    仅方向技能/超额lift 两维有 bootstrap CI；其他键（如 n_blocks）忽略。
    """
    out = {}
    mapping = {
        'direction_skill': ('方向技能', normalize_direction_skill),
        'lift': ('超额lift', normalize_lift),
    }
    for k, v in (ci_raw or {}).items():
        if k in mapping:
            dim, fn = mapping[k]
            lo, hi = v
            b = bounds.get(k)
            out[dim] = (fn(lo, b), fn(hi, b))
    return out


def normalize_direction_skill(skill, bound=None):
    """方向技能 (pp 小数) → 0-100 居中映射，50 = 与"永远看涨"无差异。
    bound 为空时用 SKILL_FLOOR 下限；雷达调用方应传组内自适应界（_group_bounds）。"""
    b = bound if bound else SKILL_FLOOR
    s = max(-b, min(b, _safe_float(skill, 0.0)))
    return (s + b) / (2 * b) * 100


def normalize_lift(lift, bound=None):
    """超额 lift (pp 小数) → 0-100 居中映射，50 = 与无条件买入基准无差异。
    bound 为空时用 LIFT_FLOOR 下限；雷达调用方应传组内自适应界（_group_bounds）。"""
    b = bound if bound else LIFT_FLOOR
    s = max(-b, min(b, _safe_float(lift, 0.0)))
    return (s + b) / (2 * b) * 100


def normalize_return(avg_return, bound=None):
    """平均收益：居中映射 [0, 100]，50 = 零收益（NaN 记中性 50）。
    bound 为空时用 RETURN_FLOOR 下限；雷达调用方应传组内自适应界。"""
    b = bound if bound else RETURN_FLOOR
    r = max(-b, min(b, _safe_float(avg_return, 0.0)))
    return (r + b) / (2 * b) * 100


def normalize_sharpe(sharpe, bound=None):
    """单期夏普比率：居中映射 [0, 100]，50 = 零夏普（NaN 记中性 50）。
    bound 为空时用 SHARPE_FLOOR 下限；雷达调用方应传组内自适应界。"""
    b = bound if bound else SHARPE_FLOOR
    s = max(-b, min(b, _safe_float(sharpe, 0.0)))
    return (s + b) / (2 * b) * 100


def metrics_to_dimensions(metrics, bounds=None):
    """
    将 calculate_metrics() 的指标字典转换为雷达 5 维度分（0-100）。

    D3 口径：方向技能 / 超额 lift 为主维度（基准扣除后），不含绝对准确率/胜率。
    参数:
    - metrics: performance_monitor.calculate_metrics() 的返回值
    - bounds: {指标key: 组内自适应边界}（_group_bounds 结果）；缺省时该维回退 FLOOR 下限。

    返回:
    - {维度名: 分数}（仅含 D3/质量维度；样本量 n 不作轴）
    """
    if not metrics:
        metrics = {}
    bounds = bounds or {}
    return {
        '方向技能': round(normalize_direction_skill(
            metrics.get('direction_skill'), bounds.get('direction_skill')), 1),
        '超额lift': round(normalize_lift(
            metrics.get('lift'), bounds.get('lift')), 1),
        '平均收益': round(normalize_return(
            metrics.get('avg_return'), bounds.get('avg_return')), 1),
        '夏普比率': round(normalize_sharpe(
            metrics.get('sharpe_ratio'), bounds.get('sharpe_ratio')), 1),
        # 买入平均收益复用收益居中映射：衡量"喊涨时平均赚多少"，纯质量、无市场涨跌干扰
        '买入平均收益': round(normalize_return(
            metrics.get('buy_avg_return'), bounds.get('buy_avg_return')), 1),
    }


def _strip_unsafe_glyphs(text):
    """去除 matplotlib 中文字体可能缺失的字符（如 emoji ⭐），保留中英文/数字/括号。"""
    return re.sub(r'[^\w一-鿿（）()·\- ]', '', str(text)).strip()


# ════════════════════════════════════════════════════════════
# 基础渲染：单系列雷达图 PNG
# ════════════════════════════════════════════════════════════

def _render_single_radar_png_bytes(title, dimensions, color, size=220, composite=None,
                                   ci=None):
    """
    渲染单系列多边形雷达图，返回 PNG bytes（轴数 = PERF_DIMENSIONS 维度数）。

    参数:
    - title: 图表标题（如 "1天" / "银行股"）
    - dimensions: {维度名: 0-100 分数}
    - color: 填充/描边颜色
    - size: 图片尺寸（像素，换算为 figsize）
    - composite: 标题"综合 X"用的标量；None 时回退 5 维等权均值（兼容旧调用）。
      调用方应传加权综合分 composite_score(dimensions)，使标题与排名口径一致。
    - ci: {维度名: (lo, hi)} 归一化后（0-100）的 95%CI；在对应维度轴画径向误差须，
      提示不确定性（避免把形状当能力，D3）。仅方向技能/超额lift 两维有 bootstrap CI。
    """
    categories = PERF_DIMENSIONS
    n = len(categories)

    values = [dimensions.get(c, 0) for c in categories]
    avg = composite if composite is not None else float(np.mean(values))
    values_closed = values + values[:1]
    angles = [i / float(n) * 2 * np.pi for i in range(n)]
    angles_closed = angles + angles[:1]

    figsize = size / 100.0
    fig, ax = plt.subplots(
        figsize=(figsize, figsize),
        subplot_kw=dict(polar=True),
        facecolor='white',
    )

    ax.set_ylim(0, 100)

    # 背景网格圈（recessive）
    for gv in [20, 40, 60, 80]:
        ax.plot(angles_closed, [gv] * (n + 1), color=COLOR_GRID,
                linewidth=0.5, alpha=0.30, zorder=1)

    # 数据多边形（细描边 + 浅色填充 + 白边标记点 = relief）
    ax.fill(angles_closed, values_closed, alpha=0.20, color=color, zorder=3)
    ax.plot(angles_closed, values_closed, 'o-', linewidth=1.8, color=color,
            markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.6,
            markersize=4, zorder=4)

    # CI 径向误差须（墨色；两端 cap；clip 到 [0,100] 防越界变形）
    if ci:
        for i, cat in enumerate(categories):
            if cat not in ci:
                continue
            lo, hi = ci[cat]
            lo = max(0.0, min(100.0, lo))
            hi = max(0.0, min(100.0, hi))
            a = angles[i]
            ax.plot([a, a], [lo, hi], color=INK_MUTED, linewidth=1.4, zorder=5)
            for v in (lo, hi):
                ax.plot([a - 0.05, a + 0.05], [v, v], color=INK_MUTED,
                        linewidth=1.4, zorder=5)

    # 轴标签（墨色）
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=8, color=INK)
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], fontsize=5.5, color=INK_FAINT)
    ax.set_rlabel_position(0)

    # 标题（含综合分，墨色）
    ax.set_title(f'{title}（综合 {avg:.0f}）', fontsize=9.5, pad=14, color=INK)

    ax.grid(True, alpha=0.25, color=COLOR_GRID)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def _save_png_bytes(fig):
    """通用：把 figure 渲染为 PNG bytes。"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.12)
    plt.close(fig)
    return buf.getvalue()


def _style_bar_axis(ax):
    """统一柱状图坐标轴样式：去除上/右边框、浅色网格、墨色刻度。"""
    ax.grid(axis='both', color=COLOR_GRID, alpha=0.30, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#dddddd')
    ax.tick_params(axis='both', labelsize=8, colors=INK_MUTED)


# ════════════════════════════════════════════════════════════
# 章节 1：三周期整体性能雷达（small multiples）
# ════════════════════════════════════════════════════════════

def generate_overall_radar_section(horizon_metrics, window_name='3个月',
                                   ci=None, guardrail_html=None):
    """
    生成「模型整体性能雷达」HTML 区块 + CID 附件。

    参数:
    - horizon_metrics: {周期: calculate_metrics() 结果}（建议传入 3个月窗口）
    - window_name: 统计窗口名称（用于说明文字）
    - ci: {周期: {'direction_skill':(lo,hi), 'lift':(lo,hi)}} 原始小数 bootstrap CI，
      绘制方向技能/超额lift 两维的径向误差须（不确定性可视化，D3 防误判）
    - guardrail_html: 组合层护栏摘要 HTML 片段（20d 净IR/PBO/DSR，D2），追加到节末

    返回: (html, {cid: png_bytes})
    """
    cells = []
    attachments = {}
    ci = ci or {}

    # 雷达 5 维按组内自适应尺度（0 恒为中性，仅缩放幅度，提升横向区分度）
    _m_list = [horizon_metrics.get(h) or {} for h in HORIZONS]
    _bounds = _group_bounds(_m_list)

    for h in HORIZONS:
        m = horizon_metrics.get(h) or {}
        if m.get('total_predictions', 0) == 0:
            continue
        dims = metrics_to_dimensions(m, _bounds)
        color = HORIZON_COLORS[h]
        cid = f'perf_radar_{h}d'
        radar_ci = _radar_ci_from_raw(ci.get(h), _bounds)
        attachments[cid] = _render_single_radar_png_bytes(
            f'{HORIZON_NAMES[h]}周期', dims, color, size=230,
            composite=composite_score(dims), ci=radar_ci)

        # 关键指标（D3 口径：方向技能 / 超额lift 主色标注，不含绝对准确率/胜率）
        ds = m.get('direction_skill', 0)
        lift = m.get('lift', 0)
        ds_color = COLOR_GREEN if ds >= 0.02 else COLOR_ORANGE if ds >= 0 else COLOR_RED
        lift_color = COLOR_GREEN if lift >= 0.02 else COLOR_ORANGE if lift >= 0 else COLOR_RED
        avg_ret = m.get('avg_return', 0)
        ret_color = COLOR_GREEN if avg_ret >= 0 else COLOR_RED
        buy_ret = _safe_float(m.get('buy_avg_return'), 0.0)
        buy_ret_color = COLOR_GREEN if buy_ret >= 0 else COLOR_RED
        cells.append(f"""            <td style="border: none; text-align: center; padding: 6px; vertical-align: top; width: 33%;">
                <div style="background: #fafafa; border-radius: 8px; padding: 8px; margin: 2px;">
                    <img src="cid:{cid}" style="width: 100%; max-width: 230px; height: auto;" alt="{HORIZON_NAMES[h]}周期性能雷达">
                    <div style="font-size: 11px; color: #666; margin-top: 4px; line-height: 1.7;">
                        样本 <b style="color:#333;">{m.get('total_predictions', 0)}</b><br>
                        方向技能 <b style="color: {ds_color};">{ds:+.1%}</b><br>
                        超额lift <b style="color: {lift_color};">{lift:+.1%}</b><br>
                        平均收益 <b style="color: {ret_color};">{avg_ret:+.2%}</b><br>
                        买入均收 <b style="color: {buy_ret_color};">{buy_ret:+.2%}</b>
                        · 夏普 <b style="color:#333;">{m.get('sharpe_ratio', 0):.2f}</b>
                    </div>
                </div>
            </td>
""")

    if not cells:
        return '', {}

    html = _SECTION_H2.format(title='一、模型整体性能雷达')
    html += _CAPTION.format(
        text=f'统计窗口：{window_name} | 5 维度均为 D3 口径指标，归一化至 0–100：'
             '方向技能=准确率−永远看涨占比（50=无技能）· '
             '超额lift=信号净胜率−无条件买入基准（50=无超额）· '
             '平均收益（50=零收益）· 夏普（50=零夏普）· 买入平均收益（50=零收益）| '
             '5 维均按三周期组内自适应尺度缩放（0 恒为雷达中心，仅调幅度，便于横向对比）| '
             '方向技能/超额lift 两轴的径向短线 = block bootstrap 95%CI（越宽越不可靠，'
             'CI 跨 0 视为不显著——形状大≠有能力）| '
             '样本量 n 见各图下方文字（仅作可信度参考，不参与雷达形状）| '
             '标题"综合"=加权综合分(方向技能×3·超额lift×3·其余×1，D3 双主指标为主，雷达5轴形状仍等权) | '
             '评估一律以基准扣除后的技能/超额为准（AGENTS D3），绝对准确率/胜率不作判定依据 | '
             '颜色深浅区分周期（浅=1天 → 深=20天）')
    html += '    <table style="border: 0; border-collapse: collapse; width: 100%;">\n        <tr>\n'
    html += ''.join(cells)
    html += '        </tr>\n    </table>\n'
    if guardrail_html:
        html += '<div style="margin-top: 12px;">' + guardrail_html + '</div>\n'
    return html, attachments


# ════════════════════════════════════════════════════════════
# 章节 2：各周期时间窗口超额lift / 方向技能柱状图（small multiples）
# ════════════════════════════════════════════════════════════

def generate_window_bar_section(window_metrics,
                                time_windows=((30, '1个月'), (90, '3个月'), (180, '6个月')),
                                ci=None, guardrail_html=None):
    """
    生成「各周期时间窗口表现」HTML 区块 + CID 附件：
    上下两张图，结构一致——上图为超额lift、下图为方向技能（D3 双主指标，0 基准虚线），
    每张图内 3 个子图（1天/5天/20天），每个子图展示 1个月/3个月/6个月。

    参数:
    - window_metrics: {窗口天数: {周期: metrics}}
    - time_windows: [(天数, 名称), ...]
    - ci: {(天数, 周期): {'lift_ci': (lo,hi), 'ds_ci': (lo,hi)}} block bootstrap 95%CI，
      有则绘制误差须（不确定性可视化，D3）；样本不足的格子不画须。
    - guardrail_html: 组合层护栏摘要 HTML 片段（20d 净IR/PBO/DSR，D2），追加到本节末尾。

    返回: (html, {cid: png_bytes})
    """
    windows = list(time_windows)
    x = np.arange(len(windows))
    xlabels = [name for _, name in windows]
    ci = ci or {}

    # (metrics键, 图标题, 纵轴名) —— D3 双主指标：超额lift + 方向技能（pp，可正可负）
    panels = [
        ('perf_window_lift', 'lift', '各周期在不同时间窗口的超额 lift', '超额lift (pp)'),
        ('perf_window_ds', 'direction_skill', '各周期在不同时间窗口的方向技能', '方向技能 (pp)'),
    ]

    attachments = {}
    imgs = []
    for cid, key, suptitle, ylabel in panels:
        fig, axes = plt.subplots(1, len(HORIZONS), figsize=(9.6, 3.1),
                                 facecolor='white', sharey=True)
        if len(HORIZONS) == 1:
            axes = [axes]

        any_data = False
        ci_key = 'lift_ci' if key == 'lift' else 'ds_ci'
        for ax, h in zip(axes, HORIZONS):
            vals, counts = [], []
            err_lo, err_hi = [], []
            for days, _ in windows:
                m = (window_metrics.get(days, {}) or {}).get(h, {}) or {}
                vals.append(float(m.get(key, 0)) * 100)
                counts.append(int(m.get('total_predictions', 0)))
                c = ci.get((days, h))
                if c and c.get(ci_key):
                    lo, hi = c[ci_key]
                    err_lo.append(max(0.0, vals[-1] - lo * 100))
                    err_hi.append(max(0.0, hi * 100 - vals[-1]))
                else:
                    err_lo.append(0.0)
                    err_hi.append(0.0)
            if any(c > 0 for c in counts):
                any_data = True

            color = HORIZON_COLORS[h]
            ax.bar(x, vals, color=color, width=0.62,
                   edgecolor='white', linewidth=0.8, zorder=3)
            # block bootstrap 95%CI 误差须（点估计不足信，D3 要求给区间）
            if any(e > 0 for e in err_hi + err_lo):
                ax.errorbar(x, vals, yerr=[err_lo, err_hi],
                            fmt='none', ecolor=INK_MUTED, elinewidth=1.2,
                            capsize=3, zorder=6)
            ax.axhline(0, color=INK_FAINT, linestyle='--', linewidth=0.9,
                       alpha=0.8, zorder=2)  # 0 基准：无技能 / 无超额（基准扣除后）
            ax.set_title(f'{HORIZON_NAMES[h]}周期', fontsize=10.5, color=INK, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=8.5, color=INK_MUTED)
            _style_bar_axis(ax)

            # 数值直接标注（墨色；无样本标灰）
            for xi, v, c in zip(x, vals, counts):
                if c > 0:
                    ax.text(xi, v + (0.5 if v >= 0 else -1.8), f'{v:+.1f}pp',
                            ha='center', va='bottom' if v >= 0 else 'top',
                            fontsize=8.5, color=INK, zorder=5)
                else:
                    ax.text(xi, 0, '无样本', ha='center', va='center',
                            fontsize=7.5, color=INK_FAINT, zorder=5)

        # 对称 y 轴（容纳负值 + CI 误差须，0 为中性）
        if any_data:
            # 汇总各柱点值 + CI 上下界，取最大幅度
            ext = 0.0
            for h in HORIZONS:
                for days, _ in windows:
                    m = (window_metrics.get(days, {}) or {}).get(h, {}) or {}
                    v = float(m.get(key, 0)) * 100
                    ext = max(ext, abs(v))
                    c = ci.get((days, h))
                    if c and c.get(ci_key):
                        lo, hi = c[ci_key]
                        ext = max(ext, abs(lo * 100), abs(hi * 100))
            bound = max(5.0, ext * 1.15)
            for ax in axes:
                ax.set_ylim(-bound, bound)
                ax.set_yticks([-int(bound), 0, int(bound)])

        axes[0].set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
        fig.suptitle(suptitle, fontsize=12, color=INK, y=1.04)
        fig.tight_layout()

        if not any_data:
            plt.close(fig)
            continue
        attachments[cid] = _save_png_bytes(fig)
        imgs.append((cid, suptitle))

    if not imgs:
        return '', {}

    html = _SECTION_H2.format(title='二、各周期时间窗口表现')
    html += _CAPTION.format(
        text='上图=超额lift（信号净胜率 − 无条件买入基准）| 下图=方向技能（准确率 − 永远看涨占比）| '
             '两者均为基准扣除后的百分点，虚线 0 = 无超额 / 无技能 | '
             '误差须 = block bootstrap 95%CI（按交易日分块，CI 跨 0 视为不显著）| '
             '绝对准确率/胜率不作判定依据（AGENTS D3）| '
             '颜色深浅区分周期（浅=1天 → 深=20天）')
    for cid, suptitle in imgs:
        html += ('    <div style="text-align: center; margin: 6px 0;">'
                 f'<img src="cid:{cid}" style="max-width: 680px; width: 100%; height: auto;" '
                 f'alt="{suptitle}"></div>\n')
    if guardrail_html:
        html += '<div style="margin-top: 12px;">' + guardrail_html + '</div>\n'
    return html, attachments


# ════════════════════════════════════════════════════════════
# 章节 3：板块性能雷达网格
# ════════════════════════════════════════════════════════════

def generate_sector_radar_section(sector_metrics, min_samples=5, items_per_row=4,
                                  ci=None, guardrail_html=None):
    """
    生成「板块性能雷达」网格 HTML 区块 + CID 附件（仿 A股个股雷达网格）。

    参数:
    - sector_metrics: {板块代码: {'name': 中文名, 'metrics': calculate_metrics() 结果}}
    - min_samples: 最小样本数，低于此值的板块跳过（并打印日志，不做静默截断）
    - items_per_row: 每行几张
    - ci: {板块代码: {'direction_skill':(lo,hi), 'lift':(lo,hi)}} 原始小数 bootstrap CI，
      绘制方向技能/超额lift 两维径向误差须（D3 防误判）
    - guardrail_html: 组合层护栏摘要 HTML 片段（20d 净IR/PBO/DSR，D2），追加到节末

    返回: (html, {cid: png_bytes})
    """
    items = []
    attachments = {}
    dropped = []
    rendered_metrics = []
    ci = ci or {}

    for sector, info in sector_metrics.items():
        m = info.get('metrics') or {}
        total = int(m.get('total_predictions', 0))
        name = info.get('name', sector)
        if total < min_samples:
            dropped.append((name, total))
            continue
        rendered_metrics.append(m)

    # 雷达 5 维按组内（已渲染板块）自适应尺度，提升横向区分度
    _bounds = _group_bounds(rendered_metrics)

    for sector, info in sector_metrics.items():
        m = info.get('metrics') or {}
        total = int(m.get('total_predictions', 0))
        name = info.get('name', sector)
        if total < min_samples:
            continue

        dims = metrics_to_dimensions(m, _bounds)
        avg = composite_score(dims)  # 加权综合分（方向技能/超额lift 为主，D3）
        color = _get_color(avg)  # 状态三色：≥60 绿 / 40-60 橙 / <40 红
        cid = f'perf_sector_{sector}'
        radar_ci = _radar_ci_from_raw(ci.get(sector), _bounds)
        try:
            attachments[cid] = _render_single_radar_png_bytes(
                name, dims, color, size=190, composite=avg, ci=radar_ci)
        except Exception as e:  # 单板块失败不影响整体
            print(f'  [perf-radar] 板块图表生成失败: {name} {e}')
            continue

        items.append({
            'name': name,
            'avg': avg,
            'total': total,
            'direction_skill': m.get('direction_skill', 0),
            'lift': m.get('lift', 0),
            'avg_return': m.get('avg_return', 0),
            'buy_avg_return': m.get('buy_avg_return', 0),
            'ds_ci': (ci.get(sector) or {}).get('direction_skill'),
            'lift_ci': (ci.get(sector) or {}).get('lift'),
            'cid': cid,
        })

    # 被过滤板块明确记录（不做静默截断）
    for name, total in dropped:
        print(f'  [perf-radar] 板块样本不足跳过: {name} (n={total} < {min_samples})')

    if not items:
        return '', {}

    items.sort(key=lambda it: it['avg'], reverse=True)

    html = _SECTION_H2.format(title='三、板块表现雷达图')
    html += _CAPTION.format(
        text=f'统计口径：20天周期 / 3个月窗口 | 仅展示样本数 ≥ {min_samples} 的板块 | '
             '5 维度同整体雷达（方向技能 / 超额lift 为主，D3 口径）| 5 维均按'
             '板块组内自适应尺度缩放（0=中性，仅调幅度）| 方向技能/超额lift 的径向短线'
             ' = block bootstrap 95%CI，跨 0 标"不显著"（形状大≠有能力）| 综合分=加权'
             '(方向技能×3·超额lift×3·其余×1) | 颜色为综合分状态：'
             '<span style="color:#16a34a;">≥60</span> / '
             '<span style="color:#ea580c;">40–60</span> / '
             '<span style="color:#dc2626;">&lt;40</span>')
    html += '    <table style="border: 0; border-collapse: collapse; width: 100%;">\n        <tr>\n'

    for i, it in enumerate(items):
        if i % items_per_row == 0 and i > 0:
            html += '        </tr><tr>\n'
        avg_color = (COLOR_GREEN if it['avg'] >= 60
                     else COLOR_ORANGE if it['avg'] >= 40 else COLOR_RED)
        ds = _safe_float(it.get('direction_skill'), 0.0)
        ds_color = COLOR_GREEN if ds >= 0.02 else COLOR_ORANGE if ds >= 0 else COLOR_RED
        lift = _safe_float(it.get('lift'), 0.0)
        lift_color = COLOR_GREEN if lift >= 0.02 else COLOR_ORANGE if lift >= 0 else COLOR_RED
        ret_color = COLOR_GREEN if it['avg_return'] >= 0 else COLOR_RED
        buy_ret = _safe_float(it.get('buy_avg_return'), 0.0)
        buy_ret_color = COLOR_GREEN if buy_ret >= 0 else COLOR_RED

        def _sig(lo_hi):
            if not lo_hi:
                return ''
            lo, hi = lo_hi
            return '✅ 显著' if lo > 0 else ('❌ 显著为负' if hi < 0 else '⚠️ 不显著')

        ds_sig = _sig(it.get('ds_ci'))
        lift_sig = _sig(it.get('lift_ci'))
        html += f"""            <td style="border: none; text-align: center; padding: 5px; vertical-align: top; width: {100.0 / items_per_row:.0f}%;">
                <div style="background: #fafafa; border-radius: 6px; padding: 5px; margin: 2px;">
                    <img src="cid:{it['cid']}" style="width: 100%; max-width: 190px; height: auto;" alt="{it['name']}">
                    <div style="font-size: 10px; color: #666; margin-top: 2px; line-height: 1.6;">
                        综合 <b style="color: {avg_color};">{it['avg']:.0f}</b>
                        | 方向技能 <b style="color: {ds_color};">{ds:+.1%}</b>{(' <span style="color:#b45309;">' + ds_sig + '</span>') if ds_sig else ''}<br>
                        超额lift <b style="color: {lift_color};">{lift:+.1%}</b>{(' <span style="color:#b45309;">' + lift_sig + '</span>') if lift_sig else ''}
                        | 收益 <b style="color: {ret_color};">{it['avg_return']:+.1%}</b>
                        | 买入均收 <b style="color: {buy_ret_color};">{buy_ret:+.1%}</b>
                        | n={it['total']}
                    </div>
                </div>
            </td>
"""

    # 补齐末行空单元格
    remainder = len(items) % items_per_row
    if remainder > 0:
        for _ in range(items_per_row - remainder):
            html += f'            <td style="border: none; width: {100.0 / items_per_row:.0f}%;"></td>\n'

    html += '        </tr>\n    </table>\n'
    if guardrail_html:
        html += '<div style="margin-top: 12px;">' + guardrail_html + '</div>\n'
    return html, attachments


# ════════════════════════════════════════════════════════════
# 章节 4：三周期模式平均收益水平柱状图（对齐 md 报告的 D3 排序口径）
# ════════════════════════════════════════════════════════════

def generate_pattern_bar_section(pattern_stats, pattern_names):
    """
    生成「三周期模式平均收益」水平柱状图 HTML 区块 + CID 附件。

    D3 口径：模式排名/主轴用平均收益（与 md 报告一致），胜率仅作参考标注。
    参数:
    - pattern_stats: {模式: {'total', 'correct', 'win_rate', 'avg_return'}}
    - pattern_names: {模式: 中文名}

    返回: (html, {cid: png_bytes})
    """
    if not pattern_stats:
        return '', {}

    # 按平均收益升序排列 → barh 后最优模式在顶部（与 md 报告排序一致，D3 禁胜率排名）
    ordered = sorted(pattern_stats.items(),
                     key=lambda kv: _safe_float(kv[1].get('avg_return'), 0.0))
    labels, avg_returns, totals, win_rates = [], [], [], []
    for pattern, stats in ordered:
        name = _strip_unsafe_glyphs(pattern_names.get(pattern, ''))
        labels.append(f'{pattern} {name}'.strip())
        avg_returns.append(_safe_float(stats.get('avg_return'), 0.0) * 100)
        totals.append(int(stats.get('total', 0)))
        win_rates.append(_safe_float(stats.get('win_rate'), 0.0) * 100)
    # 状态三色（带数值标签，颜色非唯一编码）：按平均收益正负/显著
    colors = [COLOR_GREEN if ar >= 1.0
              else COLOR_ORANGE if ar >= 0 else COLOR_RED
              for ar in avg_returns]

    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.0, max(2.8, 0.5 * len(ordered) + 1.4)),
                           facecolor='white')
    ax.barh(y, avg_returns, color=colors, height=0.62,
            edgecolor='white', linewidth=0.8, zorder=3)
    ax.axvline(0, color=INK_FAINT, linestyle='--', linewidth=0.9,
               alpha=0.8, zorder=2)  # 0 基准：零收益
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    bound = max(5.0, max(abs(ar) for ar in avg_returns) * 1.2)
    ax.set_xlim(-bound, bound)
    ax.set_xticks([-int(bound), 0, int(bound)])
    ax.set_xlabel('平均收益 (%)', fontsize=9.5, color=INK_MUTED)
    ax.set_title('三周期模式平均收益（3个月窗口）', fontsize=12, color=INK, pad=10)
    _style_bar_axis(ax)

    # 数值 + 胜率(参考) + 样本量直接标注（墨色）
    for yi, ar, wr, t in zip(y, avg_returns, win_rates, totals):
        ax.text(ar + (0.3 if ar >= 0 else -0.3), yi,
                f'{ar:+.1f}% (胜率{wr:.0f}% n={t})',
                va='center', ha='left' if ar >= 0 else 'right',
                fontsize=8.8, color=INK, zorder=5)

    fig.tight_layout()

    cid = 'perf_pattern_bar'
    attachments = {cid: _save_png_bytes(fig)}

    html = _SECTION_H2.format(title='四、三周期模式验证')
    html += _CAPTION.format(
        text='模式编码：110 = 1天涨·5天涨·20天跌 | 主轴 = 该模式 20天平均收益（D3 口径，'
             '与报告排序一致），括号内胜率为参考、不作判定依据 | 虚线 0 = 零收益 | '
             '⚠️ 本表不构成交易依据：生产历史为重叠窗口（未 embargo），见报告正文 | '
             '颜色：'
             '<span style="color:#16a34a;">收益≥+1%</span> / '
             '<span style="color:#ea580c;">0~+1%</span> / '
             '<span style="color:#dc2626;">&lt;0</span>')
    html += ('    <div style="text-align: center;">'
             f'<img src="cid:{cid}" style="max-width: 620px; width: 100%; height: auto;" '
             'alt="三周期模式胜率柱状图"></div>\n')
    return html, attachments


# ════════════════════════════════════════════════════════════
# 章节 5：个股表现（全部排名条形图 + Top N 雷达网格）
# ════════════════════════════════════════════════════════════

def _sanitize_cid_code(code):
    """把股票代码转成合法 CID 片段（只保留字母数字，如 0700.HK → 0700HK）。"""
    return re.sub(r'[^A-Za-z0-9]', '', str(code))


def _stock_rank_bar_png_bytes(items):
    """
    渲染「全部个股综合分排名」水平条形图，返回 PNG bytes。

    参数:
    - items: [{'label', 'avg', 'total'}]，调用方已按综合分升序排列（最高分在顶部）
    """
    n = len(items)
    labels = [it['label'] for it in items]
    avgs = [it['avg'] for it in items]
    totals = [it['total'] for it in items]
    colors = [_get_color(a) for a in avgs]
    group_mean = float(np.mean(avgs)) if avgs else 0.0

    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * n + 1.5), facecolor='white')
    ax.barh(y, avgs, color=colors, height=0.66,
            edgecolor='white', linewidth=0.8, zorder=3)
    # 全组均值参考线（综合分非百分比，无 50 基准语义）
    ax.axvline(group_mean, color=INK_FAINT, linestyle='--', linewidth=0.9,
               alpha=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.8, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel('综合分（加权：方向技能/超额lift 为主）', fontsize=9.5, color=INK_MUTED)
    ax.set_title('个股综合分排名（20天 · 3个月窗口）', fontsize=12, color=INK, pad=10)
    _style_bar_axis(ax)

    # 数值 + 样本量直接标注（墨色）
    for yi, a, t in zip(y, avgs, totals):
        ax.text(a + 1.2, yi, f'{a:.0f}  (n={t})', va='center', ha='left',
                fontsize=8.0, color=INK, zorder=5)
    # 均值参考线标注
    ax.text(group_mean, n - 0.4, f'均值 {group_mean:.0f}', va='bottom', ha='center',
            fontsize=7.8, color=INK_FAINT, zorder=5)

    fig.tight_layout()
    return _save_png_bytes(fig)


def generate_stock_section(stock_bundle, min_samples=5, top_n=10, items_per_row=5,
                           ci=None, guardrail_html=None):
    """
    生成「个股表现」HTML 区块 + CID 附件：
      - 一张水平条形图：全部个股按综合分排名（紧凑、便于横向对比）
      - Top N 个股雷达网格：综合分最高者单独展示 5 维度细节

    参数:
    - stock_bundle: {代码: {'name', 'code', 'sector', 'metrics'}}
    - min_samples: 最小样本数，低于此值的个股跳过（并打印日志，不静默截断）
    - top_n: 雷达网格展示的个股数量
    - items_per_row: 雷达网格每行几张
    - ci: {代码: {'direction_skill':(lo,hi), 'lift':(lo,hi)}} 原始小数 bootstrap CI，
      绘制方向技能/超额lift 两维径向误差须（D3 防误判）
    - guardrail_html: 组合层护栏摘要 HTML 片段（20d 净IR/PBO/DSR，D2），追加到节末

    返回: (html, {cid: png_bytes})
    """
    items = []
    dropped = []
    rendered_metrics = []
    ci = ci or {}

    for code, info in stock_bundle.items():
        m = info.get('metrics') or {}
        total = int(m.get('total_predictions', 0))
        name = info.get('name', code)
        if total < min_samples:
            dropped.append((name, code, total))
            continue
        rendered_metrics.append(m)

    # 雷达 5 维按组内（已渲染个股）自适应尺度，提升横向区分度
    _bounds = _group_bounds(rendered_metrics)

    for code, info in stock_bundle.items():
        m = info.get('metrics') or {}
        total = int(m.get('total_predictions', 0))
        name = info.get('name', code)
        if total < min_samples:
            continue

        dims = metrics_to_dimensions(m, _bounds)
        avg = composite_score(dims)  # 加权综合分（方向技能/超额lift 为主，D3）
        items.append({
            'code': code,
            'name': name,
            'label': f'{_strip_unsafe_glyphs(name)} {code}',
            'avg': avg,
            'total': total,
            'dims': dims,
            'metrics': m,
            'ds_ci': (ci.get(code) or {}).get('direction_skill'),
            'lift_ci': (ci.get(code) or {}).get('lift'),
        })

    for name, code, total in dropped:
        print(f'  [perf-stock] 个股样本不足跳过: {name} {code} (n={total} < {min_samples})')

    if not items:
        return '', {}

    attachments = {}

    # ── 5.1 全部个股综合分排名条形图 ──
    ranked_asc = sorted(items, key=lambda it: it['avg'])  # barh 最高分落在顶部
    attachments['perf_stock_rank'] = _stock_rank_bar_png_bytes(ranked_asc)

    html = _SECTION_H2.format(title='五、个股表现')
    html += _CAPTION.format(
        text=f'统计口径：20天周期 / 3个月窗口 | 排名条覆盖全部 {len(items)} 只个股（样本 ≥ {min_samples}）| '
             f'下方雷达为综合分 Top {min(top_n, len(items))} 的细节 | '
             '5 维度同整体雷达（方向技能 / 超额lift 为主，D3 口径），5 维均按'
             '个股组内自适应尺度缩放（0=中性，仅调幅度）| 方向技能/超额lift 的径向短线'
             ' = block bootstrap 95%CI，跨 0 标"不显著"（形状大≠有能力）| 综合分=加权'
             '(方向技能×3·超额lift×3·其余×1) | 状态三色：'
             '<span style="color:#16a34a;">≥60</span> / '
             '<span style="color:#ea580c;">40–60</span> / '
             '<span style="color:#dc2626;">&lt;40</span>')
    html += ('    <div style="text-align: center;">'
             '<img src="cid:perf_stock_rank" style="max-width: 660px; width: 100%; height: auto;" '
             'alt="个股综合分排名"></div>\n')

    # ── 5.2 Top N 个股雷达网格 ──
    top_items = sorted(items, key=lambda it: it['avg'], reverse=True)[:top_n]
    html += '    <table style="border: 0; border-collapse: collapse; width: 100%;">\n        <tr>\n'

    for i, it in enumerate(top_items):
        if i % items_per_row == 0 and i > 0:
            html += '        </tr><tr>\n'
        cid = f"perf_stock_{_sanitize_cid_code(it['code'])}"
        radar_ci = _radar_ci_from_raw(
            {'direction_skill': it['ds_ci'], 'lift': it['lift_ci']}
            if (it['ds_ci'] or it['lift_ci']) else {}, _bounds)
        try:
            attachments[cid] = _render_single_radar_png_bytes(
                it['name'], it['dims'], _get_color(it['avg']), size=185,
                composite=it['avg'], ci=radar_ci)
        except Exception as e:  # 单只失败不影响整体
            print(f"  [perf-stock] 个股图表生成失败: {it['name']} {e}")
            continue

        avg_color = (COLOR_GREEN if it['avg'] >= 60
                     else COLOR_ORANGE if it['avg'] >= 40 else COLOR_RED)
        ds = _safe_float(it['metrics'].get('direction_skill'), 0.0)
        ds_color = COLOR_GREEN if ds >= 0.02 else COLOR_ORANGE if ds >= 0 else COLOR_RED
        lift = _safe_float(it['metrics'].get('lift'), 0.0)
        lift_color = COLOR_GREEN if lift >= 0.02 else COLOR_ORANGE if lift >= 0 else COLOR_RED
        avg_ret = it['metrics'].get('avg_return', 0)
        ret_color = COLOR_GREEN if avg_ret >= 0 else COLOR_RED
        buy_ret = _safe_float(it['metrics'].get('buy_avg_return'), 0.0)
        buy_ret_color = COLOR_GREEN if buy_ret >= 0 else COLOR_RED

        def _sig(lo_hi):
            if not lo_hi:
                return ''
            lo, hi = lo_hi
            return '✅ 显著' if lo > 0 else ('❌ 显著为负' if hi < 0 else '⚠️ 不显著')

        ds_sig = _sig(it.get('ds_ci'))
        lift_sig = _sig(it.get('lift_ci'))
        html += f"""            <td style="border: none; text-align: center; padding: 4px; vertical-align: top; width: {100.0 / items_per_row:.0f}%;">
                <div style="background: #fafafa; border-radius: 6px; padding: 4px; margin: 2px;">
                    <img src="cid:{cid}" style="width: 100%; max-width: 185px; height: auto;" alt="{it['name']}">
                    <div style="font-size: 9.5px; color: #666; margin-top: 2px; line-height: 1.55;">
                        <b style="color:#333;">{it['code']}</b><br>
                        综合 <b style="color: {avg_color};">{it['avg']:.0f}</b>
                        | 方向技能 <b style="color: {ds_color};">{ds:+.1%}</b>{(' <span style="color:#b45309;">' + ds_sig + '</span>') if ds_sig else ''}<br>
                        超额lift <b style="color: {lift_color};">{lift:+.1%}</b>{(' <span style="color:#b45309;">' + lift_sig + '</span>') if lift_sig else ''}
                        | 收益 <b style="color: {ret_color};">{avg_ret:+.1%}</b>
                        | 买入均收 <b style="color: {buy_ret_color};">{buy_ret:+.1%}</b><br>
                        n={it['total']}
                    </div>
                </div>
            </td>
"""

    # 补齐末行空单元格
    remainder = len(top_items) % items_per_row
    if remainder > 0:
        for _ in range(items_per_row - remainder):
            html += f'            <td style="border: none; width: {100.0 / items_per_row:.0f}%;"></td>\n'

    html += '        </tr>\n    </table>\n'
    if guardrail_html:
        html += '<div style="margin-top: 12px;">' + guardrail_html + '</div>\n'
    return html, attachments
