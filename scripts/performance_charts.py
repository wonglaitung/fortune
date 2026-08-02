#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测性能报告可视化图表生成器 (Performance Report Charts)

为港股「预测性能报告」邮件生成可视化图表，风格与 A股雷达图保持一致：
matplotlib 渲染 PNG → CID 内嵌进 HTML 邮件（send_email_with_images）。

图表集：
  1. 三周期整体性能雷达（small multiples：1天/5天/20天 各一张，5 维度）
  2. 各周期时间窗口准确率柱状图（small multiples：1天/5天/20天 各一组）
  3. 板块性能雷达网格（每个板块一张，按综合分三色上色）
  4. 三周期模式胜率水平柱状图（8 种模式，按胜率三色上色）

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

# 雷达 5 维度（均为真实模型质量指标，归一化到 0-100；不含样本量/正收益占比等环境敏感量）
# 注：样本量 n 不是模型能力维度，故不作雷达轴，仅在图下方文字中标注作为可信度上下文
PERF_DIMENSIONS = ['准确率', '平均收益', '夏普比率', '买入胜率', '买入平均收益']

# 综合分加权（胜率为主、准确率为辅）：买入胜率×3 主导，准确率×0.5 降权，其余×1。
# 仅用于「综合分」这一标量（排名条 / Top10 选取 / 状态三色 / 雷达标题综合分）；
# 雷达 5 轴多边形本身不加权（极坐标对单轴加权会扭曲形状，5 维仍等权呈现）。
PERF_DIM_WEIGHTS = {
    '准确率': 0.5,
    '平均收益': 1.0,
    '夏普比率': 1.0,
    '买入胜率': 3.0,
    '买入平均收益': 1.0,
}


def composite_score(dimensions):
    """5 维度按 PERF_DIM_WEIGHTS 加权平均 → 综合分（0-100）。

    与等权均值不同：买入胜率权重最高，使综合分/排名向"喊涨命中率"倾斜；
    准确率降权为辅。雷达多边形形状不受影响。
    """
    wsum = 0.0
    wtot = 0.0
    for k, w in PERF_DIM_WEIGHTS.items():
        wsum += w * dimensions.get(k, 0)
        wtot += w
    return wsum / wtot if wtot > 0 else 0.0

# 归一化常量（在图下方 caption 中向读者说明）
RETURN_BOUND = 0.15      # 平均收益 / 买入平均收益 ±15% 映射到 [0, 100]（50 = 零收益）
# 单期夏普比率居中映射 [-1, +1] → [0, 100]（50 = 零夏普）。
# 注：calculate_metrics 的 sharpe 是单持有期信噪比(mean/std)，真实量级 ~±1；
#     不能年化(滚动样本高度重叠、违反 i.i.d.)，故上限取单期量级 1.0 而非年化的 3.0；
#     居中后负值显示为 <50、不再被截断贴底，轴才有区分度。
SHARPE_BOUND = 1.0

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


def normalize_accuracy(accuracy):
    """方向准确率 (0-1) → 0-100"""
    return max(0.0, min(100.0, _safe_float(accuracy) * 100))


def normalize_return(avg_return):
    """平均收益：clamp 到 ±15% 后线性映射 [0, 100]，50 = 零收益（NaN 记中性 50）"""
    r = max(-RETURN_BOUND, min(RETURN_BOUND, _safe_float(avg_return, 0.0)))
    return (r + RETURN_BOUND) / (2 * RETURN_BOUND) * 100


def normalize_sharpe(sharpe):
    """单期夏普比率：居中映射 [-1, +1] → [0, 100]，50 = 零夏普（NaN 记中性 50）。
    负值显示为 <50、不再贴底；正负两侧对称，轴具备区分度。"""
    s = max(-SHARPE_BOUND, min(SHARPE_BOUND, _safe_float(sharpe, 0.0)))
    return (s + SHARPE_BOUND) / (2 * SHARPE_BOUND) * 100


def normalize_buy_win_rate(win_rate):
    """买入信号胜率 (0-1) → 0-100"""
    return max(0.0, min(100.0, _safe_float(win_rate) * 100))


def metrics_to_dimensions(metrics):
    """
    将 calculate_metrics() 的指标字典转换为雷达 5 维度分（0-100）。

    参数:
    - metrics: performance_monitor.calculate_metrics() 的返回值

    返回:
    - {维度名: 分数}（仅含模型质量维度；样本量 n 不作轴）
    """
    if not metrics:
        metrics = {}
    return {
        '准确率': round(normalize_accuracy(metrics.get('accuracy')), 1),
        '平均收益': round(normalize_return(metrics.get('avg_return')), 1),
        '夏普比率': round(normalize_sharpe(metrics.get('sharpe_ratio')), 1),
        '买入胜率': round(normalize_buy_win_rate(metrics.get('buy_win_rate')), 1),
        # 买入平均收益复用 ±15% 映射：衡量"喊涨时平均赚多少"，纯质量、无市场涨跌干扰
        '买入平均收益': round(normalize_return(metrics.get('buy_avg_return')), 1),
    }


def _strip_unsafe_glyphs(text):
    """去除 matplotlib 中文字体可能缺失的字符（如 emoji ⭐），保留中英文/数字/括号。"""
    return re.sub(r'[^\w一-鿿（）()·\- ]', '', str(text)).strip()


# ════════════════════════════════════════════════════════════
# 基础渲染：单系列雷达图 PNG
# ════════════════════════════════════════════════════════════

def _render_single_radar_png_bytes(title, dimensions, color, size=220, composite=None):
    """
    渲染单系列多边形雷达图，返回 PNG bytes（轴数 = PERF_DIMENSIONS 维度数）。

    参数:
    - title: 图表标题（如 "1天" / "银行股"）
    - dimensions: {维度名: 0-100 分数}
    - color: 填充/描边颜色
    - size: 图片尺寸（像素，换算为 figsize）
    - composite: 标题"综合 X"用的标量；None 时回退 5 维等权均值（兼容旧调用）。
      调用方应传加权综合分 composite_score(dimensions)，使标题与排名口径一致。
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

def generate_overall_radar_section(horizon_metrics, window_name='3个月'):
    """
    生成「模型整体性能雷达」HTML 区块 + CID 附件。

    参数:
    - horizon_metrics: {周期: calculate_metrics() 结果}（建议传入 3个月窗口）
    - window_name: 统计窗口名称（用于说明文字）

    返回: (html, {cid: png_bytes})
    """
    cells = []
    attachments = {}

    for h in HORIZONS:
        m = horizon_metrics.get(h) or {}
        if m.get('total_predictions', 0) == 0:
            continue
        dims = metrics_to_dimensions(m)
        color = HORIZON_COLORS[h]
        cid = f'perf_radar_{h}d'
        attachments[cid] = _render_single_radar_png_bytes(
            f'{HORIZON_NAMES[h]}周期', dims, color, size=230,
            composite=composite_score(dims))

        # 关键指标（墨色文本，准确率按状态上色并带数值 = 非颜色唯一编码）
        acc = m.get('accuracy', 0)
        acc_color = (COLOR_GREEN if acc >= 0.60
                     else COLOR_ORANGE if acc >= 0.50 else COLOR_RED)
        avg_ret = m.get('avg_return', 0)
        ret_color = COLOR_GREEN if avg_ret >= 0 else COLOR_RED
        buy_ret = _safe_float(m.get('buy_avg_return'), 0.0)
        buy_ret_color = COLOR_GREEN if buy_ret >= 0 else COLOR_RED
        buy_wr = _safe_float(m.get('buy_win_rate'), 0.0)
        cells.append(f"""            <td style="border: none; text-align: center; padding: 6px; vertical-align: top; width: 33%;">
                <div style="background: #fafafa; border-radius: 8px; padding: 8px; margin: 2px;">
                    <img src="cid:{cid}" style="width: 100%; max-width: 230px; height: auto;" alt="{HORIZON_NAMES[h]}周期性能雷达">
                    <div style="font-size: 11px; color: #666; margin-top: 4px; line-height: 1.7;">
                        样本 <b style="color:#333;">{m.get('total_predictions', 0)}</b><br>
                        准确率 <b style="color: {acc_color};">{acc:.2%}</b><br>
                        平均收益 <b style="color: {ret_color};">{avg_ret:+.2%}</b><br>
                        买入胜率 <b style="color:#333;">{buy_wr:.1%}</b>
                        · 买入均收 <b style="color: {buy_ret_color};">{buy_ret:+.2%}</b><br>
                        夏普 <b style="color:#333;">{m.get('sharpe_ratio', 0):.2f}</b>
                    </div>
                </div>
            </td>
""")

    if not cells:
        return '', {}

    html = _SECTION_H2.format(title='一、模型整体性能雷达')
    html += _CAPTION.format(
        text=f'统计窗口：{window_name} | 5 维度均为模型质量指标，归一化至 0–100：'
             '准确率=方向正确率 · 平均收益=全样本±15%映射(50为零收益) · 夏普=单期±1居中映射(50为零,负值<50) · '
             '买入胜率=预测上涨样本正确率 · 买入平均收益=喊涨样本平均收益(同±15%映射) | '
             '样本量 n 见各图下方文字（仅作可信度参考，不参与雷达形状）| '
             '标题"综合"=加权综合分(买入胜率×3·准确率×0.5·其余×1，胜率为主，雷达5轴形状仍等权) | '
             '颜色深浅区分周期（浅=1天 → 深=20天）')
    html += '    <table style="border: 0; border-collapse: collapse; width: 100%;">\n        <tr>\n'
    html += ''.join(cells)
    html += '        </tr>\n    </table>\n'
    return html, attachments


# ════════════════════════════════════════════════════════════
# 章节 2：各周期时间窗口准确率柱状图（small multiples）
# ════════════════════════════════════════════════════════════

def generate_window_bar_section(window_metrics,
                                time_windows=((30, '1个月'), (90, '3个月'), (180, '6个月'))):
    """
    生成「各周期时间窗口表现」HTML 区块 + CID 附件：
    上下两张图，结构一致——上图为准确率、下图为买入胜率（新增），
    每张图内 3 个子图（1天/5天/20天），每个子图展示 1个月/3个月/6个月。

    参数:
    - window_metrics: {窗口天数: {周期: metrics}}
    - time_windows: [(天数, 名称), ...]

    返回: (html, {cid: png_bytes})
    """
    windows = list(time_windows)
    x = np.arange(len(windows))
    xlabels = [name for _, name in windows]

    # (metrics键, 图标题, 纵轴名) —— 准确率 + 买入胜率，虚线均为 50% 基准
    panels = [
        ('perf_window_acc', 'accuracy', '各周期在不同时间窗口的准确率', '准确率'),
        ('perf_window_bwr', 'buy_win_rate', '各周期在不同时间窗口的买入胜率', '买入胜率'),
    ]

    attachments = {}
    imgs = []
    for cid, key, suptitle, ylabel in panels:
        fig, axes = plt.subplots(1, len(HORIZONS), figsize=(9.6, 3.1),
                                 facecolor='white', sharey=True)
        if len(HORIZONS) == 1:
            axes = [axes]

        any_data = False
        for ax, h in zip(axes, HORIZONS):
            vals, counts = [], []
            for days, _ in windows:
                m = (window_metrics.get(days, {}) or {}).get(h, {}) or {}
                vals.append(float(m.get(key, 0)) * 100)
                counts.append(int(m.get('total_predictions', 0)))
            if any(c > 0 for c in counts):
                any_data = True

            color = HORIZON_COLORS[h]
            ax.bar(x, vals, color=color, width=0.62,
                   edgecolor='white', linewidth=0.8, zorder=3)
            ax.axhline(50, color=INK_FAINT, linestyle='--', linewidth=0.9,
                       alpha=0.8, zorder=2)  # 50% 基准（随机方向 / 抛硬币）
            ax.set_title(f'{HORIZON_NAMES[h]}周期', fontsize=10.5, color=INK, pad=8)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=8.5, color=INK_MUTED)
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            _style_bar_axis(ax)

            # 数值直接标注（墨色；无样本标灰）
            for xi, v, c in zip(x, vals, counts):
                if c > 0:
                    ax.text(xi, v + 2, f'{v:.0f}%', ha='center', va='bottom',
                            fontsize=8.5, color=INK, zorder=5)
                else:
                    ax.text(xi, 3, '无样本', ha='center', va='bottom',
                            fontsize=7.5, color=INK_FAINT, zorder=5)

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
        text='上图=准确率（方向预测正确比例）| 下图=买入胜率（模型喊涨的样本中真涨比例，'
             '与实盘盈亏直接挂钩）| 虚线均为 50% 基准 | '
             '颜色深浅区分周期（浅=1天 → 深=20天）')
    for cid, suptitle in imgs:
        html += ('    <div style="text-align: center; margin: 6px 0;">'
                 f'<img src="cid:{cid}" style="max-width: 680px; width: 100%; height: auto;" '
                 f'alt="{suptitle}"></div>\n')
    return html, attachments


# ════════════════════════════════════════════════════════════
# 章节 3：板块性能雷达网格
# ════════════════════════════════════════════════════════════

def generate_sector_radar_section(sector_metrics, min_samples=5, items_per_row=4):
    """
    生成「板块性能雷达」网格 HTML 区块 + CID 附件（仿 A股个股雷达网格）。

    参数:
    - sector_metrics: {板块代码: {'name': 中文名, 'metrics': calculate_metrics() 结果}}
    - min_samples: 最小样本数，低于此值的板块跳过（并打印日志，不做静默截断）
    - items_per_row: 每行几张

    返回: (html, {cid: png_bytes})
    """
    items = []
    attachments = {}
    dropped = []

    for sector, info in sector_metrics.items():
        m = info.get('metrics') or {}
        total = int(m.get('total_predictions', 0))
        name = info.get('name', sector)
        if total < min_samples:
            dropped.append((name, total))
            continue

        dims = metrics_to_dimensions(m)
        avg = composite_score(dims)  # 加权综合分（胜率为主、准确率为辅）
        color = _get_color(avg)  # 状态三色：≥60 绿 / 40-60 橙 / <40 红
        cid = f'perf_sector_{sector}'
        try:
            attachments[cid] = _render_single_radar_png_bytes(name, dims, color, size=190, composite=avg)
        except Exception as e:  # 单板块失败不影响整体
            print(f'  [perf-radar] 板块图表生成失败: {name} {e}')
            continue

        items.append({
            'name': name,
            'avg': avg,
            'total': total,
            'accuracy': m.get('accuracy', 0),
            'avg_return': m.get('avg_return', 0),
            'buy_avg_return': m.get('buy_avg_return', 0),
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
             '5 维度同整体雷达 | 综合分=加权(买入胜率×3·准确率×0.5·其余×1，胜率为主) | 颜色为综合分状态：'
             '<span style="color:#16a34a;">≥60</span> / '
             '<span style="color:#ea580c;">40–60</span> / '
             '<span style="color:#dc2626;">&lt;40</span>')
    html += '    <table style="border: 0; border-collapse: collapse; width: 100%;">\n        <tr>\n'

    for i, it in enumerate(items):
        if i % items_per_row == 0 and i > 0:
            html += '        </tr><tr>\n'
        avg_color = (COLOR_GREEN if it['avg'] >= 60
                     else COLOR_ORANGE if it['avg'] >= 40 else COLOR_RED)
        acc_color = (COLOR_GREEN if it['accuracy'] >= 0.60
                     else COLOR_ORANGE if it['accuracy'] >= 0.50 else COLOR_RED)
        ret_color = COLOR_GREEN if it['avg_return'] >= 0 else COLOR_RED
        buy_ret = _safe_float(it.get('buy_avg_return'), 0.0)
        buy_ret_color = COLOR_GREEN if buy_ret >= 0 else COLOR_RED
        html += f"""            <td style="border: none; text-align: center; padding: 5px; vertical-align: top; width: {100.0 / items_per_row:.0f}%;">
                <div style="background: #fafafa; border-radius: 6px; padding: 5px; margin: 2px;">
                    <img src="cid:{it['cid']}" style="width: 100%; max-width: 190px; height: auto;" alt="{it['name']}">
                    <div style="font-size: 10px; color: #666; margin-top: 2px; line-height: 1.6;">
                        综合 <b style="color: {avg_color};">{it['avg']:.0f}</b>
                        | 准确率 <b style="color: {acc_color};">{it['accuracy']:.1%}</b><br>
                        收益 <b style="color: {ret_color};">{it['avg_return']:+.1%}</b>
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
    return html, attachments


# ════════════════════════════════════════════════════════════
# 章节 4：三周期模式胜率水平柱状图
# ════════════════════════════════════════════════════════════

def generate_pattern_bar_section(pattern_stats, pattern_names):
    """
    生成「三周期模式胜率」水平柱状图 HTML 区块 + CID 附件。

    参数:
    - pattern_stats: {模式: {'total', 'correct', 'win_rate', 'avg_return'}}
    - pattern_names: {模式: 中文名}

    返回: (html, {cid: png_bytes})
    """
    if not pattern_stats:
        return '', {}

    # 升序排列 → barh 后最优模式在顶部
    ordered = sorted(pattern_stats.items(), key=lambda kv: kv[1].get('win_rate', 0))
    labels, win_rates, totals, colors = [], [], [], []
    for pattern, stats in ordered:
        name = _strip_unsafe_glyphs(pattern_names.get(pattern, ''))
        labels.append(f'{pattern} {name}'.strip())
        wr = float(stats.get('win_rate', 0)) * 100
        win_rates.append(wr)
        totals.append(int(stats.get('total', 0)))
        # 状态三色（带数值标签，颜色非唯一编码）
        colors.append(COLOR_GREEN if wr >= 60
                      else COLOR_ORANGE if wr >= 50 else COLOR_RED)

    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(8.0, max(2.8, 0.5 * len(ordered) + 1.4)),
                           facecolor='white')
    ax.barh(y, win_rates, color=colors, height=0.62,
            edgecolor='white', linewidth=0.8, zorder=3)
    ax.axvline(50, color=INK_FAINT, linestyle='--', linewidth=0.9,
               alpha=0.8, zorder=2)  # 50% 基准
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('20天方向胜率', fontsize=9.5, color=INK_MUTED)
    ax.set_title('三周期模式胜率（3个月窗口）', fontsize=12, color=INK, pad=10)
    _style_bar_axis(ax)

    # 数值 + 样本量直接标注（墨色）
    for yi, wr, t in zip(y, win_rates, totals):
        ax.text(wr + 1.5, yi, f'{wr:.1f}%  (n={t})', va='center', ha='left',
                fontsize=8.8, color=INK, zorder=5)

    fig.tight_layout()

    cid = 'perf_pattern_bar'
    attachments = {cid: _save_png_bytes(fig)}

    html = _SECTION_H2.format(title='四、三周期模式验证')
    html += _CAPTION.format(
        text='模式编码：110 = 1天涨·5天涨·20天跌 | 胜率 = 该模式下 20天方向命中率 | '
             '虚线为 50% 基准 | 颜色：'
             '<span style="color:#16a34a;">≥60%</span> / '
             '<span style="color:#ea580c;">50–60%</span> / '
             '<span style="color:#dc2626;">&lt;50%</span>')
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
    ax.set_xlabel('综合分（加权：胜率为主）', fontsize=9.5, color=INK_MUTED)
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


def generate_stock_section(stock_bundle, min_samples=5, top_n=10, items_per_row=5):
    """
    生成「个股表现」HTML 区块 + CID 附件：
      - 一张水平条形图：全部个股按综合分排名（紧凑、便于横向对比）
      - Top N 个股雷达网格：综合分最高者单独展示 5 维度细节

    参数:
    - stock_bundle: {代码: {'name', 'code', 'sector', 'metrics'}}
    - min_samples: 最小样本数，低于此值的个股跳过（并打印日志，不静默截断）
    - top_n: 雷达网格展示的个股数量
    - items_per_row: 雷达网格每行几张

    返回: (html, {cid: png_bytes})
    """
    items = []
    dropped = []

    for code, info in stock_bundle.items():
        m = info.get('metrics') or {}
        total = int(m.get('total_predictions', 0))
        name = info.get('name', code)
        if total < min_samples:
            dropped.append((name, code, total))
            continue

        dims = metrics_to_dimensions(m)
        avg = composite_score(dims)  # 加权综合分（胜率为主、准确率为辅）
        items.append({
            'code': code,
            'name': name,
            'label': f'{_strip_unsafe_glyphs(name)} {code}',
            'avg': avg,
            'total': total,
            'dims': dims,
            'metrics': m,
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
             '5 维度同整体雷达，综合分=加权(买入胜率×3·准确率×0.5·其余×1，胜率为主) | 状态三色：'
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
        try:
            attachments[cid] = _render_single_radar_png_bytes(
                it['name'], it['dims'], _get_color(it['avg']), size=185,
                composite=it['avg'])
        except Exception as e:  # 单只失败不影响整体
            print(f"  [perf-stock] 个股图表生成失败: {it['name']} {e}")
            continue

        avg_color = (COLOR_GREEN if it['avg'] >= 60
                     else COLOR_ORANGE if it['avg'] >= 40 else COLOR_RED)
        acc = it['metrics'].get('accuracy', 0)
        acc_color = (COLOR_GREEN if acc >= 0.60
                     else COLOR_ORANGE if acc >= 0.50 else COLOR_RED)
        avg_ret = it['metrics'].get('avg_return', 0)
        ret_color = COLOR_GREEN if avg_ret >= 0 else COLOR_RED
        buy_ret = _safe_float(it['metrics'].get('buy_avg_return'), 0.0)
        buy_ret_color = COLOR_GREEN if buy_ret >= 0 else COLOR_RED

        html += f"""            <td style="border: none; text-align: center; padding: 4px; vertical-align: top; width: {100.0 / items_per_row:.0f}%;">
                <div style="background: #fafafa; border-radius: 6px; padding: 4px; margin: 2px;">
                    <img src="cid:{cid}" style="width: 100%; max-width: 185px; height: auto;" alt="{it['name']}">
                    <div style="font-size: 9.5px; color: #666; margin-top: 2px; line-height: 1.55;">
                        <b style="color:#333;">{it['code']}</b><br>
                        综合 <b style="color: {avg_color};">{it['avg']:.0f}</b>
                        | 准确率 <b style="color: {acc_color};">{acc:.1%}</b><br>
                        收益 <b style="color: {ret_color};">{avg_ret:+.1%}</b>
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
    return html, attachments
