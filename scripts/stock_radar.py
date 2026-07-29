#!/usr/bin/env python3
"""
A股六边形强度图生成器 (Hexagonal Stock Strength Radar Chart)

为每只股票生成6维度雷达图，用于嵌入邮件报告。

6 Dimensions:
  - Trend:    CatBoost 20d prediction probability
  - Return:   return_score from risk-reward analysis
  - Risk:     inverted risk_score (higher = safer)
  - Tech:     MA arrangement + RSI position composite
  - Momentum: recent return momentum (1d/5d/20d)
  - Signal:   win_rate + pattern + profit_loss_ratio composite

使用方法:
    python3 scripts/stock_radar.py --code 000001 --name "PingAn" \\
        --prob-20d 0.75 --return-score 65 --risk-score 30
"""

import os
import sys
import base64
import io
import json
import argparse
from datetime import datetime

import numpy as np

# ── matplotlib setup ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Configuration ──
DIMENSIONS = ['Trend', 'Return', 'Risk', 'Tech', 'Momentum', 'Signal']

COLOR_GREEN = '#16a34a'
COLOR_ORANGE = '#ea580c'
COLOR_RED = '#dc2626'
COLOR_GRID = '#cccccc'


def compute_a_stock_dimensions(analysis=None, predictions=None, risk_reward=None,
                                win_rate=None, pl_ratio=None):
    """
    Compute 6 dimension scores (0-100) from A-share stock data.

    Args:
        analysis: dict from stock_analyses[code]
            keys: current_price, change_percent, ma5, ma20, rsi_14,
                  return_5d, return_20d
        predictions: dict of {horizon: {'probability': float}}
        risk_reward: dict with 'risk_score', 'return_score'
        win_rate: float (0-100)
        pl_ratio: float (profit/loss ratio)

    Returns:
        dict: {dim_name: score_0_100}
    """
    if analysis is None:
        analysis = {}
    if predictions is None:
        predictions = {}
    if risk_reward is None:
        risk_reward = {}

    scores = {}

    # ═══ 1. Trend: CatBoost 20d probability ═══
    prob_20d = 0.50
    if isinstance(predictions.get(20), dict):
        prob_20d = predictions[20].get('probability', 0.50)
    elif isinstance(predictions.get(20), (int, float)):
        prob_20d = predictions[20]
    if prob_20d == 0.50 and isinstance(predictions.get(5), dict):
        prob_20d = predictions[5].get('probability', 0.50)
    scores['trend'] = round(max(0, min(100, prob_20d * 100)), 1)

    # ═══ 2. Return: return_score ═══
    rs = risk_reward.get('return_score', 50)
    scores['return'] = max(0, min(100, rs if rs is not None else 50))

    # ═══ 3. Risk: inverted risk_score (higher = safer) ═══
    risk_val = risk_reward.get('risk_score', 50)
    scores['risk'] = max(0, min(100, 100 - (risk_val if risk_val is not None else 50)))

    # ═══ 4. Tech: MA arrangement + RSI ═══
    price = analysis.get('current_price', 0) or 0
    ma5 = analysis.get('ma5', 0) or 0
    ma20 = analysis.get('ma20', 0) or 0
    rsi = analysis.get('rsi_14', 50) or 50

    # MA score
    if ma5 > 0 and ma20 > 0 and price > ma5 > ma20:
        ma_score = 85
    elif ma5 > 0 and ma20 > 0 and price > ma20 >= ma5:
        ma_score = 65
    elif ma5 > 0 and ma20 > 0 and price > ma5:
        ma_score = 55
    elif ma5 > 0 and ma20 > 0 and price < ma5 < ma20:
        ma_score = 35
    elif ma5 > 0 and ma20 > 0 and price < ma5 and ma5 > ma20:
        ma_score = 25
    else:
        ma_score = 45

    # RSI score
    if rsi < 20:
        rsi_score = 85
    elif rsi < 30:
        rsi_score = 75
    elif rsi < 40:
        rsi_score = 65
    elif rsi < 50:
        rsi_score = 55
    elif rsi < 60:
        rsi_score = 50
    elif rsi < 70:
        rsi_score = 40
    elif rsi < 80:
        rsi_score = 30
    else:
        rsi_score = 20

    # Return_5d boost
    r5 = analysis.get('return_5d', 0) or 0
    if r5 > 15:
        boost = 15
    elif r5 > 8:
        boost = 8
    elif r5 > 2:
        boost = 3
    elif r5 > -2:
        boost = 0
    elif r5 > -8:
        boost = -5
    else:
        boost = -10

    scores['tech'] = round(max(0, min(100, ma_score * 0.50 + rsi_score * 0.30 + 50 * 0.20 + boost)), 1)

    # ═══ 5. Momentum: recent returns ═══
    r20 = analysis.get('return_20d', 0) or 0
    chg = analysis.get('change_percent', 0) or 0

    def ret_score(ret, bound):
        return max(0, min(100, 50 + ret / bound * 50))

    m20 = ret_score(r20, 25)
    m5d = ret_score(r5, 15)
    m1 = ret_score(chg, 10)

    scores['momentum'] = round(m20 * 0.40 + m5d * 0.35 + m1 * 0.25, 1)

    # ═══ 6. Signal: win_rate + pl_ratio ═══
    wr = win_rate if win_rate is not None else 50
    pl = pl_ratio if pl_ratio is not None else 1.0

    wr_score = max(0, min(100, wr))
    pl_score = max(0, min(100, 30 + (pl - 0.5) * 20))

    scores['signal'] = round(wr_score * 0.60 + pl_score * 0.40, 1)

    return scores


def _get_color(avg_score):
    """Determine chart color from average score."""
    if avg_score >= 60:
        return COLOR_GREEN
    elif avg_score >= 40:
        return COLOR_ORANGE
    return COLOR_RED


def generate_radar_png_base64(name, code, dimensions, size=180):
    """
    Generate hexagonal radar chart as base64-encoded PNG.

    Args:
        name: stock name
        code: stock code
        dimensions: dict of {dim_name: score_0_100}
        size: image size in pixels

    Returns:
        str: base64-encoded PNG
    """
    categories = DIMENSIONS
    N = len(categories)

    values = [dimensions.get(d.lower(), 50) for d in categories]
    avg_score = float(np.mean(values))
    main_color = _get_color(avg_score)

    # Close the polygon
    values_closed = values + values[:1]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles_closed = angles + angles[:1]

    figsize = size / 100.0
    fig, ax = plt.subplots(
        figsize=(figsize, figsize),
        subplot_kw=dict(polar=True),
        facecolor='white'
    )

    ax.set_ylim(0, 100)

    # Background grid lines
    for gv in [20, 40, 60, 80]:
        ax.plot(angles_closed, [gv] * (N + 1), color=COLOR_GRID, linewidth=0.4, alpha=0.25, zorder=1)

    # Data polygon
    ax.fill(angles_closed, values_closed, alpha=0.18, color=main_color, zorder=3)
    ax.plot(angles_closed, values_closed, 'o-', linewidth=1.5, color=main_color,
            markersize=3, zorder=4)

    # Axis labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=6.5, color='#444')

    # Y-axis labels
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], fontsize=5, color='#999')
    ax.set_rlabel_position(0)

    # Title
    plt.title(f'{name} ({code})', fontsize=7.5, pad=12, color='#333')

    ax.grid(True, alpha=0.2, color=COLOR_GRID)

    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.read()).decode('utf-8')


def generate_html_radar_section(three_horizon_results, stock_analyses, max_stocks=None):
    """
    Generate HTML block with radar charts for email embedding.

    Returns (html, attachments):
        html: HTML string with <img src="cid:radar_{code}"> references
        attachments: dict of {cid_name: png_bytes} for MIME inline embedding

    Args:
        three_horizon_results: dict of {code: data}
        stock_analyses: dict of {code: analysis}
        max_stocks: max stocks to display (None = all)
    """
    if not three_horizon_results:
        return '', {}

    radar_items = []
    attachments = {}

    for code, data in three_horizon_results.items():
        analysis = stock_analyses.get(code, {})
        preds = data.get('predictions', {})
        risk_reward = data.get('risk_reward', {})
        win_rate = data.get('win_rate', 50)
        pl_ratio = data.get('profit_loss_ratio', 1.0)

        dims = compute_a_stock_dimensions(
            analysis=analysis,
            predictions=preds,
            risk_reward=risk_reward,
            win_rate=win_rate,
            pl_ratio=pl_ratio
        )

        if max(dims.values()) < 10:
            continue

        name = analysis.get('name', code)
        avg_score = float(np.mean(list(dims.values())))

        # Generate chart as PNG bytes (not base64)
        try:
            png_bytes = _generate_radar_png_bytes(name, code, dims, size=180)
            cid = f"radar_{code}"
            attachments[cid] = png_bytes
            radar_items.append({
                'code': code,
                'name': name,
                'avg_score': avg_score,
                'cid': cid,
                'dims': dims
            })
        except Exception as e:
            print(f"  [radar] Warning: {code} {name} chart failed: {e}")
            continue

    if not radar_items:
        return '', {}

    # Sort by avg score descending
    radar_items.sort(key=lambda x: x['avg_score'], reverse=True)

    if max_stocks:
        radar_items = radar_items[:max_stocks]

    html = """
    <h2 style="color: #007bff; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 5px;">
    Stock Strength Radar
    </h2>
    <p style="color: #666; font-size: 11px; margin: 5px 0 10px 0;">
    6-dimension: Trend / Return / Risk / Tech / Momentum / Signal
    | <span style="color:#16a34a;">Avg &ge; 60</span>
    | <span style="color:#ea580c;">40 &le; Avg &lt; 60</span>
    | <span style="color:#dc2626;">Avg &lt; 40</span>
    </p>
    <table style="border: 0; border-collapse: collapse; width: 100%;">
        <tr>
"""
    items_per_row = 4
    for i, item in enumerate(radar_items):
        if i % items_per_row == 0 and i > 0:
            html += """        </tr><tr>
"""
        avg_color = '#16a34a' if item['avg_score'] >= 60 else ('#ea580c' if item['avg_score'] >= 40 else '#dc2626')
        dims = item['dims']
        html += f"""            <td style="border: none; text-align: center; padding: 5px; vertical-align: top; width: 25%;">
                <div style="background: #fafafa; border-radius: 6px; padding: 5px; margin: 2px;">
                    <img src="cid:{item['cid']}" style="width: 100%; max-width: 180px; height: auto;" alt="{item['name']}">
                    <div style="font-size: 9px; color: #666; margin-top: 2px;">
                        Avg: <span style="color: {avg_color}; font-weight: bold;">{item['avg_score']:.1f}</span>
                        | T:{dims.get('trend', 0):.0f} R:{dims.get('return', 0):.0f} K:{dims.get('risk', 0):.0f}
                        | M:{dims.get('momentum', 0):.0f} S:{dims.get('signal', 0):.0f}
                    </div>
                </div>
            </td>
"""
    # Fill remaining cells
    remainder = len(radar_items) % items_per_row
    if remainder > 0:
        for _ in range(items_per_row - remainder):
            html += """            <td style="border: none; width: 25%;"></td>
"""

    html += """        </tr>
    </table>
"""
    return html, attachments


def _generate_radar_png_bytes(name, code, dimensions, size=180):
    """Generate radar chart, return raw PNG bytes (for CID email embedding)."""
    categories = DIMENSIONS
    N = len(categories)

    values = [dimensions.get(d.lower(), 50) for d in categories]
    avg_score = float(np.mean(values))
    main_color = _get_color(avg_score)

    values_closed = values + values[:1]
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles_closed = angles + angles[:1]

    figsize = size / 100.0
    fig, ax = plt.subplots(
        figsize=(figsize, figsize),
        subplot_kw=dict(polar=True),
        facecolor='white'
    )

    ax.set_ylim(0, 100)
    for gv in [20, 40, 60, 80]:
        ax.plot(angles_closed, [gv] * (N + 1), color=COLOR_GRID, linewidth=0.4, alpha=0.25, zorder=1)
    ax.fill(angles_closed, values_closed, alpha=0.18, color=main_color, zorder=3)
    ax.plot(angles_closed, values_closed, 'o-', linewidth=1.5, color=main_color, markersize=3, zorder=4)
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=6.5, color='#444')
    ax.set_yticks([20, 40, 60, 80])
    ax.set_yticklabels(['20', '40', '60', '80'], fontsize=5, color='#999')
    ax.set_rlabel_position(0)
    plt.title(f'{name} ({code})', fontsize=7.5, pad=12, color='#333')
    ax.grid(True, alpha=0.2, color=COLOR_GRID)
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.close()
    return buf.getvalue()


# ── CLI ──
def main():
    parser = argparse.ArgumentParser(description='A-Share Stock Strength Radar Chart')
    parser.add_argument('--code', default='000000', help='stock code')
    parser.add_argument('--name', default='Unknown', help='stock name')
    parser.add_argument('--prob-20d', type=float, default=0.50, help='CatBoost 20d prob')
    parser.add_argument('--return-score', type=float, default=50, help='return_score')
    parser.add_argument('--risk-score', type=float, default=50, help='risk_score')
    parser.add_argument('--ma5', type=float, default=0, help='MA5 price')
    parser.add_argument('--ma20', type=float, default=0, help='MA20 price')
    parser.add_argument('--price', type=float, default=0, help='current price')
    parser.add_argument('--rsi', type=float, default=50, help='RSI(14)')
    parser.add_argument('--return-5d', type=float, default=0, help='5d return %')
    parser.add_argument('--return-20d', type=float, default=0, help='20d return %')
    parser.add_argument('--change', type=float, default=0, help='today change %')
    parser.add_argument('--win-rate', type=float, default=50, help='historical win rate')
    parser.add_argument('--pl-ratio', type=float, default=1.0, help='profit/loss ratio')
    parser.add_argument('--output', '-o', default=None, help='save PNG to file')
    parser.add_argument('--batch', default=None, help='batch JSON input file')
    parser.add_argument('--output-dir', default='output/radar_charts', help='batch output dir')

    args = parser.parse_args()

    if args.batch:
        with open(args.batch, 'r') as f:
            stocks_data = json.load(f)
        os.makedirs(args.output_dir, exist_ok=True)
        for item in stocks_data:
            dims = compute_a_stock_dimensions(
                analysis=item.get('analysis', {}),
                predictions=item.get('predictions', {}),
                risk_reward=item.get('risk_reward', {}),
                win_rate=item.get('win_rate', 50),
                pl_ratio=item.get('pl_ratio', 1.0)
            )
            name = item.get('name', item.get('analysis', {}).get('name', 'N/A'))
            code = item.get('code', 'unknown')
            b64 = generate_radar_png_base64(name, code, dims)
            out_path = os.path.join(args.output_dir, f"{code}.png")
            with open(out_path, 'wb') as f:
                f.write(base64.b64decode(b64))
            print(f"  {code} {name} -> {out_path}")
        return

    # Single stock mode
    analysis = {
        'current_price': args.price,
        'change_percent': args.change,
        'ma5': args.ma5,
        'ma20': args.ma20,
        'rsi_14': args.rsi,
        'return_5d': args.return_5d,
        'return_20d': args.return_20d,
    }
    predictions = {20: {'probability': args.prob_20d}}
    risk_reward = {'risk_score': args.risk_score, 'return_score': args.return_score}

    dims = compute_a_stock_dimensions(analysis, predictions, risk_reward, args.win_rate, args.pl_ratio)

    print("\nStock Strength Radar Scores:")
    print("=" * 40)
    for k, v in dims.items():
        bar = '#' * int(v / 5) + '-' * (20 - int(v / 5))
        print(f"  {k:8s} | {v:5.1f} | {bar}")
    avg = np.mean(list(dims.values()))
    print(f"  {'AVERAGE':8s} | {avg:5.1f}")

    b64 = generate_radar_png_base64(args.name, args.code, dims)

    if args.output:
        with open(args.output, 'wb') as f:
            f.write(base64.b64decode(b64))
        print(f"\nSaved: {args.output}")
    else:
        print(f"\nBase64 (first 50): {b64[:50]}...")


if __name__ == '__main__':
    main()