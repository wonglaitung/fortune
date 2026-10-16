#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股题材小宇宙 —— 关联关系建图器（银行 / 人工智能 / 芯片）

用"图关联关系"方法，把 `hk_themes_config.py` 里的**种子 + 边**拼成一张**带类型的图**，
并做**社区对照**：数据自己发现的社区（Louvain，基于价格相关 / 南向持仓）
⇄ 人为题材标签（银行 / AI / 芯片）。

设计要点（方法学，避免循环论证）：
- **E2 同业**：按定义连（同题材）= 结构边。**Louvain 用它做对照会循环**（必然复现题材）
  ⇒ 社区对照**只用经验边**（E6 价格相关 / E5 南向），结构边仅用于覆盖与查表。
- **E6 需行情**（网络）⇒ 默认关闭（`--with-correlation` 打开）。离线时只用 E5（本地 CSV）。
- **E3 产业链 / E4 集团**：人工判断、无 PIT ⇒ 软边，输出里显式标注。

输出（默认 `data/hk_themes/`）：
    theme_graph_nodes.csv      节点（code,name,theme,sector,is_core,market）
    theme_graph_edges.csv      类型化边（u,v,edge_type,weight）
    theme_ah_links.csv         E1 A-H 双重上市对（hk_code,a_code,name）
    theme_communities.json     经验社区 + 与题材的对照指标
    theme_graph_summary.json   汇总

用法：
    python3 scripts/build_hk_themes_graph.py                       # 离线（E2/E3/E4/E5 + 社区对照用 E5）
    python3 scripts/build_hk_themes_graph.py --with-correlation    # 加 E6 价格相关（需网络）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hk_themes_config as cfg  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "data" / "hk_themes"
SOUTHBOUND_CSV = ROOT / "data" / "southbound_holdings" / "southbound_long.csv"


# ------------------------------------------------------------------ E5 南向
def load_southbound_top(csv_path: Path, universe: set, top_k: int = 10):
    """读南向持仓 CSV，取**最新交易日**持有比例最高的 top_k（限宇宙内）。"""
    if not csv_path.exists():
        return []
    latest_date, rows = None, []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            code = r.get("pool_code")
            if code not in universe:
                continue
            d = r.get("date")
            try:
                ratio = float(r.get("hold_ratio") or 0.0)
            except ValueError:
                ratio = 0.0
            if latest_date is None or d > latest_date:
                latest_date, rows = d, [(code, ratio)]
            elif d == latest_date:
                rows.append((code, ratio))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [(c, r) for c, r in rows[:top_k]]


# ------------------------------------------------------------------ 建边
def build_edges(southbound_top):
    """返回类型化边列表 [{'u','v','edge_type','weight','hard'}]。"""
    edges = []
    universe = set(cfg.get_all_stocks())

    def add(u, v, etype, weight):
        if u in universe and v in universe and u != v:
            edges.append({
                "u": u, "v": v, "edge_type": etype,
                "weight": weight, "hard": etype in cfg.HARD_EDGE_TYPES,
            })

    # E2 同业：同题材内（种子 + E2 成员）全连 —— 结构边（按定义，非经验）
    for theme in cfg.THEMES:
        members = set(cfg.get_seed_stocks(theme))
        for c in cfg.HK_THEMES_EXTENSION.get(theme, {}).get("E2_PEER", {}):
            members.add(c)
        for u, v in combinations(sorted(members), 2):
            add(u, v, "E2_PEER", 1.0)

    # E3 产业链（软边）
    for u, v, _ in cfg.HK_THEMES_SUPPLY:
        add(u, v, "E3_SUPPLY", 0.5)

    # E4 集团（软边）
    for grp in cfg.HK_THEMES_GROUPS.values():
        for u, v in combinations(sorted(grp["members"]), 2):
            add(u, v, "E4_GROUP", 0.5)

    # E5 南向共同持仓（硬边；数据推导）
    for (u, _), (v, _) in combinations(sorted(southbound_top), 2):
        add(u, v, "E5_SOUTHBOUND", 0.7)

    # 去重（同 u-v 多类型保留，但同类型不重复）
    seen, uniq = set(), []
    for e in edges:
        key = (e["u"], e["v"], e["edge_type"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    uniq.sort(key=lambda e: (e["edge_type"], e["u"], e["v"]))
    return uniq


# ------------------------------------------------------------------ E6 相关
def build_correlation_edges(codes, period="2y", threshold=0.5):
    """E6：价格相关阈值网络（需网络）。返回 (edges, partition, modularity)。"""
    from ml_services import stock_network_analysis as sna

    data = sna.fetch_all_stock_data(list(codes), period=period)
    if len(data) < 5:
        print("  ⚠️ 行情不足，跳过 E6")
        return [], {}, 0.0
    returns_df = sna.build_returns_dataframe(data)
    pearson, _ = sna.compute_correlation_matrices(returns_df)
    graph = sna.build_threshold_network(pearson, list(returns_df.columns), threshold=threshold)
    partition, modularity = sna.detect_communities(graph)

    edges = []
    for u, v, d in graph.edges(data=True):
        w = float(d.get("weight", 1.0))
        edges.append({"u": u, "v": v, "edge_type": "E6_CORR",
                      "weight": round(w, 4), "hard": True})
    return edges, partition, modularity


# ------------------------------------------------------------------ 社区对照
def theme_purity(partition, mapping):
    """社区纯度：每个社区取最高频题材，命中数 / 总数。"""
    if not partition:
        return None
    buckets = defaultdict(lambda: defaultdict(int))
    for node, comm in partition.items():
        theme = mapping.get(node, {}).get("theme")
        if theme:
            buckets[comm][theme] += 1
    total = sum(sum(v.values()) for v in buckets.values())
    hit = sum(max(v.values()) for v in buckets.values())
    return {
        "purity": round(hit / total, 4) if total else None,
        "contingency": {str(c): dict(v) for c, v in sorted(buckets.items())},
        "n_communities": len(buckets),
    }


def cross_check(partition, mapping):
    """经验社区 ⇄ 题材标签 的对照；含 ARI（有 sklearn 则算）。"""
    out = theme_purity(partition, mapping)
    if out is None:
        return None
    labels, preds = [], []
    node_to_comm = partition
    for node in partition:
        theme = mapping.get(node, {}).get("theme")
        if theme:
            labels.append(theme)
            preds.append(node_to_comm[node])
    try:
        import warnings
        from sklearn.metrics import adjusted_rand_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["ari"] = round(float(adjusted_rand_score(labels, preds)), 4)
    except Exception:
        out["ari"] = None
    return out


def estimate_n_eff(edges, mapping, theme):
    """用同题材平均相关（E6 边权）估有效独立数 N/(1+(N-1)·r̄)；无 E6 时返回 None。"""
    codes = [c for c, m in mapping.items() if m["theme"] == theme]
    if len(codes) < 2:
        return None
    pair_w = [e["weight"] for e in edges
              if e["edge_type"] == "E6_CORR" and e["u"] in codes and e["v"] in codes]
    if not pair_w:
        return None
    r = sum(pair_w) / len(pair_w)
    n = len(codes)
    return round(n / (1 + (n - 1) * max(r, 0.0)), 2)


# ------------------------------------------------------------------ 输出
def write_outputs(out_dir: Path, nodes, edges, ah_links, communities, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "theme_graph_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "theme", "sector", "is_core", "market"])
        for n in nodes:
            w.writerow([n["code"], n["name"], n["theme"], n["sector"], n["is_core"], n["market"]])
    with (out_dir / "theme_graph_edges.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "edge_type", "weight", "hard"])
        for e in edges:
            w.writerow([e["u"], e["v"], e["edge_type"], e["weight"], e["hard"]])
    with (out_dir / "theme_ah_links.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hk_code", "a_code", "name"])
        for row in ah_links:
            w.writerow(list(row))
    (out_dir / "theme_communities.json").write_text(
        json.dumps(communities, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "theme_graph_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def print_ascii(nodes, edges, ah_links, themes_by_node):
    print("\n" + "=" * 66)
    print("港股题材小宇宙 —— 关联图")
    print("=" * 66)
    for theme, meta in cfg.THEMES.items():
        members = [n for n in nodes if n["theme"] == theme]
        seeds = {c for c in cfg.get_seed_stocks(theme)}
        print(f"\n[{meta['name']}] {len(members)} 只")
        for n in sorted(members, key=lambda x: (not x["is_core"], x["code"])):
            mark = "★" if n["code"] in seeds else " "
            print(f"   {mark} {n['code']} {n['name']}")
    print("\n[A-H 双重上市桥梁 E1]")
    for hk, a, name in ah_links:
        print(f"   {hk} ({name}) ⟷ A:{a}")
    by_type = defaultdict(int)
    for e in edges:
        by_type[e["edge_type"]] += 1
    print("\n[边统计]")
    for t in cfg.EDGE_TYPES:
        if by_type.get(t):
            tag = "硬" if t in cfg.HARD_EDGE_TYPES else "软"
            print(f"   {t:16s} {by_type[t]:4d}  ({tag})")


# ------------------------------------------------------------------ 主流程
def build_graph(with_correlation=False, southbound_topk=10, threshold=0.5,
                period="2y", out_dir=DEFAULT_OUT_DIR):
    mapping = cfg.HK_THEMES_SECTOR_MAPPING
    universe = set(cfg.get_all_stocks())

    nodes = [{"code": c, "name": m["name"], "theme": m["theme"],
              "sector": m["sector"], "is_core": m["is_core"], "market": "HK"}
             for c, m in mapping.items()]
    nodes.sort(key=lambda n: (n["theme"], not n["is_core"], n["code"]))

    southbound_top = load_southbound_top(SOUTHBOUND_CSV, universe, southbound_topk)
    edges = build_edges(southbound_top)
    ah_links = cfg.get_ah_links()

    communities = {"source": [], "partition": {}, "cross_check": None}
    if southbound_top:
        communities["source"].append("E5_SOUTHBOUND")
    corr_edges = []
    if with_correlation:
        print("📈 计算 E6 价格相关（需网络）...")
        corr_edges, partition, modularity = build_correlation_edges(
            universe, period=period, threshold=threshold)
        edges = edges + [e for e in corr_edges
                         if not any(x["u"] == e["u"] and x["v"] == e["v"]
                                    and x["edge_type"] == "E6_CORR" for x in edges)]
        communities["source"].append("E6_CORR")
        communities["modularity"] = modularity
        communities["cross_check"] = cross_check(partition, mapping)
        communities["partition"] = {k: int(v) for k, v in partition.items()}
    else:
        # 离线：只用 E5 经验边做社区对照（避免用结构边 E2 循环论证）
        g = nx.Graph()
        g.add_nodes_from(universe)
        for e in edges:
            if e["edge_type"] == "E5_SOUTHBOUND":
                g.add_edge(e["u"], e["v"], weight=e["weight"])
        if g.number_of_edges() > 0:
            from ml_services import stock_network_analysis as sna
            partition, modularity = sna.detect_communities(g)
            communities["modularity"] = modularity
            communities["cross_check"] = cross_check(partition, mapping)
            communities["partition"] = {k: int(v) for k, v in partition.items()}
        else:
            communities["cross_check"] = None

    summary = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "edges_by_type": {t: sum(1 for e in edges if e["edge_type"] == t)
                          for t in cfg.EDGE_TYPES},
        "themes": {t: len(cfg.get_theme_stocks(t)) for t in cfg.THEMES},
        "n_ah_links": len(ah_links),
        "southbound_top": southbound_top,
        "community_sources": communities["source"],
        "n_eff_estimate": {t: estimate_n_eff(edges, mapping, t) for t in cfg.THEMES},
        "filter": cfg.HK_THEMES_FILTER,
        "note": "N_eff ≪ N（同题材相关）；E3/E4 为人工软边；社区对照仅用经验边。",
    }

    write_outputs(Path(out_dir), nodes, edges, ah_links, communities, summary)
    print_ascii(nodes, edges, ah_links, mapping)
    print(f"\n📁 输出目录: {out_dir}")
    print(f"   节点 {summary['n_nodes']} · 边 {summary['n_edges']} · A-H 对 {summary['n_ah_links']}")
    cc = communities.get("cross_check")
    if cc:
        print(f"   社区对照：社区数={cc['n_communities']} 纯度={cc['purity']} ARI={cc['ari']}")
    else:
        print("   社区对照：无经验边（离线且无南向），跳过（避免循环论证）")
    return summary


def main():
    ap = argparse.ArgumentParser(description="港股题材小宇宙建图器")
    ap.add_argument("--with-correlation", action="store_true", help="加 E6 价格相关（需网络）")
    ap.add_argument("--southbound-topk", type=int, default=10, help="E5 取南向持有比例前 K")
    ap.add_argument("--threshold", type=float, default=0.5, help="E6 相关阈值")
    ap.add_argument("--period", default="2y", help="E6 行情窗口")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    problems = cfg.validate()
    if problems:
        print("❌ 配置自检失败：")
        for p in problems:
            print("   - " + p)
        return 1
    build_graph(with_correlation=args.with_correlation,
                southbound_topk=args.southbound_topk,
                threshold=args.threshold, period=args.period,
                out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
