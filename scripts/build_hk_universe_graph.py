#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
港股小宇宙（`hk_universe_config.py`，种子 + 关系网络）—— 关联建图 + 社区对照

用**价格相关（经验边）**验证这四张"围绕种子的关系网"是否**有价格实体**
（Louvain 社区 ⇄ 种子网络标签）。

方法学红线（同 D51）：
- **结构边 E_CORE（种子↔成员、组内）是按定义连的** ⇒ 用它跑 Louvain 会**循环**（必然复现网络）；
  ⇒ **社区对照只用经验边**（E5 南向共持 / E6 价格相关）。
- **E_AH**（A-H 双重上市）是**跨市场**硬边，单列输出（桥梁），不进 HK-HK 社区。
- 离线无经验边 ⇒ **跳过对照**（不装作有结果）。

输出（默认 `data/hk_universe/`）：
    universe_nodes.csv / universe_edges.csv / universe_ah_links.csv
    universe_communities.json / universe_summary.json

用法：
    python3 scripts/build_hk_universe_graph.py                    # 离线（E_CORE + E5）
    python3 scripts/build_hk_universe_graph.py --with-correlation # 加 E6 价格相关（需网络）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hk_universe_config as cfg  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "data" / "hk_universe"
SOUTHBOUND_CSV = ROOT / "data" / "southbound_holdings" / "southbound_long.csv"

# 宇宙内 A-H 双重上市（H ↔ A）
AH_PAIRS = {
    "0939.HK": "601939", "1288.HK": "601288", "1398.HK": "601398",
    "3968.HK": "600036", "1658.HK": "601658", "3328.HK": "601328",
    "0998.HK": "601998", "1988.HK": "600016", "3988.HK": "601988",
    "0981.HK": "688981", "1347.HK": "688347", "1385.HK": "688385",
    "3898.HK": "688187",
}


def network_of():
    """code → 所属种子网（种子本身 → 自己）。"""
    out = {}
    for core in cfg.HK_CORE_HOLDINGS:
        out[core] = core
    for core, groups in cfg.HK_RELATION_GROUPS.items():
        for codes in groups.values():
            for c in codes:
                out[c] = core
    return out


def load_southbound_top(csv_path: Path, universe: set, top_k: int = 12):
    if not csv_path.exists():
        return []
    latest, rows = None, []
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
            if latest is None or d > latest:
                latest, rows = d, [(code, ratio)]
            elif d == latest:
                rows.append((code, ratio))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def build_structural_edges():
    """E_CORE：种子↔成员（星）+ 组内（团）。按定义 ⇒ 结构边。"""
    edges = []
    for core, groups in cfg.HK_RELATION_GROUPS.items():
        for codes in groups.values():
            for c in codes:
                edges.append((core, c, "E_CORE_STAR", 1.0))
            for u, v in combinations(sorted(codes), 2):
                edges.append((u, v, "E_CORE_GROUP", 1.0))
    return edges


def build_e5(universe, top_k):
    top = load_southbound_top(SOUTHBOUND_CSV, universe, top_k)
    return [(u, v, "E5_SOUTHBOUND", 0.7) for (u, _), (v, _) in combinations(sorted(top), 2)], top


def build_e6(universe, period, threshold):
    from ml_services import stock_network_analysis as sna
    data = sna.fetch_all_stock_data(sorted(universe), period=period)
    if len(data) < 5:
        print("  ⚠️ 行情不足，跳过 E6")
        return [], {}, 0.0
    rets = sna.build_returns_dataframe(data)
    pearson, _ = sna.compute_correlation_matrices(rets)
    g = sna.build_threshold_network(pearson, list(rets.columns), threshold=threshold)
    part, mod = sna.detect_communities(g)
    edges = [(u, v, "E6_CORR", round(float(d.get("weight", 1.0)), 4)) for u, v, d in g.edges(data=True)]
    return edges, part, mod


def cross_check(partition, netmap, names):
    if not partition:
        return None
    buckets = defaultdict(lambda: defaultdict(int))
    for node, comm in partition.items():
        net = netmap.get(node)
        if net:
            buckets[comm][net] += 1
    total = sum(sum(v.values()) for v in buckets.values())
    hit = sum(max(v.values()) for v in buckets.values())
    out = {"n_communities": len(buckets), "purity": round(hit / total, 4) if total else None,
           "contingency": {str(c): {names.get(k, k): n for k, n in v.items()} for c, v in sorted(buckets.items())}}
    labels = [netmap[n] for n in partition if n in netmap]
    preds = [partition[n] for n in partition if n in netmap]
    try:
        import warnings
        from sklearn.metrics import adjusted_rand_score
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["ari"] = round(float(adjusted_rand_score(labels, preds)), 4)
    except Exception:
        out["ari"] = None
    return out


def write_outputs(out_dir, nodes, edges, ah_links, communities, summary):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "universe_nodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["code", "name", "network", "sector", "is_core"])
        for n in nodes:
            w.writerow([n["code"], n["name"], n["network"], n["sector"], n["is_core"]])
    with (out_dir / "universe_edges.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["u", "v", "edge_type", "weight"])
        for e in edges:
            w.writerow(list(e))
    with (out_dir / "universe_ah_links.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["hk_code", "a_code", "name"])
        for hk, a in ah_links:
            w.writerow([hk, a, cfg.HK_UNIVERSE.get(hk, "")])
    (out_dir / "universe_communities.json").write_text(json.dumps(communities, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "universe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_graph(with_correlation=False, threshold=0.5, period="2y", southbound_topk=12, out_dir=DEFAULT_OUT_DIR):
    netmap = network_of()
    names = {core: cfg.HK_CORE_HOLDINGS[core] for core in cfg.HK_CORE_HOLDINGS}
    universe = set(cfg.HK_UNIVERSE)

    nodes = [{"code": c, "name": cfg.HK_UNIVERSE[c], "network": netmap.get(c, "?"),
              "sector": cfg.HK_UNIVERSE_SECTOR_MAPPING.get(c, {}).get("sector", "?"),
              "is_core": cfg.is_core_holding(c)} for c in cfg.HK_UNIVERSE]
    nodes.sort(key=lambda n: (n["network"], not n["is_core"], n["code"]))

    edges = build_structural_edges()
    e5, top = build_e5(universe, southbound_topk)
    edges += e5
    ah_links = [(hk, a) for hk, a in AH_PAIRS.items() if hk in universe]

    communities = {"source": (["E5_SOUTHBOUND"] if top else []), "partition": {}, "cross_check": None}
    if with_correlation:
        print("📈 计算 E6 价格相关（需网络）...")
        e6, part, mod = build_e6(universe, period, threshold)
        edges += e6
        communities["source"].append("E6_CORR")
        communities["modularity"] = mod
        communities["partition"] = {k: int(v) for k, v in part.items()}
        communities["cross_check"] = cross_check(part, netmap, names)
    else:
        if e5:
            from ml_services import stock_network_analysis as sna
            g = nx.Graph(); g.add_nodes_from(universe)
            for u, v, _, w in e5:
                g.add_edge(u, v, weight=w)
            part, mod = sna.detect_communities(g)
            communities["modularity"] = mod
            communities["partition"] = {k: int(v) for k, v in part.items()}
            communities["cross_check"] = cross_check(part, netmap, names)

    summary = {
        "n_nodes": len(nodes), "n_edges": len(edges),
        "edges_by_type": {t: sum(1 for e in edges if e[2] == t)
                          for t in ("E_CORE_STAR", "E_CORE_GROUP", "E5_SOUTHBOUND", "E6_CORR")},
        "networks": {c: 1 + sum(len(v) for v in cfg.HK_RELATION_GROUPS[c].values()) for c in cfg.HK_CORE_HOLDINGS},
        "n_ah_links": len(ah_links),
        "community_sources": communities["source"],
        "note": "结构边(E_CORE)按定义⇒仅供查表；社区对照只用经验边(E5/E6)。",
    }
    write_outputs(Path(out_dir), nodes, edges, ah_links, communities, summary)
    _print(nodes, edges, ah_links, communities, names)
    print(f"\n📁 {out_dir}")
    return summary


def _print(nodes, edges, ah_links, communities, names):
    print("=" * 70)
    print("港股小宇宙 —— 关联图（种子 + 关系网络）")
    print("=" * 70)
    by_net = defaultdict(list)
    for n in nodes:
        by_net[n["network"]].append(n)
    for core, members in by_net.items():
        print(f"\n[{names.get(core, core)}] {len(members)} 只")
        for n in members:
            print(f"   {'★' if n['is_core'] else ' '} {n['code']} {n['name']}")
    print("\n[A-H 双重上市桥梁]")
    for hk, a in ah_links:
        print(f"   {hk} ({cfg.HK_UNIVERSE.get(hk, '')}) ⟷ A:{a}")
    bt = defaultdict(int)
    for e in edges:
        bt[e[2]] += 1
    print("\n[边统计]  " + " | ".join(f"{k}={bt.get(k, 0)}" for k in
                                       ("E_CORE_STAR", "E_CORE_GROUP", "E5_SOUTHBOUND", "E6_CORR")))
    cc = communities.get("cross_check")
    if cc:
        print(f"[社区对照] 社区 {cc['n_communities']} · 纯度 {cc['purity']} · ARI {cc['ari']}")
    else:
        print("[社区对照] 无经验边（离线且无南向）⇒ 跳过（避免循环论证）")


def main():
    ap = argparse.ArgumentParser(description="港股小宇宙建图器")
    ap.add_argument("--with-correlation", action="store_true")
    ap.add_argument("--southbound-topk", type=int, default=12)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--period", default="2y")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()
    probs = cfg.validate()
    if probs:
        print("❌ 配置自检失败：")
        for p in probs:
            print("   - " + p)
        return 1
    build_graph(with_correlation=args.with_correlation, threshold=args.threshold,
                period=args.period, southbound_topk=args.southbound_topk, out_dir=Path(args.out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
