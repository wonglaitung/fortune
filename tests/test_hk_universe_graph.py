"""港股小宇宙建图器（scripts/build_hk_universe_graph.py）—— D53。"""

import csv

import hk_universe_config as cfg
from scripts.build_hk_universe_graph import build_graph, network_of


def test_network_of_covers_universe():
    nm = network_of()
    assert set(nm) == set(cfg.HK_UNIVERSE)
    for core in cfg.HK_CORE_HOLDINGS:
        assert nm[core] == core


def test_offline_graph_structure(tmp_path):
    summary = build_graph(with_correlation=False, out_dir=tmp_path)
    assert summary["n_nodes"] == len(cfg.HK_UNIVERSE)
    assert summary["n_edges"] > 0
    assert summary["n_ah_links"] > 0
    for name in ("universe_nodes.csv", "universe_edges.csv",
                 "universe_ah_links.csv", "universe_communities.json",
                 "universe_summary.json"):
        assert (tmp_path / name).exists(), name


def test_edges_valid(tmp_path):
    build_graph(with_correlation=False, out_dir=tmp_path)
    universe = set(cfg.HK_UNIVERSE)
    with (tmp_path / "universe_edges.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        assert r["u"] in universe and r["v"] in universe and r["u"] != r["v"]
        assert r["edge_type"] in {"E_CORE_STAR", "E_CORE_GROUP", "E5_SOUTHBOUND", "E6_CORR"}
