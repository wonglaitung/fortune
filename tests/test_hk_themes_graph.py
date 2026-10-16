"""港股题材建图器（scripts/build_hk_themes_graph.py）—— D51。"""

import csv

import hk_themes_config as cfg
from scripts.build_hk_themes_graph import build_graph


def test_offline_graph_structure(tmp_path):
    summary = build_graph(with_correlation=False, out_dir=tmp_path)

    assert summary["n_nodes"] == len(cfg.get_all_stocks())
    assert summary["n_edges"] > 0
    assert summary["n_ah_links"] == len(cfg.get_ah_links())

    for name in ("theme_graph_nodes.csv", "theme_graph_edges.csv",
                 "theme_ah_links.csv", "theme_communities.json",
                 "theme_graph_summary.json"):
        assert (tmp_path / name).exists(), name


def test_edges_have_valid_type_and_nodes(tmp_path):
    build_graph(with_correlation=False, out_dir=tmp_path)
    universe = set(cfg.get_all_stocks())
    with (tmp_path / "theme_graph_edges.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        assert r["edge_type"] in cfg.EDGE_TYPES
        assert r["u"] in universe and r["v"] in universe
        assert r["u"] != r["v"]


def test_e1_links_match_config(tmp_path):
    build_graph(with_correlation=False, out_dir=tmp_path)
    with (tmp_path / "theme_ah_links.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {(r["hk_code"], r["a_code"]) for r in rows} == \
        {(hk, a) for hk, a, _ in cfg.get_ah_links()}
