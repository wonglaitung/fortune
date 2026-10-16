"""⓪ 同一性闸可执行检查（scripts/parity_check.py）。"""

import pickle

import scripts.parity_check as pc


def _reset():
    pc._results.clear()


def _make_pkl(tmp_path, **md):
    p = tmp_path / "m.pkl"
    base = {"feature_columns": ["a", "b"], "community_ids": [0, 1], "horizon": 20, "learner": "catboost"}
    base.update(md)
    with p.open("wb") as f:
        pickle.dump(base, f)
    return p


def test_missing_model_fails(tmp_path):
    _reset()
    pc.check_artifact(tmp_path / "nope.pkl")
    assert any(s == pc.FAIL for s, _, _ in pc._results)


def test_community_feature_without_ids_fails(tmp_path):
    _reset()
    p = _make_pkl(tmp_path, feature_columns=["net_comm_0", "x"], community_ids=None)
    pc.check_artifact(p)
    assert any(s == pc.FAIL and "社区" in n for s, n, _ in pc._results)


def test_consistent_model_passes(tmp_path):
    _reset()
    p = _make_pkl(tmp_path, feature_columns=["net_comm_0", "hibor_x"], community_ids=[0, 1])
    pc.check_artifact(p)
    assert not any(s == pc.FAIL for s, _, _ in pc._results)


def test_keyword_counts():
    cols = ["f1_a", "x_f1_b", "hibor_z", "net_centrality_c", "plain"]
    assert pc._count(cols, r"^f1_|_f1_") == 2
    assert pc._count(cols, r"hibor") == 1
    assert pc._count(cols, r"^net_") == 1
