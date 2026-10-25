import inspect
import json

import ml_services.ml_trading_model as mt


def test_override_merges_and_ignores_unknown(tmp_path, monkeypatch):
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'lgbm_best_params.json').write_text(json.dumps({
        '20': {'n_estimators': 123, 'bogus': 1},
        '5': {'num_leaves': 7},
    }))
    monkeypatch.chdir(tmp_path)

    p20 = mt.load_lgbm_params(20)
    assert p20['n_estimators'] == 123
    assert 'bogus' not in p20
    assert p20['num_leaves'] == mt.LGBM_DEFAULT_PARAMS['num_leaves']

    p5 = mt.load_lgbm_params(5)
    assert p5['num_leaves'] == 7
    assert p5['n_estimators'] == mt.LGBM_DEFAULT_PARAMS['n_estimators']


def test_no_hardcoded_params_in_train_path():
    src = inspect.getsource(mt.TradingModel._train_lightgbm)
    assert 'load_lgbm_params(horizon)' in src
    for key in ('learning_rate=', 'num_leaves=', 'colsample_bytree=', "'n_estimators':"):
        assert key not in src, f'{key} 仍写死在 _train_lightgbm'
