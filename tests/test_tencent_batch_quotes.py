import pandas as pd

import data_services.tencent_finance as tf


def _line(code, name, price, prev, pct):
    f = ['0'] * 78
    f[0], f[1], f[2], f[3], f[4] = '100', name, code, str(price), str(prev)
    f[5], f[6] = '0', '1000'
    f[30] = '2026/10/28 16:08:30'
    f[31] = f'{price - prev:.3f}'
    f[32] = str(pct)
    f[37] = '12345.6'
    return f'v_hk{code}="{"~".join(f)}";'


class _Resp:
    def __init__(self, text):
        self.content = text.encode('gbk')

    def raise_for_status(self):
        return None


def test_batch_quotes_parse_and_chunk(monkeypatch):
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        if 'hk00700' in url:
            return _Resp('\n'.join([
                _line('00700', '腾讯控股', 387.6, 385.2, 0.62),
                _line('00005', '汇丰控股', 135.6, 134.3, 0.97),
            ]))
        return _Resp(_line('02318', '中国平安', 48.38, 48.68, -0.62))

    monkeypatch.setattr(tf.requests, 'get', fake_get)
    df = tf.get_hk_batch_quotes(['0700.HK', '00005', '02318'], chunk_size=2)

    assert len(calls) == 2
    assert list(df['code']) == ['00700', '00005', '02318']
    assert df.loc[df['code'] == '00700', 'pct'].iloc[0] == 0.62
    assert df.loc[df['code'] == '00700', 'price'].iloc[0] == 387.6
    assert df.loc[df['code'] == '02318', 'pct'].iloc[0] == -0.62


def test_batch_quotes_skips_malformed(monkeypatch):
    def fake_get(url, headers=None, timeout=None):
        return _Resp('v_hk00700="short";\n' + _line('00700', 'X', 10.0, 9.0, 11.1))

    monkeypatch.setattr(tf.requests, 'get', fake_get)
    df = tf.get_hk_batch_quotes(['00700'])
    assert len(df) == 1 and df.iloc[0]['pct'] == 11.1
