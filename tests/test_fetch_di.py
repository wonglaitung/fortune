from scripts.fetch_di import _num, _pair


def test_num():
    assert _num('HKD 425.2400') == 425.24
    assert _num('17,277,864(L)') == 17277864


def test_pair():
    assert _pair('58,494(L) 0(S)') == (58494.0, 0.0)
    assert _pair('1,383(L)') == (1383.0, None)
