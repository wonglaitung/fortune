from scripts.extract_event_magnitude import extract_cb, extract_lockup, extract_placement, extract_privatisation


def test_placement():
    t = ('placing, through the Placing Agents, 241,418,625 Placing Shares at a price of HK$10.65 '
         'per Placing Share. The Placing Price represents a discount of approximately 4.91% to the '
         'Closing Price of HK$11.20. The gross proceeds of the Placing will be approximately HK$2.57 billion.')
    m = extract_placement(t)
    assert m['new_shares'] == 241_418_625
    assert m['issue_price'] == 10.65
    assert abs(m['discount_pct'] - 0.0491) < 1e-9
    assert m['proceeds'] == 2.57e9


def test_cb():
    t = ('the Bonds, to be issued in the aggregate principal amount of U.S.$3,500,000,000, '
         'initial Conversion Price of HK$56.00, convertible into approximately 625,203,614 Conversion Shares, '
         'representing approximately 3.43% of the existing issued shares.')
    m = extract_cb(t)
    assert m['cb_principal'] == 3.5e9
    assert m['conversion_price'] == 56.0
    assert m['conversion_shares'] == 625_203_614
    assert abs(m['cb_dilution_pct'] - 0.0343) < 1e-9


def test_privatisation():
    t = 'the Scheme Consideration of HK$155.00 in cash for every Scheme Share, a premium of approximately 30.3%'
    m = extract_privatisation(t)
    assert m['offer_price'] == 155.0
    assert abs(m['offer_premium_pct'] - 0.303) < 1e-9


def test_lockup():
    assert extract_lockup('for a further period of 365 days')['lockup_days'] == 365
