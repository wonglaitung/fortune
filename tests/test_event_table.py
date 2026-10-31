from scripts.build_event_table import classify


def test_profit_alerts_split():
    assert classify('PROFIT WARNING ANNOUNCEMENT')[:2] == ('PROFIT_WARNING', 'high')
    assert classify('ANNOUNCEMENT ON POSITIVE PROFIT ALERT FOR 2017')[:2] == ('POSITIVE_PROFIT_ALERT', 'high')


def test_placement_company_vs_shareholder():
    assert classify('PLACING AND ISSUE OF SHARES UNDER GENERAL MANDATE')[0] == 'PLACEMENT_COMPANY'
    assert classify('PLACING OF SHARES BY THE CONTROLLING SHAREHOLDER')[0] == 'PLACEMENT_SHAREHOLDER'
    assert classify('Issue of New Shares Pursuant to Share Award Scheme') is None


def test_convertible_issuance_vs_conversion():
    assert classify('PROPOSED ISSUE OF US$200 MILLION ZERO COUPON CONVERTIBLE BONDS DUE 2022')[0] == 'CB_ISSUANCE'
    assert classify('FULL CONVERSION OF CONVERTIBLE BONDS DUE 2018')[0] == 'CB_CONVERSION'
    assert classify('JONATHAN SYMONDS CBE JOINS HSBC GROUP') is None


def test_rights_review():
    typ, conf, direction = classify('PROPOSED RIGHTS ISSUE')
    assert typ == 'RIGHTS_ISSUE' and conf == 'review' and direction == '-'


def test_stage_flags():
    from scripts.build_event_table import stage
    assert stage('COMPLETION OF PLACING OF NEW SHARES') == 'completion'
    assert stage('MONTHLY UPDATE ON PRIVATISATION') == 'update'
    assert stage('PROPOSED PLACING OF NEW SHARES') == 'announcement'
