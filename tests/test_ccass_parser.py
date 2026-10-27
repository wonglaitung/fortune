from scripts.fetch_ccass import parse_rows


def _html(rows):
    head = ('<tr><th>Participant ID</th><th>Name</th><th>Address</th>'
            '<th>Shareholding</th><th>%</th></tr>')
    body = head + ''.join(rows)
    return f'<html><body><table>{body}</table></body></html>'


def test_parse_rows_basic():
    row = ('<tr><td>Participant ID: C00019</td>'
           '<td>Name of CCASS Participant (* for Consenting Investor Participants ): HSBC</td>'
           '<td>Address: KOWLOON</td>'
           '<td>Shareholding: 1,234,567</td>'
           '<td>% of the total number of Issued Shares/ Warrants/ Units: 12.34%</td></tr>')
    df = parse_rows(_html([row] * 12))
    assert len(df) == 12
    assert df.iloc[0]['participant_id'] == 'C00019'
    assert df.iloc[0]['shares'] == 1234567
    assert df.iloc[0]['pct'] == 12.34


def test_parse_rows_empty_when_no_table():
    assert parse_rows('<html></html>').empty


def test_compact_strips_name_address(tmp_path):
    import pandas as pd
    from scripts.compact_ccass import compact_date

    src = tmp_path / 'src' / '20260101'
    src.mkdir(parents=True)
    pd.DataFrame({
        'participant_id': ['C00019', 'A00003'],
        'name': ['HSBC', 'CSDC'],
        'address': ['KOWLOON', 'BEIJING'],
        'shares': [10, 20],
        'pct': [1.5, 3.0],
    }).to_csv(src / '00700.csv', index=False)

    dst = tmp_path / 'dst' / '20260101'
    n, p = compact_date(str(src), str(dst))
    assert n == 1 and p == 2
    out = pd.read_csv(dst / '00700.csv')
    assert list(out.columns) == ['participant_id', 'shares', 'pct']
    parts = pd.read_csv(dst / '_participants.csv')
    assert set(parts['participant_id']) == {'C00019', 'A00003'}
