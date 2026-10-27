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
