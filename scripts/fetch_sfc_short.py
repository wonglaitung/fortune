#!/usr/bin/env python3
"""SFC 聚合卖空持仓周报 增量刷新（D20 生产纳入配套）。

源（SFC 每周发布，自 2012-09-07 起）：
  https://www.sfc.hk/-/media/EN/pdf/spr/YYYY/MM/DD/Short_Position_Reporting_Aggregated_Data_YYYYMMDD.csv
列：Date(DD/MM/YYYY), Stock Code, Stock Name,
    Aggregated Reportable Short Positions (Shares),
    Aggregated Reportable Short Positions (HK$)

归一化本地 schema 并追加到 data/sfc_short/sfc_short_long.csv，刷新 manifest.json：
  date(YYYYMMDD 整数), stock_code(去前导零), stock_name, short_shares, short_hk$

用法：
  python3 scripts/fetch_sfc_short.py            # 增量：只抓晚于本地最新周的新周
  python3 scripts/fetch_sfc_short.py --full     # 全量回填（2012-09-07 起，仅首次/修复）
  python3 scripts/fetch_sfc_short.py --dry      # 只列将要抓的周，不写盘
"""
import argparse
import datetime as dt
import io
import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFC_CSV = os.path.join(BASE, 'data', 'sfc_short', 'sfc_short_long.csv')
MANIFEST = os.path.join(BASE, 'data', 'sfc_short', 'manifest.json')
START_DEFAULT = dt.date(2012, 9, 7)  # SFC 聚合卖空发布起始周


def sfc_url(d: dt.date) -> str:
    ymd = d.strftime('%Y%m%d')
    return (f"https://www.sfc.hk/-/media/EN/pdf/spr/{d.year}/{d.month:02d}/"
            f"{d.day:02d}/Short_Position_Reporting_Aggregated_Data_{ymd}.csv")


def fetch_week(d: dt.date, timeout: int = 30):
    """抓取单周 CSV；返回 (DataFrame|None, err|None)。"""
    import requests

    url = sfc_url(d)
    last_err = None
    for _ in range(2):  # 简易重试
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=timeout)
        except Exception as e:  # 网络异常（含挂死风险已由 timeout 兜底）
            last_err = f"req err {e}"
            continue
        if r.status_code != 200:
            last_err = f"HTTP {r.status_code}"
            if r.status_code == 404:
                return None, last_err
            continue
        # 未发布/软 404：SFC 对不存在的周返回 200 + HTML 页面，按未发布静默跳过
        head = r.text.lstrip()[:200].lower()
        if not head.startswith('date,') and ('<html' in head or '<!doctype' in head):
            return None, None
        try:
            df = pd.read_csv(io.StringIO(r.text))
            df = df.rename(columns={
                'Date': 'date',
                'Stock Code': 'stock_code',
                'Stock Name': 'stock_name',
                'Aggregated Reportable Short Positions (Shares)': 'short_shares',
                'Aggregated Reportable Short Positions (HK$)': 'short_hk$',
            })
            # Date DD/MM/YYYY -> YYYYMMDD 整数
            df['date'] = pd.to_datetime(
                df['date'], format='%d/%m/%Y', errors='coerce').dt.strftime('%Y%m%d')
            df['date'] = df['date'].astype('Int64')
            # 股票码去前导零（'00700'->'700'，'1'->'1'），与 load_sfc_long 一致
            df['stock_code'] = df['stock_code'].astype(str).str.lstrip('0').replace('', '0')
            df['short_shares'] = pd.to_numeric(df['short_shares'], errors='coerce')
            df['short_hk$'] = pd.to_numeric(
                df['short_hk$'].astype(str).str.replace(',', ''), errors='coerce')
            df = df[['date', 'stock_code', 'stock_name', 'short_shares', 'short_hk$']]
            df = df.dropna(subset=['date', 'stock_code'])
            return df, None
        except Exception as e:
            last_err = f"parse err {e}"
            return None, last_err
    return None, last_err


def load_existing():
    if os.path.exists(SFC_CSV):
        df = pd.read_csv(SFC_CSV, dtype={'stock_code': str}, low_memory=False)
        df['date'] = df['date'].astype('Int64')
        return df
    return pd.DataFrame(columns=['date', 'stock_code', 'stock_name', 'short_shares', 'short_hk$'])


def candidate_fridays(start: dt.date, end: dt.date):
    """生成 start..end 的每周五（SFC 报告日≈每周最后交易日）。"""
    d = start
    while d.weekday() != 4:  # 4=Friday
        d += dt.timedelta(days=1)
    out = []
    while d <= end:
        out.append(d)
        d += dt.timedelta(days=7)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', action='store_true', help='从 2012-09-07 全量回填（仅首次/修复）')
    ap.add_argument('--dry', action='store_true', help='只列待抓周，不写盘')
    ap.add_argument('--start', default=None, help='起始周 YYYYMMDD（覆盖默认）')
    args = ap.parse_args()

    existing = load_existing()
    latest_d = (int(existing['date'].max()) if not existing.empty else None)
    latest_date = (dt.datetime.strptime(str(latest_d), '%Y%m%d').date()
                   if latest_d else None)

    if args.full or latest_date is None:
        start = (dt.datetime.strptime(args.start, '%Y%m%d').date()
                 if args.start else START_DEFAULT)
    else:
        start = latest_date + dt.timedelta(days=7)
    end = dt.date.today() + dt.timedelta(days=7)  # 允许抓到刚发布的最近一周

    fridays = candidate_fridays(start, end)
    have = set(existing['date'].tolist()) if not existing.empty else set()
    new_frames, new_dates = [], []
    for f in fridays:
        got = False
        for off in (0, -1, -2, -3, 1, 2, 3):  # 假期可能令报告日偏移 ±几天
            d = f + dt.timedelta(days=off)
            if latest_date and d <= latest_date:
                continue
            ymd = int(d.strftime('%Y%m%d'))
            if ymd in have:
                got = True
                break
            df, err = fetch_week(d)
            if df is not None and not df.empty:
                new_frames.append(df)
                new_dates.append(ymd)
                have.add(ymd)
                got = True
                print(f"  ✅ 抓取 {d} ({len(df)} 行)")
                break
            if err and 'HTTP 404' not in err:
                print(f"  ⚠️ {d}: {err}")
        if not got and not args.dry:
            pass  # 该周暂无数据，跳过

    if not new_frames:
        print("无新周需抓取（SFC 已最新）")
        return
    if args.dry:
        print(f"dry: 将抓 {len(new_dates)} 周: {new_dates[:8]}"
              f"{'...' if len(new_dates) > 8 else ''}")
        return

    combined = pd.concat([existing] + new_frames, ignore_index=True)
    combined['date'] = combined['date'].astype('Int64')
    combined = combined.drop_duplicates(subset=['date', 'stock_code'], keep='last')
    combined = combined.sort_values(['date', 'stock_code']).reset_index(drop=True)
    os.makedirs(os.path.dirname(SFC_CSV), exist_ok=True)
    combined.to_csv(SFC_CSV, index=False)

    weeks = sorted(int(w) for w in combined['date'].dropna().unique().tolist())
    manifest = {'weeks': len(weeks), 'rows': int(len(combined)),
                'manifest': [[str(w), 'OK'] for w in weeks]}
    with open(MANIFEST, 'w') as fh:
        json.dump(manifest, fh, indent=2)
    print(f"✅ 写入 {SFC_CSV}：{len(combined)} 行 / {len(weeks)} 周；"
          f"本次新增 {len(new_dates)} 周（{min(new_dates)}..{max(new_dates)}）")


if __name__ == '__main__':
    main()
