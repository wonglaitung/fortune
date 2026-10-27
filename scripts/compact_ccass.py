#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 CCASS 原始 CSV 压缩成「ID + 股数 + 占比」+ 参与者字典，供长期归档。

原始目录保留（本地）；本脚本输出到 --dst，供打 tar.gz 提交到私有数据仓。
- 每股：participant_id, shares, pct（去掉 name/address，体积降 5–10 倍）
- 每日一份 _participants.csv（ID → 名称/地址），便于还原
"""
import argparse
import glob
import os

import pandas as pd


def compact_date(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    parts = {}
    n = 0
    for f in sorted(glob.glob(os.path.join(src_dir, '*.csv'))):
        b = os.path.basename(f)
        if b.startswith('_'):
            continue
        df = pd.read_csv(f, dtype={'participant_id': str})
        for _, r in df.iterrows():
            pid = r.get('participant_id')
            if isinstance(pid, str) and pid not in parts:
                parts[pid] = (r.get('name'), r.get('address'))
        cols = ['participant_id', 'shares', 'pct']
        out = df.reindex(columns=cols)
        out.to_csv(os.path.join(dst_dir, b), index=False)
        n += 1
    if parts:
        pd.DataFrame(
            [(k, v[0], v[1]) for k, v in sorted(parts.items())],
            columns=['participant_id', 'name', 'address'],
        ).to_csv(os.path.join(dst_dir, '_participants.csv'), index=False)
    return n, len(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/ccass')
    ap.add_argument('--dst', required=True)
    args = ap.parse_args()

    dates = sorted(d for d in os.listdir(args.src)
                   if d.isdigit() and os.path.isdir(os.path.join(args.src, d)))
    total = 0
    for d in dates:
        n, p = compact_date(os.path.join(args.src, d), os.path.join(args.dst, d))
        total += n
        print(f'{d}: {n} 只，参与者 {p}')
    print(f'完成：{len(dates)} 个日期，{total} 个文件 → {args.dst}')


if __name__ == '__main__':
    main()
