#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取更大港股池（Part B）到 data/hk_universe_cache/

腾讯接口正常（需 5 位代码格式，如 '00700'），此前失败是瞬时 DNS + 格式问题。
列表：约 150 只港股大中盘（5 位代码）。
用法：python3 ml_services/fetch_hk_universe.py
"""

import os
import sys
import time
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_services.tencent_finance import get_hk_stock_data_tencent

CACHE_DIR = 'data/hk_universe_cache'

# 约 150 只港股（5 位代码）。含原有自选池 + 大中盘补充。
UNIVERSE = [
    # 银行/保险/券商
    '00005','00011','00939','01398','01288','03988','03968','02388','02628','02318',
    '02601','01339','01299','02378','00388','06030','06886','03908','06099','01776',
    # 科技/互联网/软件
    '00700','09988','03690','01810','09618','09999','09888','01024','02015','09868',
    '09992','00788','00772','01024','00268','00285','00981','01347','00763','03759',
    # 能源/公用/资源
    '00883','01088','01171','00386','00857','00902','00916','00956','00991','01193',
    '00688','01114','01919','01138','02866','02689','00966','01313','02899','01208',
    # 汽车/工业/制造
    '01211','00175','02333','02015','04899','00489','03808','01616','02382','01316',
    # 电信/基建
    '00941','00728','00762','00002','00006','01093','01099','01800','00390','01133',
    # 地产/物业/建筑
    '00016','00012','00101','00004','01997','01109','00688','02007','03333','00813',
    '02333','01119','01972','06098','06049','09909',
    # 消费/零售/品牌
    '00291','00151','02319','06186','02331','02020','01876','00700','09992','09998',
    # 医药/生物
    '02269','01177','01093','01099','02186','01801','06185','09995','01530','01890',
    # 传媒/教育/娱乐
    '01810','00960','03800','00981','00700','01123','01060','02020',
    # 物流/航空/航运
    '00293','00694','00916','01919','01138','00316','02343',
    # 综合/其他蓝筹
    '00001','00003','00017','00019','00066','00083','00144','00322','00027','00012',
]

def fetch(code):
    df = get_hk_stock_data_tencent(code, period_days=1460)
    return df


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    ok, fail = 0, []
    for i, code in enumerate(dict.fromkeys(UNIVERSE)):
        path = os.path.join(CACHE_DIR, f"{code}.pkl")
        if os.path.exists(path):
            ok += 1
            continue
        try:
            df = fetch(code)
            if df is None or len(df) < 300:
                fail.append((code, 'short/None'))
                continue
            df.to_pickle(path)
            ok += 1
        except Exception as e:
            fail.append((code, str(e)[:60]))
        time.sleep(0.15)
        if (i + 1) % 25 == 0:
            print(f"  ... {i+1}/{len(dict.fromkeys(UNIVERSE))}  ok={ok} fail={len(fail)}")
    print(f"完成: 成功 {ok}，失败 {len(fail)}")
    for c, e in fail[:20]:
        print(f"  {c}: {e}")


if __name__ == '__main__':
    main()