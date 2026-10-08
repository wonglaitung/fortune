#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冻结港股原始行情快照（恢复双跑可复现性）

背景（2026-10-07 实证）：个股实时取数的 `stock_df.index[-1]` 直接进入特征缓存键
（ml_trading_model._get_feature_cache_key 含 last_date）。双跑间隔若数据源更新，
末日漂移 → 缓存键全变 → 全量重算 → 共同期 40% 特征列值变（net_* 居首）→
prediction_analysis.csv md5 不同。宏观早有 US_MARKET_SNAPSHOT_DIR（lessons 三.29），
个股原本没有任何冻结，是「双冻结」清单的漏项。

用法：
    python3 scripts/pin_hk_snapshot.py                    # 建立/刷新快照
    python3 scripts/pin_hk_snapshot.py --check            # 校验完整性（缺一即非零退出）
    HK_MARKET_SNAPSHOT_DIR=data/hk_market_snapshot \
        bash scripts/run_walk_forward.sh ...               # 用快照跑（可复现）

双向落盘（lessons「冻结必须双向落盘」）：成功写 .pkl，失败写 .failed 并删除
同名 .pkl —— 只写成功数据等于没冻结，且旧 .pkl 会冒充新快照让 --check 误通过。
"""
import argparse
import os
import pickle
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

DEFAULT_SNAP_DIR = os.path.join(BASE, 'data', 'hk_market_snapshot')
# period_days 上限：取数端 period_days_needed = max(1460, (now-start).days+120)，
# 2019-06 起约 2804 天，留余量取 5000。消费端按 period_days 截尾。
SNAP_PERIOD_DAYS = 5000


def _default_codes():
    from config import TRAINING_STOCKS
    return [c.replace('.HK', '') for c in TRAINING_STOCKS]


def _atomic_save(obj, path):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def do_pin(codes, snap_dir, period_days):
    from data_services.tencent_finance import get_hk_stock_data_tencent
    # 建快照必须走实时取数，防止读到自己（或残留的）快照
    os.environ.pop('HK_MARKET_SNAPSHOT_DIR', None)

    os.makedirs(snap_dir, exist_ok=True)
    ok, nodata, failed = [], [], []
    for code in codes:
        pkl = os.path.join(snap_dir, f"{code}.pkl")
        fail = os.path.join(snap_dir, f"{code}.failed")
        no = os.path.join(snap_dir, f"{code}.nodata")
        try:
            df = get_hk_stock_data_tencent(code, period_days=period_days)
            if df is None or len(df) < 100:
                raise ValueError(f"行数不足: {0 if df is None else len(df)}")
            _atomic_save(df, pkl)
            for p in (fail, no):
                if os.path.exists(p):
                    os.remove(p)
            ok.append((code, len(df), df.index[-1]))
        except ValueError as e:
            # 「该股无数据」是冻结得下的确定结论（如已退市），消费端按跳过处理
            if os.path.exists(pkl):
                os.remove(pkl)
            if os.path.exists(fail):
                os.remove(fail)
            with open(no, 'w') as f:
                f.write(f"{type(e).__name__}: {e}\n")
            nodata.append((code, str(e)[:60]))
        except BaseException as e:
            # 网络/IO 意外：写 failed 并删旧 pkl，让 --check 阻断重试
            if os.path.exists(pkl):
                os.remove(pkl)
            if os.path.exists(no):
                os.remove(no)
            with open(fail, 'w') as f:
                f.write(f"{type(e).__name__}: {e}\n")
            failed.append((code, str(e)[:80]))

    print(f"✅ 快照完成: 有数据 {len(ok)} / 无数据 {len(nodata)} / 意外失败 {len(failed)}"
          f"  -> {os.path.relpath(snap_dir, BASE)}")
    ends = sorted({str(e.date()) for _, _, e in ok})
    if ends:
        print(f"   数据末日分布: {', '.join(ends)}")
    if nodata:
        print(f"⚠️  无数据 {len(nodata)} 只（已冻结为 .nodata，消费端跳过，须人工核对）:")
        for c, m in nodata:
            print(f"   {c}: {m}")
    if failed:
        print(f"❌ 意外失败 {len(failed)} 只（已写 .failed 哨兵，--check 会阻断）:")
        for c, m in failed[:10]:
            print(f"   {c}: {m}")
    print("   复现方式: 前置环境变量（回测必设，生产预测勿设）:")
    print(f"     HK_MARKET_SNAPSHOT_DIR={os.path.relpath(snap_dir, BASE)}")
    print("     US_MARKET_SNAPSHOT_DIR=data/us_market_snapshot")
    print("     GATE_SOURCE_CSV=output/<基线>/prediction_analysis.csv")
    print("   快照模式下未命中会 SystemExit，禁止静默实时取数")
    return 1 if failed else 0


def do_check(codes, snap_dir):
    if not os.path.isdir(snap_dir):
        print(f"❌ 快照目录不存在: {snap_dir}  —— 先运行 pin_hk_snapshot.py")
        return 1
    missing, failed, nodata, ok = [], [], [], []
    for code in codes:
        pkl = os.path.join(snap_dir, f"{code}.pkl")
        if os.path.exists(os.path.join(snap_dir, f"{code}.failed")):
            failed.append(code)
        elif os.path.exists(pkl):
            ok.append(code)
        elif os.path.exists(os.path.join(snap_dir, f"{code}.nodata")):
            nodata.append(code)
        else:
            missing.append(code)
    if missing or failed:
        print(f"❌ 快照不完整: 有数据 {len(ok)} / 无数据 {len(nodata)}"
              f" / 应冻结 {len(codes)}")
        if missing:
            print(f"   未冻结 {len(missing)}: {', '.join(missing[:15])}")
        if failed:
            print(f"   .failed {len(failed)}（网络/IO 意外）: {', '.join(failed[:15])}")
        print("   修复: python3 scripts/pin_hk_snapshot.py")
        return 1
    print(f"✅ 快照完整: 有数据 {len(ok)} + 冻结的无数据 {len(nodata)}"
          f" = {len(ok) + len(nodata)}/{len(codes)}，0 个 .failed")
    if nodata:
        print(f"   ⚠️  冻结为「无数据」（须人工核对确非网络问题）: {', '.join(nodata)}")
    return 0


def main():
    ap = argparse.ArgumentParser(description='港股原始行情快照：冻结 / 校验')
    ap.add_argument('--check', action='store_true', help='只校验完整性，不写入')
    ap.add_argument('--snap-dir', default=DEFAULT_SNAP_DIR, help='快照目录')
    ap.add_argument('--stocks', nargs='+', help='股票代码列表（默认 TRAINING_STOCKS）')
    ap.add_argument('--period-days', type=int, default=SNAP_PERIOD_DAYS,
                    help=f'冻结的天数上限（默认 {SNAP_PERIOD_DAYS}）')
    args = ap.parse_args()

    codes = args.stocks or _default_codes()
    codes = [c.replace('.HK', '').zfill(5) for c in codes]
    if args.check:
        return do_check(codes, args.snap_dir)
    return do_pin(codes, args.snap_dir, args.period_days)


if __name__ == '__main__':
    sys.exit(main())
