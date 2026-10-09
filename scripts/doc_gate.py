#!/usr/bin/env python3
"""文档与提交闸门 —— 三道闸 B 闸的可执行部分。

散文清单靠"记得执行"，本脚本靠**非零退出**。设计原则见
`docs/AI_WORKFLOW.md` §3.3「规则一律落成脚本或可 grep 的清单」。

检查项（doc 子命令，默认）：
  L1  markdown 内链断链            必须 0
  L2  markdown 表格列数一致        必须 0 异常
  L3  表格单元内被转义的 shell 管道 必须 0（`\\|` 会改变命令语义）
  L4  已撤销/已外迁内容残留        必须 0
  L5  A 闸可运行（--help 不得崩溃）

检查项（--tests 追加）：
  T1  全量 pytest

用法：
  python3 scripts/doc_gate.py            # 仅文档检查（秒级）
  python3 scripts/doc_gate.py --tests    # 追加全量测试（约 5 分钟）
  python3 scripts/doc_gate.py --staged   # 提交规范：暂存区禁入文件检查

非零退出 = 阻断。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 扫描范围：版本库内的活文档。排除 output/（历史快照，按规则不追新）
SCAN_GLOBS = [
    "AGENTS.md",
    "README.md",
    "docs/*.md",
    ".opencode/command/*.md",
]

# B 闸第 4 项「无过期残留」的种子表 —— 已撤销 / 已外迁 / 已证伪的内容。
# 发现即 FAIL。决策变更时在此**追加**，不要删除历史条目。
BANNED_PATTERNS = [
    (r"WALKFORWARD_DATA_END", "已撤销的端点截断变量（2026-10-05 撤销）"),
    (r"跨快照数值极差", "已撤销的归因（实为旧特征缓存差异）"),
]

# 行内出现这些标记 = 该行是「记录撤销本身」的条目，不算残留
WITHDRAWN_MARKERS = ("已撤销", "~~", "不可引用", "已被证伪", "已证伪")

# 用于区分「shell 命令里的管道」与「数学/正则里的竖线」
_SHELL_CMD = re.compile(
    r"\b(grep|awk|sed|python3?|git|xargs|curl|wget|find|cut|sort|uniq|tr)\b"
)


# 提交规范：禁入暂存区的路径（AGENTS「Git 提交规范」的可执行版）
# 说明：`.pkl`/`.csv` 走「新增才拦」策略 —— `data/a_stock_models/*.pkl` 等
# 既有资产由 CI 定期更新（`Update A股 prediction results`），拦更新会卡死 CI 提交。
FORBIDDEN_STAGED = [
    (r"^data/model_accuracy\.json$", "按规范不提交（运行时状态）"),
    (r"^data/us_market_cache/", "缓存不入库"),
]

# 例外：回测入库 CSV（scripts/commit_backtest_result.py 自动提交，AGENTS Git 规范）
HK20D_CSV = re.compile(r"^output/\d{8}_\d{6}_catboost_20d/prediction_analysis\.csv$")
# 例外：SFC 聚合卖空周报（模型输入特征，D20 生产纳入；fetch_sfc_short.py 周更，
# 工作流 commit 步骤写回仓库，类比 HK20D_CSV 例外）
SFC_CSV = re.compile(r"^data/sfc_short/(sfc_short_long\.csv|manifest\.json)$")


def _files() -> list[Path]:
    out: list[Path] = []
    for g in SCAN_GLOBS:
        out.extend(sorted(ROOT.glob(g)))
    return [p for p in out if p.is_file()]


# ---------------------------------------------------------------- L1 断链
def check_links() -> list[str]:
    fails = []
    pat = re.compile(r"\]\(([^)\s#]+)(?:#[^)]*)?\)")
    for p in _files():
        base = p.parent
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in pat.finditer(line):
                t = m.group(1)
                if t.startswith(("http://", "https://", "mailto:")):
                    continue
                if not (base / t).exists():
                    fails.append(f"{p.relative_to(ROOT)}:{i} 断链 -> {t}")
    return fails


# ------------------------------------------------------- L2/L3 表格完整性
def _table_rows(lines: list[str]) -> list[list[str]]:
    """按表分块，返回每个表格的行号列表（行号为 1-based）。"""
    blocks, i = [], 0
    sep = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and sep.match(lines[i + 1]):
            j, rows = i, []
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(j + 1)
                j += 1
            blocks.append(rows)
            i = j
        else:
            i += 1
    return blocks


def _ncols(row: str) -> int:
    """列数：转义的 \\| 视为内容，不计为分隔符。"""
    return row.count("|") - row.count("\\|")


def check_tables() -> list[str]:
    fails = []
    for p in _files():
        lines = p.read_text(encoding="utf-8").splitlines()
        for rows in _table_rows(lines):
            header = _ncols(lines[rows[0] - 1])
            for ln in rows:
                n = _ncols(lines[ln - 1])
                if n != header:
                    fails.append(
                        f"{p.relative_to(ROOT)}:{ln} 表格列数 {n} != 表头 {header}"
                    )
    return fails


def check_pipe_escaped_in_table() -> list[str]:
    """含 `|` 的 **shell 命令**被塞进表格并转义 —— 命令语义会被改变。

    转义后的字符串在 GNU ERE 下不作析取（实测 26 行 -> 11 行，漏掉关键判定行）。
    规则：这类命令一律放代码块。

    数学/正则里的 `|`（如 `|X|`）用 `\\|` 转义是**正确**的，不在本检查范围。
    """
    fails = []
    for p in _files():
        lines = p.read_text(encoding="utf-8").splitlines()
        for rows in _table_rows(lines):
            for ln in rows:
                row = lines[ln - 1]
                if "\\|" in row and _SHELL_CMD.search(row):
                    fails.append(
                        f"{p.relative_to(ROOT)}:{ln} 表格内含被转义的 shell 管道 "
                        f"（命令会被改坏，移入代码块）"
                    )
    return fails


# ------------------------------------------------------- L4 过期残留
def check_banned() -> list[str]:
    fails = []
    for p in _files():
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if any(m in line for m in WITHDRAWN_MARKERS):
                continue  # 该行是在**记录撤销本身**，不算残留
            for pat, why in BANNED_PATTERNS:
                if re.search(pat, line):
                    fails.append(f"{p.relative_to(ROOT)}:{i} 过期残留 /{pat}/ —— {why}")
    return fails


# ------------------------------------------------------- L5 A 闸可运行
def check_gate_runnable() -> list[str]:
    script = ROOT / "scripts" / "presentation_gate.py"
    if not script.exists():
        return [f"缺少 A 闸脚本 {script}"]
    r = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True, timeout=60,
    )
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        return [f"presentation_gate.py --help 失败 exit={r.returncode}: {' | '.join(tail)}"]
    return []


# ------------------------------------------------------- T1 测试
def check_tests() -> list[str]:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--timeout=600"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if r.returncode != 0:
        tail = [x for x in r.stdout.splitlines() if x.startswith("FAILED")]
        return ["pytest 失败（{} 例）: {}".format(len(tail), "; ".join(tail[:8]))]
    return []


# ------------------------------------------------------- 提交规范
def check_staged() -> list[str]:
    fails = []
    r = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0:
        return ["git diff --cached 执行失败：" + r.stderr.strip()]
    staged = [x for x in r.stdout.splitlines() if x.strip()]
    # 「新增」必须相对 HEAD 判定：`git ls-files` 读的是索引，
    # 暂存后的新文件本身就在索引里，用它判定会恒为"已跟踪"而全部放行。
    ra = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                        capture_output=True, text=True, cwd=str(ROOT))
    added = set(ra.stdout.splitlines()) if ra.returncode == 0 else set()
    for f in staged:
        for pat, why in FORBIDDEN_STAGED:
            if re.search(pat, f):
                fails.append(f"暂存区禁入 {f} —— {why}")
        suf = Path(f).suffix.lower()
        if suf == ".bak":
            fails.append(f"暂存区禁入 {f} —— 备份文件不入库")
        elif suf in {".pkl", ".csv"} and f in added and not HK20D_CSV.search(f) and not SFC_CSV.search(f):
            fails.append(f"暂存区禁入 {f} —— 新增 {suf} 数据文件（更新既有跟踪文件不受限）")
    return fails


def check_py_compile() -> list[str]:
    fails = []
    r = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                       capture_output=True, text=True, cwd=str(ROOT))
    for f in [x for x in r.stdout.splitlines() if x.endswith(".py")]:
        p = ROOT / f
        if not p.exists():
            continue
        cp = subprocess.run([sys.executable, "-m", "py_compile", str(p)],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            fails.append(f"py_compile 失败 {f}: {cp.stderr.strip().splitlines()[-1]}")
    return fails


# ---------------------------------------------------------------- runner
def _run(name: str, fn) -> int:
    fails = fn()
    if fails:
        print(f"❌ {name}")
        for x in fails:
            print("   ✗ " + x)
        return 1
    print(f"✅ {name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="文档与提交闸门（B 闸可执行部分）")
    ap.add_argument("--tests", action="store_true", help="追加全量 pytest（约 5 分钟）")
    ap.add_argument("--staged", action="store_true", help="提交规范 + 暂存 py 语法检查")
    args = ap.parse_args()

    rc = 0
    print("=" * 70)
    print("🚦 文档与提交闸门")
    print("=" * 70)
    rc |= _run("L1 markdown 内链断链", check_links)
    rc |= _run("L2 markdown 表格列数", check_tables)
    rc |= _run("L3 表格内转义管道 |", check_pipe_escaped_in_table)
    rc |= _run("L4 过期残留", check_banned)
    rc |= _run("L5 A 闸可运行", check_gate_runnable)
    if args.staged:
        rc |= _run("S1 提交规范", check_staged)
        rc |= _run("S2 py_compile", check_py_compile)
    if args.tests:
        rc |= _run("T1 pytest", check_tests)

    print("-" * 70)
    if rc:
        print("🛑 闸门未通过 —— 不得提交 / 不得作为结论")
    else:
        print("✅ 闸门通过")
    return rc


if __name__ == "__main__":
    sys.exit(main())
