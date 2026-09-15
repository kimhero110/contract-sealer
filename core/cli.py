"""命令行入口：不开界面也能批量盖章。

    python -m core.cli list                       # 印章库与模板
    python -m core.cli batch --template 末页落款 --out ./out a.pdf b.pdf ...

模板与印章库和图形界面共用（同一个用户数据目录），先在界面里把章导好、
模板存好，之后月底那 30 份合同交给这一条命令。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .export import DEFAULT_JPEG_QUALITY, FORMAT_JPEG, FORMAT_PNG, ExportOptions
from .randomize import RandomSpec
from .seal import default_library_dir
from .session import BatchJob, batch_stamp, load_library
from .template import default_template_dir, list_templates, load_template


def _resolve_template(name_or_path: str, template_dir: Path) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    for path in list_templates(template_dir):
        if path.stem == name_or_path:
            return path
    names = "、".join(p.stem for p in list_templates(template_dir)) or "（空）"
    raise SystemExit(f"找不到模板「{name_or_path}」。可用模板：{names}")


def _cmd_list(args: argparse.Namespace) -> int:
    seals = load_library(args.library)
    print(f"印章库（{args.library}）：")
    for name, seal in sorted(seals.items()):
        unit = "直径" if seal.kind == "seal" else "宽"
        print(f"  {name}  [{seal.kind}] {unit} {seal.phys_mm:g}mm")
    if not seals:
        print("  （空）")
    print(f"模板（{args.templates}）：")
    paths = list_templates(args.templates)
    for path in paths:
        print(f"  {path.stem}  ({len(load_template(path))} 枚章)")
    if not paths:
        print("  （空）")
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    template_path = _resolve_template(args.template, args.templates)
    seals = load_library(args.library)
    perforation_seal = None
    if args.perforation:
        perforation_seal = seals.get(args.perforation)
        if perforation_seal is None:
            raise SystemExit(f"印章库里没有「{args.perforation}」")
    options = ExportOptions(
        FORMAT_PNG if args.png else FORMAT_JPEG, args.quality
    ).normalized()
    job = BatchJob(
        paths=[str(p) for p in args.files],
        template_entries=load_template(template_path),
        seals_by_name=seals,
        out_dir=args.out,
        target_last_page=not args.first_page,
        perforation_seal=perforation_seal,
        random_spec=RandomSpec(angle_deg=args.angle, tone=args.tone, dust=args.dust),
        export_options=options,
        seed=args.seed,
    )

    def progress(done: int, total: int, text: str) -> None:
        print(f"[{done}/{total}] {text}", flush=True)

    results = batch_stamp(job, progress)
    failed = [r for r in results if not r.ok]
    for r in results:
        print(f"  ✓ {r.source} → {r.output}" if r.ok else f"  ✗ {r.source}: {r.error}")
    print(f"成功 {len(results) - len(failed)} 个，失败 {len(failed)} 个")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contract-sealer", description="扫描合同盖章工具（命令行）"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--library", type=Path, default=default_library_dir(), help="印章库目录（默认用户数据目录）"
    )
    parser.add_argument(
        "--templates", type=Path, default=default_template_dir(), help="模板目录（默认用户数据目录）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出印章库与模板").set_defaults(func=_cmd_list)

    batch = sub.add_parser("batch", help="按模板批量盖章导出")
    batch.add_argument("files", nargs="+", type=Path, help="合同文件（PDF 或图片，每个文件独立处理）")
    batch.add_argument("--template", required=True, help="模板名或模板 JSON 路径")
    batch.add_argument("--out", type=Path, default=None, help="输出目录（默认与各源文件同目录）")
    batch.add_argument("--first-page", action="store_true", help="模板盖在首页（默认末页）")
    batch.add_argument("--perforation", metavar="印章名", help="同时用该章加盖骑缝章（全部页）")
    batch.add_argument("--seed", type=int, default=None, help="随机种子（指定则可复现）")
    batch.add_argument("--angle", type=float, default=RandomSpec.angle_deg, help="随机角度上限（°）")
    batch.add_argument("--tone", type=float, default=RandomSpec.tone, help="随机色度上限（0–1）")
    batch.add_argument("--dust", type=float, default=RandomSpec.dust, help="蒙尘强度上限（0–1）")
    fmt = batch.add_mutually_exclusive_group()
    fmt.add_argument("--png", action="store_true", help="页面无损 PNG 嵌入（体积大）")
    fmt.add_argument(
        "--quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG 质量 1–100（默认 92）"
    )
    batch.set_defaults(func=_cmd_batch)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
