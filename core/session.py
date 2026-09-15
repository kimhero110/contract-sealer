"""盖章会话：文档 + 各页盖章记录 + 随机器。导出与批量盖章的无 Qt 内核。

界面层（app/）只负责把用户动作翻译成对 Session 的调用，并把记录画到画布上；
命令行入口（core/cli.py）直接驱动 Session。两者共用同一条渲染/导出路径，
所以"画布上看到的"与"命令行批出来的"是同一份算法。

会话状态模型：
- Document（core.document）提供页面图像与物理尺寸；
- 每页若干 StampRecord（印章 + 物理位置/尺寸/旋转/不透明度 + 已采样随机效果）；
- 预览与导出共用同一份 processed 图像——所见即所得；
- 骑缝章切片是 locked 记录：随机在生成时已定，导出不再重采样。
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .datestamp import KIND_DATE
from .document import MM_PER_INCH, Document, Page
from .export import ExportOptions, export_pdf, make_output_path
from .perforation import PerforationSpec, SlicePlacement, plan_perforation
from .randomize import AppliedRandom, Randomizer, RandomSpec
from .seal import Seal, list_library
from .stamp import Placement, stamp_page

# 撤销快照要还原的记录字段。新增可变字段必须同步加进来，否则撤销会漏改。
RECORD_FIELDS = (
    "center_x_mm",
    "center_y_mm",
    "size_mm",
    "rotation_deg",
    "opacity",
    "applied",
    "processed",
    "locked",
    "group",
)

# 进度回调：(已完成数, 总数, 说明文字)
ProgressFn = Callable[[int, int, str], None]
# 取消探针：返回 True 表示用户要求中止
CancelFn = Callable[[], bool]


class ExportCancelled(Exception):
    """用户中途取消：不产生任何输出文件。"""


@dataclass
class StampRecord:
    seal: Seal
    center_x_mm: float
    center_y_mm: float
    size_mm: float
    rotation_deg: float = 0.0
    opacity: float = 1.0
    applied: AppliedRandom | None = None
    processed: np.ndarray | None = None  # 随机效果后的 RGBA，预览/导出共用
    locked: bool = False                 # 骑缝章切片：导出不重采样
    group: str | None = None             # 骑缝章分组标识（sealog 用）

    def ink(self) -> np.ndarray:
        return self.processed if self.processed is not None else self.seal.image

    def placement(self) -> Placement:
        return Placement(
            center_x_mm=self.center_x_mm,
            center_y_mm=self.center_y_mm,
            size_mm=self.size_mm,
            rotation_deg=self.rotation_deg,
            opacity=self.opacity,
        )

    def to_log(self, page_index: int) -> dict:
        return {
            "page": page_index + 1,
            "seal": self.seal.name,
            "kind": "perforation_slice" if self.locked else self.seal.kind,
            "group": self.group,
            "center_mm": [round(self.center_x_mm, 2), round(self.center_y_mm, 2)],
            "size_mm": self.size_mm,
            "rotation_deg": self.rotation_deg,
            "opacity": self.opacity,
            "random": self.applied.to_dict() if self.applied else None,
        }


def natural_key(path: str | Path) -> list[int | str]:
    """自然排序：第 2 页排在第 10 页前面（多图合成文档的页序）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]


def build_document(paths: Sequence[str | Path]) -> Document:
    """打开一份合同：单个 PDF，或若干图片按自然序合成。"""
    if len(paths) == 1 and str(paths[0]).lower().endswith(".pdf"):
        return Document.open(paths[0])
    if any(str(p).lower().endswith(".pdf") for p in paths):
        raise ValueError("PDF 请一次只选一个（图片可以多选）")
    return Document.from_images(sorted(paths, key=natural_key))


def load_library(library_dir: Path) -> dict[str, Seal]:
    """印章库 → {显示名: Seal}。坏文件跳过，不让一枚损坏的章拖垮整个库。"""
    out: dict[str, Seal] = {}
    for png in list_library(library_dir):
        try:
            seal = Seal.load(png)
        except Exception:
            continue
        out[seal.name] = seal
    return out


def new_seed() -> int:
    return secrets.randbelow(2**31 - 1)


class Session:
    """一份打开的文档及其上的全部盖章记录。"""

    def __init__(self, doc: Document, seed: int | None = None):
        self.doc = doc
        self.stamps: dict[int, list[StampRecord]] = {}
        self.rng = Randomizer(new_seed() if seed is None else seed)

    # ── 记录 ──

    def stamp_count(self) -> int:
        return sum(len(v) for v in self.stamps.values())

    def records(self, page_index: int) -> list[StampRecord]:
        return self.stamps.get(page_index, [])

    def new_record(
        self,
        seal: Seal,
        spec: RandomSpec,
        center_x_mm: float = 0.0,
        center_y_mm: float = 0.0,
        size_mm: float | None = None,
        rotation_deg: float = 0.0,
        opacity: float = 1.0,
    ) -> StampRecord:
        """采样随机效果并生成一条记录（不挂到页上——界面层在用户点位后再挂）。

        日期戳不加随机手感：打印/书写的日期没有印泥深浅，加蒙尘反而假。
        """
        if seal.kind == KIND_DATE:
            processed, applied = seal.image, None
        else:
            processed, applied = self.rng.apply_auto(seal.image, spec)
        return StampRecord(
            seal=seal,
            center_x_mm=center_x_mm,
            center_y_mm=center_y_mm,
            size_mm=seal.phys_mm if size_mm is None else size_mm,
            rotation_deg=rotation_deg,
            opacity=opacity,
            applied=applied,
            processed=processed,
        )

    def add(self, page_index: int, rec: StampRecord) -> StampRecord:
        self._check_page(page_index)
        self.stamps.setdefault(page_index, []).append(rec)
        return rec

    def add_stamp(
        self,
        page_index: int,
        seal: Seal,
        center_x_mm: float,
        center_y_mm: float,
        spec: RandomSpec,
        **kwargs: float,
    ) -> StampRecord:
        return self.add(
            page_index, self.new_record(seal, spec, center_x_mm, center_y_mm, **kwargs)
        )

    def remove(self, page_index: int, recs: list[StampRecord]) -> int:
        doomed = {id(r) for r in recs}
        before = len(self.records(page_index))
        kept = [r for r in self.records(page_index) if id(r) not in doomed]
        self.stamps[page_index] = kept  # 留空列表：界面按"该页有过记录"访问，不做 KeyError 兜底
        return before - len(kept)

    def prune_empty(self) -> None:
        self.stamps = {k: v for k, v in self.stamps.items() if v}

    def reroll(self, spec: RandomSpec, rng: Randomizer | None = None) -> int:
        """重摇所有普通章/签名的随机效果（骑缝切片与日期除外）。返回重摇数。"""
        rng = rng or Randomizer(new_seed())
        count = 0
        for records in self.stamps.values():
            for rec in records:
                if rec.locked or rec.seal.kind == KIND_DATE:
                    continue
                rec.processed, rec.applied = rng.apply_auto(rec.seal.image, spec)
                count += 1
        return count

    # ── 模板 ──

    def template_entries(self, page_index: int) -> list[dict]:
        """把一页上的盖章组合导出为模板条目（相对位置，跨页面尺寸可复用）。

        日期戳不入模板：它的"印章"是临时合成的，不在库里，套用时必然落空。
        """
        page = self.doc.pages[page_index]
        return [
            {
                "seal_name": r.seal.name,
                "kind": r.seal.kind,
                "rel_x": r.center_x_mm / page.phys_w_mm,
                "rel_y": r.center_y_mm / page.phys_h_mm,
                "size_mm": r.size_mm,
                "rotation_deg": r.rotation_deg,
                "opacity": r.opacity,
            }
            for r in self.records(page_index)
            if r.seal.kind != KIND_DATE
        ]

    def apply_template(
        self,
        entries: list[dict],
        seals_by_name: dict[str, Seal],
        page_index: int,
        spec: RandomSpec,
    ) -> int:
        """把模板条目套到目标页。返回套用的章数（库里没有的章跳过）。"""
        self._check_page(page_index)
        page = self.doc.pages[page_index]
        count = 0
        for e in entries:
            seal = seals_by_name.get(e["seal_name"])
            if seal is None:
                continue
            self.add_stamp(
                page_index,
                seal,
                e["rel_x"] * page.phys_w_mm,
                e["rel_y"] * page.phys_h_mm,
                spec,
                size_mm=e["size_mm"],
                rotation_deg=e.get("rotation_deg", 0.0),
                opacity=e.get("opacity", 1.0),
            )
            count += 1
        return count

    # ── 骑缝章 ──

    def add_placements(
        self, seal: Seal, placements: list[SlicePlacement], group: str
    ) -> set[int]:
        """把切片放置结果转成 locked 记录挂到各页。返回受影响的页码集合。"""
        touched: set[int] = set()
        for pl in placements:
            page = self.doc.pages[pl.page_index]
            h_px, w_px = pl.slice_rgba.shape[:2]
            w_mm = w_px / page.dpi * MM_PER_INCH
            h_mm = h_px / page.dpi * MM_PER_INCH
            self.add(
                pl.page_index,
                StampRecord(
                    seal=seal,
                    center_x_mm=pl.right_edge_mm - w_mm / 2,
                    center_y_mm=pl.top_mm + h_mm / 2,
                    size_mm=w_mm,
                    processed=pl.slice_rgba,
                    locked=True,
                    group=group,
                ),
            )
            touched.add(pl.page_index)
        return touched

    def add_perforation(
        self,
        seal: Seal,
        random_spec: RandomSpec,
        seed: int | None = None,
        page_indices: list[int] | None = None,
        perf_spec: PerforationSpec | None = None,
    ) -> set[int]:
        """不经对话框的骑缝章：全局随机 → 切片 → 落到各页。返回受影响页码。"""
        seed = new_seed() if seed is None else seed
        processed, _ = Randomizer(seed).apply_auto(seal.image, random_spec)
        if page_indices is None:
            page_indices = list(range(len(self.doc.pages)))
        spec = perf_spec or PerforationSpec(seed=seed)
        if spec.seed != seed:
            spec = PerforationSpec(**{**spec.__dict__, "seed": seed})
        try:
            placements = plan_perforation(processed, self.doc.pages, page_indices, spec)
        except ValueError as e:
            raise ValueError(f"骑缝章：{e}") from e
        return self.add_placements(seal, placements, f"perf_{seed}")

    # ── 渲染与导出 ──

    def render_page(self, page_index: int) -> np.ndarray:
        """把该页所有记录按顺序合成到页面图像上。无记录时返回原图（不复制）。"""
        page = self.doc.pages[page_index]
        img = page.image
        for rec in self.records(page_index):
            cur = Page(image=img, phys_w_mm=page.phys_w_mm, phys_h_mm=page.phys_h_mm)
            img = stamp_page(cur, rec.ink(), rec.placement())
        return img

    def render_all(
        self, progress: ProgressFn | None = None, cancelled: CancelFn | None = None
    ) -> list[np.ndarray]:
        total = len(self.doc.pages)
        images: list[np.ndarray] = []
        for i in range(total):
            if cancelled is not None and cancelled():
                raise ExportCancelled()
            if progress is not None:
                progress(i, total, f"合成第 {i + 1}/{total} 页")
            images.append(self.render_page(i))
        return images

    def sealog(self, spec: RandomSpec) -> dict:
        return {
            # 顶层 seed 仅供参考；复现的权威数据是每枚章的 random.applied 值
            "seed": self.rng.seed,
            "seed_note": "per-stamp random.applied values are authoritative for replay",
            "random_spec": {"angle_deg": spec.angle_deg, "tone": spec.tone, "dust": spec.dust},
            "source": str(self.doc.source_path) if self.doc.source_path else None,
            "page_count": len(self.doc.pages),
            "stamps": [
                rec.to_log(page_idx)
                for page_idx, records in sorted(self.stamps.items())
                for rec in records
            ],
        }

    def output_path(self, out_dir: Path | None) -> Path:
        """默认输出路径：指定目录，否则与源文件同目录，再否则当前目录。"""
        if out_dir is None:
            out_dir = self.doc.source_path.parent if self.doc.source_path else Path.cwd()
        return make_output_path(self.doc.source_path, out_dir)

    def export(
        self,
        out_path: Path,
        spec: RandomSpec,
        options: ExportOptions | None = None,
        progress: ProgressFn | None = None,
        cancelled: CancelFn | None = None,
    ) -> Path:
        """所见即所得地导出：直接用各记录已采样的 processed 图像，不重采样。"""
        images = self.render_all(progress, cancelled)
        if cancelled is not None and cancelled():
            raise ExportCancelled()
        if progress is not None:
            progress(len(images), len(images), "写入 PDF")
        return export_pdf(self.doc.pages, images, out_path, self.sealog(spec), options)

    def close(self) -> None:
        self.doc.close()
        self.stamps = {}

    def _check_page(self, page_index: int) -> None:
        if not 0 <= page_index < len(self.doc.pages):
            raise IndexError(f"页码 {page_index} 超出范围（共 {len(self.doc.pages)} 页）")


# ── 批量盖章 ──


@dataclass
class BatchResult:
    source: str
    output: Path | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.output is not None


@dataclass(frozen=True)
class BatchJob:
    """一次批量盖章的全部参数（与界面/命令行无关）。"""

    paths: list[str]
    template_entries: list[dict]
    seals_by_name: dict[str, Seal]
    out_dir: Path | None = None            # None = 各自输出到源文件所在目录
    target_last_page: bool = True          # 模板盖末页；False = 盖首页
    perforation_seal: Seal | None = None   # 同时加盖骑缝章（全部页）
    random_spec: RandomSpec = field(default_factory=RandomSpec)
    export_options: ExportOptions = field(default_factory=ExportOptions)
    seed: int | None = None                # 指定则可复现；None 每份文件独立随机


def batch_stamp(
    job: BatchJob,
    progress: ProgressFn | None = None,
    cancelled: CancelFn | None = None,
) -> list[BatchResult]:
    """逐份文件：打开 → 套模板 →（骑缝章）→ 导出。每份独立失败，不影响其他。

    取消时已完成的文件保留，未开始的不再处理。
    """
    results: list[BatchResult] = []
    total = len(job.paths)
    for i, path in enumerate(job.paths):
        if cancelled is not None and cancelled():
            break
        if progress is not None:
            progress(i, total, Path(path).name)
        results.append(_stamp_one(path, job))
    if progress is not None:
        progress(len(results), total, "完成")
    return results


def _stamp_one(path: str, job: BatchJob) -> BatchResult:
    seed = job.seed
    try:
        session = Session(build_document([path]), seed)
    except Exception as e:
        return BatchResult(source=path, error=str(e))
    try:
        page_idx = len(session.doc.pages) - 1 if job.target_last_page else 0
        applied = session.apply_template(
            job.template_entries, job.seals_by_name, page_idx, job.random_spec
        )
        if applied == 0:
            return BatchResult(source=path, error="模板中的印章不在库中")
        if job.perforation_seal is not None:
            session.add_perforation(job.perforation_seal, job.random_spec, seed)
        out = session.export(
            session.output_path(job.out_dir), job.random_spec, job.export_options
        )
        return BatchResult(source=path, output=out)
    except Exception as e:
        return BatchResult(source=path, error=str(e))
    finally:
        session.close()
