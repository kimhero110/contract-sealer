"""无 Qt 的盖章会话：落章 → 渲染 → 导出，模板与批量，命令行入口。

这些用例在没装 PySide6 的机器上也要跑：它们守的正是"导出与批量不依赖界面"这条线。
"""

import json
from pathlib import Path

import numpy as np
import pytest

from core.cli import main as cli_main
from core.document import Document
from core.export import ExportOptions
from core.extract import extract_ink
from core.randomize import RandomSpec
from core.seal import Seal
from core.session import (
    BatchJob,
    ExportCancelled,
    Session,
    batch_stamp,
    build_document,
    load_library,
    natural_key,
)
from core.stamp import measure_ink_diameter_mm
from core.template import save_template

FIX = Path("tests/fixtures")
SPEC = RandomSpec(angle_deg=0.0, tone=0.0, dust=0.0)


@pytest.fixture(scope="module")
def seal() -> Seal:
    return Seal(name="公章", kind="seal", image=extract_ink(FIX / "seal_company.png"), phys_mm=40.0)


@pytest.fixture
def library(tmp_path, seal) -> Path:
    lib = tmp_path / "lib"
    seal.save(lib)
    return lib


def test_natural_key_orders_pages():
    names = ["p10.jpg", "p2.jpg", "P1.jpg"]
    assert sorted(names, key=natural_key) == ["P1.jpg", "p2.jpg", "p10.jpg"]


def test_build_document_rejects_mixed_pdf_and_images(tmp_path):
    with pytest.raises(ValueError):
        build_document([str(FIX / "1.jpg"), str(tmp_path / "x.pdf")])


def test_stamp_render_export_roundtrip(tmp_path, seal):
    """落一枚 40mm 章 → 导出 PDF → 页面图里量得的直径仍是 40mm（±1mm）。"""
    session = Session(Document.from_images([FIX / "1.jpg"]), seed=7)
    rec = session.add_stamp(0, seal, 100.0, 200.0, SPEC)
    assert rec.applied is not None and rec.processed is not None
    assert session.stamp_count() == 1

    img = session.render_page(0)
    assert abs(measure_ink_diameter_mm(img, session.doc.pages[0]) - 40.0) < 1.0

    out = session.export(tmp_path / "out.pdf", SPEC, ExportOptions())
    assert out.exists()
    log = json.loads(out.with_suffix(".sealog").read_text(encoding="utf-8"))
    assert log["seed"] == 7
    assert log["stamps"][0]["seal"] == "公章" and log["stamps"][0]["page"] == 1
    assert log["export_options"]["image_format"] == "jpeg"
    session.close()


def test_render_page_without_records_returns_original(seal):
    session = Session(Document.from_images([FIX / "1.jpg"]))
    assert session.render_page(0) is session.doc.pages[0].image


def test_export_reports_progress_and_honours_cancel(tmp_path, seal):
    session = Session(Document.from_images([FIX / "1.jpg", FIX / "2.jpg", FIX / "3.jpg"]))
    seen: list[tuple[int, int]] = []
    session.export(
        tmp_path / "p.pdf", SPEC, progress=lambda i, n, _t: seen.append((i, n))
    )
    assert seen[0] == (0, 3) and seen[-1] == (3, 3)

    calls = {"n": 0}

    def cancel_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    with pytest.raises(ExportCancelled):
        session.export(tmp_path / "c.pdf", SPEC, cancelled=cancel_after_first)
    assert not (tmp_path / "c.pdf").exists()
    assert not (tmp_path / "c.sealog").exists()


def test_template_roundtrip_and_missing_seal_skipped(seal):
    session = Session(Document.from_images([FIX / "1.jpg", FIX / "2.jpg"]))
    session.add_stamp(0, seal, 105.0, 148.5, SPEC, rotation_deg=3.0, opacity=0.9)
    entries = session.template_entries(0)
    assert entries[0]["rel_x"] == pytest.approx(0.5) and entries[0]["rel_y"] == pytest.approx(0.5)

    entries.append({**entries[0], "seal_name": "不存在的章"})
    applied = session.apply_template(entries, {"公章": seal}, 1, SPEC)
    assert applied == 1
    rec = session.records(1)[0]
    assert rec.rotation_deg == 3.0 and rec.opacity == 0.9
    assert (rec.center_x_mm, rec.center_y_mm) == pytest.approx((105.0, 148.5))


def test_add_perforation_covers_all_pages(seal):
    session = Session(Document.from_images([FIX / "1.jpg", FIX / "2.jpg", FIX / "3.jpg"]))
    touched = session.add_perforation(seal, SPEC, seed=11)
    assert touched == {0, 1, 2}
    recs = [r for rs in session.stamps.values() for r in rs]
    assert all(r.locked and r.group == "perf_11" for r in recs)


def test_reroll_skips_locked(seal):
    session = Session(Document.from_images([FIX / "1.jpg"]))
    normal = session.add_stamp(0, seal, 50.0, 50.0, SPEC)
    session.add_perforation(seal, SPEC, seed=3)
    before = normal.applied
    assert session.reroll(RandomSpec(angle_deg=5.0, tone=0.2, dust=0.3)) == 1
    assert normal.applied is not before


def test_remove_and_page_check(seal):
    session = Session(Document.from_images([FIX / "1.jpg"]))
    rec = session.add_stamp(0, seal, 50.0, 50.0, SPEC)
    assert session.remove(0, [rec]) == 1
    assert session.records(0) == [] and session.stamp_count() == 0
    with pytest.raises(IndexError):
        session.add_stamp(5, seal, 0.0, 0.0, SPEC)


def test_batch_stamp_last_page_with_perforation(tmp_path, library, seal):
    """批量：两份文件各自套模板到末页 + 骑缝章，输出到指定目录，坏文件独立失败。"""
    seals = load_library(library)
    assert set(seals) == {"公章"}
    tpl = [{"seal_name": "公章", "kind": "seal", "rel_x": 0.7, "rel_y": 0.8, "size_mm": 40.0}]
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    out_dir = tmp_path / "out"
    job = BatchJob(
        paths=[str(FIX / "1.jpg"), str(FIX / "2.jpg"), str(bad)],
        template_entries=tpl,
        seals_by_name=seals,
        out_dir=out_dir,
        perforation_seal=seal,
        random_spec=SPEC,
        seed=5,
    )
    seen: list[str] = []
    results = batch_stamp(job, progress=lambda _i, _n, t: seen.append(t))
    assert [r.ok for r in results] == [True, True, False]
    assert all(r.output.parent == out_dir for r in results if r.ok)
    assert seen[-1] == "完成"
    log = json.loads(results[0].output.with_suffix(".sealog").read_text(encoding="utf-8"))
    kinds = {s["kind"] for s in log["stamps"]}
    assert kinds == {"seal", "perforation_slice"}


def test_batch_cancel_stops_early(tmp_path, library):
    seals = load_library(library)
    tpl = [{"seal_name": "公章", "kind": "seal", "rel_x": 0.5, "rel_y": 0.5, "size_mm": 40.0}]
    job = BatchJob(
        paths=[str(FIX / "1.jpg"), str(FIX / "2.jpg")],
        template_entries=tpl,
        seals_by_name=seals,
        out_dir=tmp_path,
        random_spec=SPEC,
    )
    results = batch_stamp(job, cancelled=lambda: True)
    assert results == []


def test_cli_batch_and_list(tmp_path, library, capsys):
    templates = tmp_path / "templates"
    save_template(
        templates, "末页落款",
        [{"seal_name": "公章", "kind": "seal", "rel_x": 0.6, "rel_y": 0.85, "size_mm": 40.0}],
    )
    out_dir = tmp_path / "cli_out"
    code = cli_main([
        "--library", str(library), "--templates", str(templates),
        "batch", "--template", "末页落款", "--out", str(out_dir), "--seed", "1",
        "--angle", "0", "--tone", "0", "--dust", "0", "--png",
        str(FIX / "1.jpg"),
    ])
    assert code == 0
    outputs = list(out_dir.glob("*.pdf"))
    assert len(outputs) == 1
    log = json.loads(outputs[0].with_suffix(".sealog").read_text(encoding="utf-8"))
    assert log["export_options"]["image_format"] == "png"

    code = cli_main(["--library", str(library), "--templates", str(templates), "list"])
    assert code == 0
    text = capsys.readouterr().out
    assert "公章" in text and "末页落款" in text


def test_cli_unknown_template_exits(tmp_path, library):
    with pytest.raises(SystemExit):
        cli_main([
            "--library", str(library), "--templates", str(tmp_path / "none"),
            "batch", "--template", "没有这个", str(FIX / "1.jpg"),
        ])


def test_cli_failure_returns_nonzero(tmp_path, library):
    templates = tmp_path / "templates"
    save_template(
        templates, "t", [{"seal_name": "公章", "kind": "seal", "rel_x": 0.5, "rel_y": 0.5, "size_mm": 40.0}]
    )
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"\x00")
    code = cli_main([
        "--library", str(library), "--templates", str(templates),
        "batch", "--template", "t", "--out", str(tmp_path), str(bad),
    ])
    assert code == 1


def test_export_png_option_is_larger_than_jpeg(tmp_path, seal):
    """无损 PNG 换来的是体积：同一页 PNG 导出明显大于 JPEG 导出。"""
    session = Session(Document.from_images([FIX / "1.jpg"]))
    session.add_stamp(0, seal, 100.0, 100.0, SPEC)
    png = session.export(tmp_path / "png.pdf", SPEC, ExportOptions("png"))
    jpeg = session.export(tmp_path / "jpeg.pdf", SPEC, ExportOptions("jpeg", 92))
    assert png.stat().st_size > jpeg.stat().st_size * 2
    assert np.asarray(session.render_page(0)).ndim == 3
