"""懒加载测试（P0 内存修复）：LRU 容量、变异持久化、渲染次数。"""

import numpy as np
import pymupdf
import pytest

from core.document import Document, Page, PAGE_CACHE_LIMIT


def _multipage_pdf(tmp_path, n=10, size_pt=(595, 842)) -> object:
    path = tmp_path / "big.pdf"
    doc = pymupdf.open()
    for i in range(n):
        page = doc.new_page(width=size_pt[0], height=size_pt[1])
        page.draw_rect(pymupdf.Rect(50, 50, 200, 200), color=(0.8, 0.1, 0.1), fill=(0.9, 0.2, 0.2))
    doc.save(path)
    doc.close()
    return path


def test_lazy_pdf_does_not_render_all_pages(tmp_path):
    """打开 10 页 PDF 不应渲染任何全尺寸页面（懒加载核心）。"""
    path = _multipage_pdf(tmp_path, n=10)
    doc = Document.open(path)
    assert len(doc._cache) == 0, "打开时不该有渲染缓存"
    # 访问第 1 页才渲染
    img = doc.pages[0].image
    assert img.shape[0] > 0
    assert len(doc._cache) == 1
    doc.close()


def test_lru_cap(tmp_path):
    """缓存上限：顺序访问 N 页后驻留不超过 PAGE_CACHE_LIMIT。"""
    n = PAGE_CACHE_LIMIT + 4
    path = _multipage_pdf(tmp_path, n=n)
    doc = Document.open(path)
    for p in doc.pages:
        _ = p.image
    assert len(doc._cache) <= PAGE_CACHE_LIMIT
    # 再访问第一页：应重新渲染（已被淘汰）
    assert id(doc.pages[0]) not in doc._cache
    _ = doc.pages[0].image
    assert len(doc._cache) <= PAGE_CACHE_LIMIT
    doc.close()


def test_mutation_persists_after_eviction(tmp_path):
    """变异页（旋转/校准写入）不参与淘汰：改完再翻几十页，图像不变。"""
    n = PAGE_CACHE_LIMIT * 2
    path = _multipage_pdf(tmp_path, n=n)
    doc = Document.open(path)
    page = doc.pages[0]
    _ = page.image  # 确保先渲染
    override = np.zeros((10, 10, 3), dtype=np.uint8)  # 任意变异结果
    page.image = override
    assert page.mutated
    # 翻完全部页把 LRU 冲掉
    for p in doc.pages:
        _ = p.image
    # 变异结果仍在（同一对象，未被淘汰）
    assert page.image is override
    doc.close()


def test_render_count(tmp_path, monkeypatch):
    """同页重复访问只渲染一次（缓存命中）。"""
    path = _multipage_pdf(tmp_path, n=3)
    doc = Document.open(path)
    calls = {"n": 0}
    orig = doc._render_pdf_page

    def counting(index, dpi):
        calls["n"] += 1
        return orig(index, dpi)

    monkeypatch.setattr(doc, "_render_pdf_page", counting)
    # 注意：懒加载页的 render_fn 闭包引用的是 doc._render_pdf_page，patch 后生效
    for _ in range(3):
        _ = doc.pages[1].image
    assert calls["n"] <= 1, f"重复访问不应重复渲染（渲染了 {calls['n']} 次）"
    doc.close()


def test_thumbnail_does_not_touch_full_cache(tmp_path):
    """缩略图独立低 DPI 渲染，不占用全尺寸 LRU。"""
    path = _multipage_pdf(tmp_path, n=5)
    doc = Document.open(path)
    for p in doc.pages:
        t = p.thumbnail()
        assert t.shape[0] > 0
    assert len(doc._cache) == 0, "缩略图不该进全尺寸缓存"
    doc.close()


def test_eager_page_still_works():
    """直接构造 Page(image=...) 的旧路径（测试/合成页）不受影响。"""
    img = np.full((100, 80, 3), 255, dtype=np.uint8)
    page = Page(image=img, phys_w_mm=210.0, phys_h_mm=297.0)
    assert page.image is img
    assert page.mutated
    assert page.width_px == 80 and page.height_px == 100


def test_images_lazy_and_close(tmp_path):
    """图片文档同样懒加载，close 释放。"""
    from PIL import Image

    for i in range(3):
        Image.new("RGB", (800, 1100), (250, 250, 250)).save(tmp_path / f"p{i}.png")
    doc = Document.from_images([tmp_path / f"p{i}.png" for i in range(3)])
    assert len(doc._cache) == 0
    _ = doc.pages[0].image
    assert len(doc._cache) == 1
    doc.close()
    assert len(doc._cache) == 0
