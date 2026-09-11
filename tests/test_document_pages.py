"""文档页增删测试：追加导入、删页、缓存不得串页。"""

import numpy as np
import pytest

from core.document import Document, Page

FIX = "tests/fixtures"


def _doc(n: int) -> Document:
    return Document.from_images([f"{FIX}/{i}.jpg" for i in range(1, n + 1)])


def test_extend_appends_pages_and_keeps_them_renderable():
    doc = _doc(2)
    extra = _doc(1)
    added = doc.extend(extra)
    assert len(doc.pages) == 3
    assert len(added) == 1
    # 来源文档被掏空但仍被持有——它的懒加载闭包还握着句柄
    assert extra.pages == []
    assert extra in doc._sources
    assert doc.pages[2].image.shape[2] == 3  # 追加进来的页仍能渲染


def test_extend_reparents_pages_for_lru():
    doc = _doc(1)
    extra = _doc(1)
    page = extra.pages[0]
    doc.extend(extra)
    assert page._owner is doc


def test_remove_pages_returns_removed_and_shrinks():
    doc = _doc(3)
    kept_uids = [doc.pages[0].uid, doc.pages[2].uid]
    removed = doc.remove_pages([1])
    assert len(removed) == 1
    assert [p.uid for p in doc.pages] == kept_uids


def test_remove_pages_ignores_out_of_range():
    doc = _doc(2)
    assert doc.remove_pages([5, -1]) == []
    assert len(doc.pages) == 2


def test_remove_pages_purges_cache():
    """删掉的页不能在 LRU 里留下悬挂图像。"""
    doc = _doc(2)
    doomed = doc.pages[1]
    _ = doomed.image  # 进缓存
    assert doomed.uid in doc._cache
    doc.remove_pages([1])
    assert doomed.uid not in doc._cache


def test_page_uid_is_unique_even_after_gc():
    """缓存键不能用 id()：删页后对象被回收，新页复用 id 就会拿到上一页的图。"""
    doc = _doc(1)
    seen = set()
    for _ in range(50):
        page = Page(image=np.zeros((4, 4, 3), np.uint8))
        assert page.uid not in seen
        seen.add(page.uid)
        del page
    assert doc.pages[0].uid not in seen


def test_insert_pages_restores_order():
    doc = _doc(3)
    uids = [p.uid for p in doc.pages]
    removed = doc.remove_pages([1])
    doc.insert_pages([(1, removed[0])])
    assert [p.uid for p in doc.pages] == uids


def test_pages_carry_source_name():
    doc = _doc(2)
    assert doc.pages[0].source_name == "1.jpg"
    assert doc.pages[1].source_name == "2.jpg"


def test_close_releases_appended_sources():
    doc = _doc(1)
    extra = _doc(1)
    doc.extend(extra)
    doc.close()
    assert doc._sources == []


def test_mixed_pdf_and_images_rejected(tmp_path):
    with pytest.raises(ValueError):
        Document.open(tmp_path / "x.docx")


def test_image_revision_tracks_mutation():
    """缩略图缓存靠 revision 失效：写入图像必须让它变，否则撤销后显示旧图。"""
    doc = _doc(1)
    page = doc.pages[0]
    before = page.revision
    page.image = np.zeros((10, 10, 3), np.uint8)
    assert page.revision == before + 1
