"""印章库测试：文件名转义、同名不得静默覆盖、自动改名。"""

import numpy as np
import pytest

from core.extract import KIND_SEAL
from core.seal import Seal, list_library, slug_taken, unique_name


def _seal(name: str) -> Seal:
    img = np.zeros((20, 20, 4), dtype=np.uint8)
    img[:, :, 0] = 200
    img[:, :, 3] = 255
    return Seal(name=name, kind=KIND_SEAL, image=img, phys_mm=40.0)


def test_save_and_load_round_trip(tmp_path):
    path = _seal("公章").save(tmp_path)
    loaded = Seal.load(path)
    assert loaded.name == "公章"
    assert loaded.kind == KIND_SEAL
    assert loaded.phys_mm == 40.0
    assert loaded.image.shape == (20, 20, 4)


def test_save_refuses_to_overwrite(tmp_path):
    _seal("公章").save(tmp_path)
    with pytest.raises(FileExistsError):
        _seal("公章").save(tmp_path)


def test_distinct_names_colliding_on_slug_do_not_clobber(tmp_path):
    """「公章(1)」和「公章 1」转义后都是 公章_1——第二个必须被拦住。"""
    _seal("公章(1)").save(tmp_path)
    assert slug_taken(tmp_path, "公章 1")
    with pytest.raises(FileExistsError):
        _seal("公章 1").save(tmp_path)
    assert len(list_library(tmp_path)) == 1


def test_overwrite_when_explicitly_allowed(tmp_path):
    _seal("公章").save(tmp_path)
    replacement = _seal("公章")
    replacement.phys_mm = 42.0
    path = replacement.save(tmp_path, overwrite=True)
    assert Seal.load(path).phys_mm == 42.0
    assert len(list_library(tmp_path)) == 1


def test_unique_name_avoids_collision(tmp_path):
    _seal("公章").save(tmp_path)
    new_name = unique_name(tmp_path, "公章")
    assert new_name != "公章"
    _seal(new_name).save(tmp_path)
    assert len(list_library(tmp_path)) == 2


def test_unique_name_is_identity_when_free(tmp_path):
    assert unique_name(tmp_path, "法人章") == "法人章"
