"""设置与用户数据目录测试：输出目录可选、设置文件损坏不能让工具打不开。"""

import json
import sys
from pathlib import Path

from core.paths import app_data_dir
from core.settings import Settings


def test_defaults_are_beside_source():
    s = Settings()
    assert s.output_dir == ""
    assert s.resolved_output_dir(Path("/tmp/合同/a.pdf")) == Path("/tmp/合同")


def test_explicit_output_dir_wins():
    s = Settings(output_dir="/tmp/盖章输出")
    assert s.resolved_output_dir(Path("/tmp/合同/a.pdf")) == Path("/tmp/盖章输出")


def test_no_source_falls_back_to_cwd():
    assert Settings().resolved_output_dir(None) == Path.cwd()


def test_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    Settings(output_dir=str(tmp_path), date_format="%Y-%m-%d", date_height_mm=6.0).save(path)
    loaded = Settings.load(path)
    assert loaded.output_dir == str(tmp_path)
    assert loaded.date_format == "%Y-%m-%d"
    assert loaded.date_height_mm == 6.0


def test_missing_file_gives_defaults(tmp_path):
    assert Settings.load(tmp_path / "nope.json") == Settings()


def test_corrupt_file_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{ 这不是 json", encoding="utf-8")
    assert Settings.load(path) == Settings()


def test_unknown_keys_ignored(tmp_path):
    """老版本写的、或未来版本新增的键不能让读取炸掉。"""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"output_dir": "/tmp/x", "从未见过的键": 1}), encoding="utf-8")
    assert Settings.load(path).output_dir == "/tmp/x"


def test_app_data_dir_follows_platform_convention(monkeypatch):
    monkeypatch.delenv("CONTRACT_SEALER_HOME", raising=False)
    d = app_data_dir()
    assert d.name == "contract-sealer"
    if sys.platform == "win32":
        assert "Roaming" in str(d) or "AppData" in str(d)
    elif sys.platform == "darwin":
        assert "Application Support" in str(d)
    else:
        # 非 Windows 上绝不能退到 ~/AppData/Roaming 这种不属于本平台的路径
        assert "AppData" not in str(d)


def test_app_data_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("CONTRACT_SEALER_HOME", str(tmp_path))
    assert app_data_dir() == tmp_path
