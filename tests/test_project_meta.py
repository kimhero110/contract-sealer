"""工程元数据一致性：版本号别再三处各说各话。"""

import re
from pathlib import Path

import core

ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml 里找不到 version"
    return match.group(1)


def test_version_matches_pyproject():
    assert core.__version__ == _pyproject_version()


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", core.__version__)


def test_license_file_exists_and_is_agpl():
    """依赖 PyMuPDF（AGPL）且对外分发 exe，整个项目必须以 AGPL 授权。"""
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in text
    assert "Version 3" in text


def test_build_artifacts_not_tracked():
    """构建产物不得再混进仓库（曾经误提交过 85MB 的 cv2.pyd）。"""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("_internal/", "dist/", "build/"):
        assert pattern in gitignore, f".gitignore 缺少 {pattern}"
    assert not (ROOT / "_internal").exists()
