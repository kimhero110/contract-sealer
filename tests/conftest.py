"""测试夹具。

素材（合成扫描件、印章、签名）不入库，由 `fixtures/generate_fixtures.py` 按需生成。
不入库的理由不只是 7.6MB 体积：它们是确定性合成的产物，入库等于把可生成物
当成源码——改了生成脚本还得记得把图重新提交一遍，迟早不同步。
真实合同与真章更是永远不该出现在仓库里。

生成放在 conftest **导入时**而不是 fixture 里：conftest 先于测试模块被导入，
这样即便某个模块在导入期就读素材也来得及。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
_REQUIRED = ("1.jpg", "2.jpg", "3.jpg", "seal_company.png", "seal_person.png",
             "sig_hxd.png", "sig_lsl.png")


def _ensure_fixtures() -> None:
    if all((FIXTURES / name).exists() for name in _REQUIRED):
        return
    # 子进程的 stdout 是管道时，Python 按本地代码页编码输出（英文 Windows 上是 cp1252），
    # 生成脚本里的中文提示会直接把它撑成 UnicodeEncodeError。强制 UTF-8。
    proc = subprocess.run(
        [sys.executable, str(FIXTURES / "generate_fixtures.py")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if proc.returncode != 0:
        # 光报 CalledProcessError 只能看到一句退出码，真正的原因在子进程输出里
        raise RuntimeError(
            "生成测试素材失败：\n" + (proc.stdout or "") + (proc.stderr or "")
        )
    missing = [n for n in _REQUIRED if not (FIXTURES / n).exists()]
    if missing:  # 生成脚本改了名字却没改这里，不能让它变成一堆看不懂的读文件失败
        raise RuntimeError(f"素材生成后仍缺少：{missing}")


_ensure_fixtures()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def scan_jpg(fixtures_dir) -> Path:
    return fixtures_dir / "1.jpg"


@pytest.fixture(scope="session")
def seal_png(fixtures_dir) -> Path:
    return fixtures_dir / "seal_company.png"


@pytest.fixture(scope="session")
def signature_png(fixtures_dir) -> Path:
    return fixtures_dir / "sig_lsl.png"
