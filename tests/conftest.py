from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


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
