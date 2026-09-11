"""印章/签名模型与印章库。

印章库位置见 core.paths.app_data_dir（方案 v1.1 §4.6）。

每枚印章 = 透明 PNG + 元数据 JSON：
- name：显示名；
- kind：seal（印章，按直径）/ signature（签名，按宽度）；
- phys_mm：真实物理尺寸（章=直径 mm，签名=宽度 mm）。

文件名由显示名转义而来，不同显示名可能撞到同一个 slug（"公章(1)" 与
"公章 1" 都变成 公章_1）。save 默认拒绝覆盖，由调用方决定改名还是
覆盖——静默顶掉用户辛苦抠出来的章是不可接受的。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .extract import KIND_SEAL, KIND_SIGNATURE
from .paths import app_data_dir

# KIND_* 由本模块再导出：调用方谈的是"印章的类型"，不该去 extract 里拿
__all__ = [
    "DEFAULT_SEAL_DIAMETER_MM",
    "DEFAULT_SIGNATURE_WIDTH_MM",
    "KIND_SEAL",
    "KIND_SIGNATURE",
    "Seal",
    "default_library_dir",
    "list_library",
    "slug_taken",
    "unique_name",
]

DEFAULT_SEAL_DIAMETER_MM = 40.0   # 公章常见直径
DEFAULT_SIGNATURE_WIDTH_MM = 35.0  # 签名常用宽度


def default_library_dir() -> Path:
    return app_data_dir() / "seals"


@dataclass
class Seal:
    name: str
    kind: str            # KIND_SEAL / KIND_SIGNATURE
    image: np.ndarray    # RGBA uint8
    phys_mm: float       # 章=直径；签名=宽度

    def save(self, library_dir: Path, overwrite: bool = False) -> Path:
        """写入印章库。目标已存在且 overwrite 为假时抛 FileExistsError。"""
        library_dir.mkdir(parents=True, exist_ok=True)
        slug = _safe_slug(self.name)
        png_path = library_dir / f"{slug}.png"
        meta_path = library_dir / f"{slug}.json"
        if not overwrite and (png_path.exists() or meta_path.exists()):
            raise FileExistsError(str(png_path))
        Image.fromarray(self.image, "RGBA").save(png_path)
        meta = {
            "name": self.name,
            "kind": self.kind,
            "phys_mm": self.phys_mm,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return png_path

    @classmethod
    def load(cls, png_path: Path) -> Seal:
        meta_path = png_path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with Image.open(png_path) as im:
            image = np.array(im.convert("RGBA"))
        return cls(
            name=meta["name"],
            kind=meta["kind"],
            image=image,
            phys_mm=float(meta["phys_mm"]),
        )


def slug_taken(library_dir: Path, name: str) -> bool:
    """该显示名对应的文件名是否已被占用。"""
    slug = _safe_slug(name)
    return (library_dir / f"{slug}.png").exists() or (library_dir / f"{slug}.json").exists()


def unique_name(library_dir: Path, name: str) -> str:
    """在显示名后追加序号，直到 slug 不再冲突。"""
    if not slug_taken(library_dir, name):
        return name
    for n in range(2, 1000):
        candidate = f"{name}-{n}"
        if not slug_taken(library_dir, candidate):
            return candidate
    return f"{name}-{int(time.time())}"


def list_library(library_dir: Path) -> list[Path]:
    if not library_dir.exists():
        return []
    return sorted(library_dir.glob("*.png"))


def _safe_slug(name: str) -> str:
    """文件名安全的标识：保留中英文数字，其余替换为 _。"""
    out = []
    for ch in name.strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or f"seal_{int(time.time())}"
