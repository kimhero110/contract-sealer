"""印章/签名模型与印章库。

印章库位置：%APPDATA%\\contract-sealer\\seals（方案 v1.1 §4.6）——
严禁放 exe 相对目录（exe 可能在局域网共享盘上，印章图会裸奔内网）。

每枚印章 = 透明 PNG + 元数据 JSON：
- name：显示名；
- kind：seal（印章，按直径）/ signature（签名，按宽度）；
- phys_mm：真实物理尺寸（章=直径 mm，签名=宽度 mm）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .extract import KIND_SEAL, KIND_SIGNATURE

DEFAULT_SEAL_DIAMETER_MM = 40.0   # 公章常见直径
DEFAULT_SIGNATURE_WIDTH_MM = 35.0  # 签名常用宽度


def default_library_dir() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    return root / "contract-sealer" / "seals"


@dataclass
class Seal:
    name: str
    kind: str            # KIND_SEAL / KIND_SIGNATURE
    image: np.ndarray    # RGBA uint8
    phys_mm: float       # 章=直径；签名=宽度

    def save(self, library_dir: Path) -> Path:
        library_dir.mkdir(parents=True, exist_ok=True)
        slug = _safe_slug(self.name)
        png_path = library_dir / f"{slug}.png"
        meta_path = library_dir / f"{slug}.json"
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
    def load(cls, png_path: Path) -> "Seal":
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
