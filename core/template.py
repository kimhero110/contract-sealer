"""落款模板：把一页上的盖章组合保存为可复用模板（M3）。

模板记录每枚章的：印章名/类型、相对页面位置（0-1 比例，跨页面尺寸可复用）、
物理尺寸、旋转、不透明度。套用模板时按目标页尺寸还原位置。
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def default_template_dir() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    return root / "contract-sealer" / "templates"


def save_template(template_dir: Path, name: str, entries: list[dict]) -> Path:
    """entries: [{seal_name, kind, rel_x, rel_y, size_mm, rotation_deg, opacity}]"""
    template_dir.mkdir(parents=True, exist_ok=True)
    if not entries:
        raise ValueError("模板为空：当前页没有任何盖章")
    path = template_dir / f"{_safe(name)}.json"
    path.write_text(
        json.dumps({"name": name, "entries": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_template(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["entries"]


def list_templates(template_dir: Path) -> list[Path]:
    if not template_dir.exists():
        return []
    return sorted(template_dir.glob("*.json"))


def _safe(name: str) -> str:
    out = [ch if (ch.isalnum() or ch in "-_") else "_" for ch in name.strip()]
    return "".join(out).strip("_") or "template"
