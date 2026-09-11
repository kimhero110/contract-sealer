"""应用设置：跨会话记住用户选择，存 settings.json。

只存"下次还想要"的东西（输出目录、日期格式与字号），不存会话状态。
读失败一律回退默认值——设置文件损坏不该让工具打不开。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import app_data_dir

# 输出目录的特殊值：跟随源文件所在目录（默认行为）
OUTPUT_DIR_BESIDE_SOURCE = ""


def settings_path() -> Path:
    return app_data_dir() / "settings.json"


@dataclass
class Settings:
    """用户偏好。字段增删要保持向后兼容：读取时未知键忽略、缺失键用默认。"""

    output_dir: str = OUTPUT_DIR_BESIDE_SOURCE   # 空 = 与源文件同目录
    date_format: str = "%Y年%m月%d日"
    date_height_mm: float = 4.5

    @classmethod
    def load(cls, path: Path | None = None) -> Settings:
        path = path or settings_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: Path | None = None) -> Path:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def resolved_output_dir(self, source_path: Path | None) -> Path:
        """实际输出目录：设置了就用设置，否则跟随源文件，再否则当前目录。"""
        if self.output_dir:
            return Path(self.output_dir)
        if source_path is not None:
            return Path(source_path).parent
        return Path.cwd()
