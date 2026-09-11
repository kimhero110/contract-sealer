"""用户数据目录（印章库 / 模板 / 设置 / 崩溃日志的唯一落点）。

严禁放 exe 相对目录：exe 可能在局域网共享盘上，印章图会裸奔内网。

各平台约定：
- Windows：%APPDATA%\\contract-sealer
- macOS：~/Library/Application Support/contract-sealer
- 其他：$XDG_CONFIG_HOME/contract-sealer，未设置则 ~/.config/contract-sealer

（曾经无条件用 %APPDATA%，非 Windows 上会退到 ~/AppData/Roaming 这种
不属于任何平台约定的路径，源码运行和测试都会落到怪地方。）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "contract-sealer"


def app_data_dir() -> Path:
    """本应用的用户数据根目录。不创建目录，调用方按需 mkdir。"""
    override = os.environ.get("CONTRACT_SEALER_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_CONFIG_HOME")
        root = Path(base) if base else Path.home() / ".config"
    return root / APP_DIR_NAME
