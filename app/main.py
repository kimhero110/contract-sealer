"""应用入口：python -m app.main"""

import faulthandler
import sys
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import THEME_QSS
from core.paths import app_data_dir


def _enable_crash_log() -> None:
    """崩溃日志：pythonw 无控制台，致命错误写入本地文件，否则用户侧无声死机。"""
    try:
        log_dir = app_data_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        faulthandler.enable(open(log_dir / "crash.log", "a", encoding="utf-8"))
    except OSError:
        pass  # 日志开不了也不能阻止启动


# 各平台的中文界面字体候选，按优先级排。都不在时不动系统默认字体——
# 指定一个不存在的字体名，Qt 会静默换成某个回退字体，中文可能变成宋体或豆腐块。
_UI_FONT_CANDIDATES = {
    "win32": ["Microsoft YaHei UI", "Microsoft YaHei"],
    "darwin": ["PingFang SC", "Hiragino Sans GB"],
}
_UI_FONT_FALLBACK = ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Micro Hei"]


def pick_ui_font(available: set[str], platform: str = sys.platform) -> str | None:
    """从本机已安装字体里挑一个合适的中文界面字体；没有合适的返回 None。"""
    for name in _UI_FONT_CANDIDATES.get(platform, []) + _UI_FONT_FALLBACK:
        if name in available:
            return name
    return None


def main() -> int:
    _enable_crash_log()
    app = QApplication(sys.argv)
    app.setApplicationName("合同盖章工具")
    app.setStyle("Fusion")          # 现代扁平基座
    app.setStyleSheet(THEME_QSS)    # 自定义主题
    family = pick_ui_font(set(QFontDatabase.families()))
    if family:
        app.setFont(QFont(family, 9))  # 统一现代中文字体（按平台选，缺失则沿用系统默认）
    # 应用图标（源码运行即生效；exe 图标由 spec 的 icon= 打进文件）
    icon_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "docs" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
