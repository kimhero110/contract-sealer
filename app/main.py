"""应用入口：python -m app.main"""

import faulthandler
import sys
from pathlib import Path

from PySide6.QtGui import QFont, QIcon
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


def main() -> int:
    _enable_crash_log()
    app = QApplication(sys.argv)
    app.setApplicationName("合同盖章工具")
    app.setStyle("Fusion")          # 现代扁平基座
    app.setStyleSheet(THEME_QSS)    # 自定义主题
    app.setFont(QFont("Microsoft YaHei UI", 9))  # 统一现代中文字体
    # 应用图标（源码运行即生效；exe 图标由 spec 的 icon= 打进文件）
    icon_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent)) / "docs" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
