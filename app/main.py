"""应用入口：python -m app.main"""

import faulthandler
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import THEME_QSS


def _enable_crash_log() -> None:
    """崩溃日志：pythonw 无控制台，致命错误写入本地文件，否则用户侧无声死机。"""
    try:
        log_dir = Path(os.environ.get("APPDATA", Path.home())) / "contract-sealer"
        log_dir.mkdir(parents=True, exist_ok=True)
        f = open(log_dir / "crash.log", "a", encoding="utf-8")
        faulthandler.enable(f)
    except OSError:
        pass  # 日志开不了也不能阻止启动


def main() -> int:
    _enable_crash_log()
    app = QApplication(sys.argv)
    app.setApplicationName("合同盖章工具")
    app.setStyle("Fusion")          # 现代扁平基座
    app.setStyleSheet(THEME_QSS)    # 自定义主题
    from PySide6.QtGui import QFont

    app.setFont(QFont("Microsoft YaHei UI", 9))  # 统一现代中文字体
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
