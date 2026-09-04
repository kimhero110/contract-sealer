"""应用入口：python -m app.main"""

import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import THEME_QSS


def main() -> int:
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
