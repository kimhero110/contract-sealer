"""共用控件。

`FitButton` 存在的唯一理由：Qt 默认允许按钮一路缩到 0 宽，文字被省略成
"↻ 重盖选…"。`QPushButton` 没有重写 `minimumSizeHint`，`QWidget` 的默认实现
返回无效尺寸，布局于是认为按钮的宽度下限是 0——窗口一旦没有最大化、右侧面板
被分割条挤窄，按钮就先于其他控件让位，文字被吃掉。

把 `minimumSizeHint` 钉在 `sizeHint` 上就从根上堵死了这条路：下限会沿布局链
一路上报给 `QSplitter`，面板因此也拖不到比内容更窄。注意必须重写方法而不是
在构造时 `setMinimumWidth(sizeHint().width())`——后者在 QSS 生效（polish）前
取值会漏掉 padding，而重写是布局时惰性求值，样式表、字体、文案改了都自动跟上。
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QPushButton, QScrollArea


class FitButton(QPushButton):
    """文字永不被省略号吃掉的按钮。"""

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()


class VScrollArea(QScrollArea):
    """只竖向滚动的滚动区，横向宽度需求如实上报。

    普通 QScrollArea 的 minimumSizeHint 与内容无关——这正是它的用途：
    多窄都能塞下，滚动就是了。但右侧面板只想在窗口变矮时竖向滚动，
    横向一旦被挤窄，按钮文字照样被切。这里把内容的最小宽度（加上竖向
    滚动条的位置）报上去，最小宽度就能沿布局链传到 QSplitter，
    面板既拖不窄、也不会冒出横向滚动条。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        inner = self.widget()
        if inner is not None:
            # 竖向滚动条随时可能出现，宽度先留出来，免得一出现就挤内容
            reserve = self.verticalScrollBar().sizeHint().width() + 2 * self.frameWidth()
            hint.setWidth(inner.minimumSizeHint().width() + reserve)
        return hint


class ElidedLabel(QLabel):
    """文字过长时中间省略，而不是把父容器撑宽。

    `QLabel` 即使开了 `wordWrap` 也只在空格处断行。像 Windows 上的印章库路径
    `C:\\Users\\<用户名>\\AppData\\Roaming\\contract-sealer\\seals` 整条没有空格，
    断不开，`minimumSizeHint` 就等于整行文字宽度——右侧面板被它一路撑到 500px 开外，
    画布跟着变窄，按钮反倒先被挤得显示不全。Linux 上路径短，这个坑照不出来。

    这里自己画省略后的文字，最小宽度只报一个下限；完整内容进 tooltip，不丢信息。
    """

    def __init__(self, text: str = "", parent=None, min_width: int = 72):
        super().__init__(text, parent)
        self._full = text
        self._min_width = min_width
        self.setToolTip(text)

    def setText(self, text: str) -> None:
        self._full = text
        self.setToolTip(text)
        super().setText(text)
        self.update()

    def minimumSizeHint(self) -> QSize:
        return QSize(self._min_width, self.fontMetrics().height() + 2)

    def sizeHint(self) -> QSize:
        # 与最小值一致：让它填满可用宽度即可，不要反过来去决定容器该多宽
        return self.minimumSizeHint()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setPen(self.palette().color(self.foregroundRole()))
        text = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideMiddle, self.width())
        painter.drawText(self.rect(), int(self.alignment()) | int(Qt.AlignmentFlag.AlignVCenter), text)
