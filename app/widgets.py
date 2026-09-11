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
from PySide6.QtWidgets import QPushButton, QScrollArea


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
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def minimumSizeHint(self) -> QSize:
        hint = super().minimumSizeHint()
        inner = self.widget()
        if inner is not None:
            # 竖向滚动条随时可能出现，宽度先留出来，免得一出现就挤内容
            reserve = self.verticalScrollBar().sizeHint().width() + 2 * self.frameWidth()
            hint.setWidth(inner.minimumSizeHint().width() + reserve)
        return hint
