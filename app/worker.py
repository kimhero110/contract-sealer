"""把耗时任务挪出主线程：导出与批量盖章期间界面照常响应，还能取消。

用法：
    result = run_with_progress(parent, "导出 PDF", lambda progress, cancelled: session.export(...))

fn 在工作线程里执行，收两个回调：progress(done, total, text) 与 cancelled() -> bool，
与 core.session 的 ProgressFn / CancelFn 同签名。主线程阻塞在一个局部事件循环里
等结果——调用方的代码保持同步写法，界面却不冻结。fn 抛出的异常在调用方线程重新抛出。

为什么不是"一切都异步"：主窗口的状态机（撤销栈、画布同步）建立在"一次一个动作"
之上；进度对话框是应用模态的，期间用户碰不到主窗口，会话状态不会被并发修改。
工作线程只**读**会话（渲染页面、写文件），不改它。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QEventLoop, QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import QProgressDialog, QWidget

from core.session import ExportCancelled

TaskFn = Callable[[Callable[[int, int, str], None], Callable[[], bool]], Any]


class _Worker(QObject):
    progressed = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(object)

    def __init__(self, fn: TaskFn):
        super().__init__()
        self._fn = fn
        self._cancel = False

    @Slot()
    def cancel(self) -> None:
        self._cancel = True

    @Slot()
    def run(self) -> None:
        try:
            result = self._fn(self.progressed.emit, lambda: self._cancel)
        except BaseException as e:  # 任何异常都得带回主线程，否则线程静默死掉
            self.failed.emit(e)
            return
        self.succeeded.emit(result)


class _Receiver(QObject):
    """住在主线程的接收端：工作线程发来的信号经队列投递，槽在主线程执行。

    直接把 lambda 连到跨线程信号，槽会在**发射方线程**里跑——那就是在工作线程里
    改进度对话框、退出主线程的事件循环，轻则卡死重则崩溃。有 QObject 接收方，
    Qt 才知道该排队送回主线程。
    """

    def __init__(self, dialog: QProgressDialog, loop: QEventLoop):
        super().__init__()
        self._dialog = dialog
        self._loop = loop
        self.outcome: dict[str, Any] = {}

    @Slot(int, int, str)
    def on_progress(self, done: int, total: int, text: str) -> None:
        self._dialog.setMaximum(max(total, 1))
        self._dialog.setValue(min(done, max(total, 1)))
        self._dialog.setLabelText(text)

    @Slot(object)
    def on_success(self, result: Any) -> None:
        self.outcome["result"] = result
        self._loop.quit()

    @Slot(object)
    def on_failure(self, error: BaseException) -> None:
        self.outcome["error"] = error
        self._loop.quit()


def run_with_progress(
    parent: QWidget | None,
    title: str,
    fn: TaskFn,
    cancel_text: str = "取消",
) -> Any:
    """在工作线程跑 fn，主线程显示可取消的进度对话框并等待结果。

    用户点取消 → cancelled() 变 True，fn 应尽快抛 ExportCancelled（本函数原样抛出）。
    """
    dialog = QProgressDialog(title, cancel_text, 0, 1, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)

    loop = QEventLoop()
    receiver = _Receiver(dialog, loop)
    thread = QThread()
    worker = _Worker(fn)
    worker.moveToThread(thread)
    queued = Qt.ConnectionType.QueuedConnection
    worker.progressed.connect(receiver.on_progress, queued)
    worker.succeeded.connect(receiver.on_success, queued)
    worker.failed.connect(receiver.on_failure, queued)
    dialog.canceled.connect(worker.cancel, Qt.ConnectionType.DirectConnection)  # 只改一个布尔标志，直连即可
    thread.started.connect(worker.run)

    thread.start()
    dialog.show()
    loop.exec()
    thread.quit()
    thread.wait()
    dialog.close()
    dialog.deleteLater()

    if "error" in receiver.outcome:
        raise receiver.outcome["error"]
    return receiver.outcome.get("result")


__all__ = ["ExportCancelled", "run_with_progress"]
