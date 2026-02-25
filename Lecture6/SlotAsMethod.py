#!/usr/bin/env -S uv run --script
#
import time

from PySide6.QtCore import QObject, Signal, Slot


class Worker(QObject):
    done = Signal()

    @Slot()  # explicitly marks this as a slot
    def work(self):
        print("Working...")
        time.sleep(2)  # simulate work
        self.done.emit()

    @Slot()
    def finish(self):
        print("Work finished!")


w = Worker()
w.done.connect(w.finish)
w.work()
