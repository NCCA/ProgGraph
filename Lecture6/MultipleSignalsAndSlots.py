#!/usr/bin/env -S uv run --script

from PySide6.QtCore import QObject, Signal


class Announcer(QObject):
    shout = Signal(str)


a = Announcer()

a.shout.connect(lambda msg: print("Listener 1:", msg))
a.shout.connect(lambda msg: print("Listener 2:", msg))

a.shout.emit("Hello Students")
