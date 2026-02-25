#!/usr/bin/env -S uv run --script

from PySide6.QtCore import QObject, Signal


class Communicator(QObject):
    # Define a signal carrying a string
    speak = Signal(str)


def listener(msg):
    print("Listener received:", msg)


c = Communicator()
c.speak.connect(listener)  # connect signal to slot
c.speak.emit("Hello World")  # emit signal
