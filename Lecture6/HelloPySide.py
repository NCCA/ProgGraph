#!/usr/bin/env -S uv run --script

import sys

from PySide6.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel("Hello, PySide!")
label.show()
app.exec()
