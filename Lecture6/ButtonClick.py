#!/usr/bin/env -S uv run --script

import sys

from PySide6.QtWidgets import QApplication, QPushButton

app = QApplication(sys.argv)
button = QPushButton("Click Me")


def on_click():
    print("Button was clicked!")


button.clicked.connect(on_click)
button.show()
app.exec()
