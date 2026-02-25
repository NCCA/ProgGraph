#!/usr/bin/env -S uv run --script

import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QMainWindow Example")

        # central widget is mandatory
        central = QWidget(self)
        layout = QVBoxLayout(central)

        layout.addWidget(QPushButton("Button 1"))
        layout.addWidget(QPushButton("Button 2"))

        self.setCentralWidget(central)


app = QApplication(sys.argv)
win = MainWindow()
win.show()
app.exec()
