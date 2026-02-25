#!/usr/bin/env -S uv run --script

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QLineEdit,
)


class FormDialog(QDialog):
    def __init__(self):
        super().__init__()
        grid = QGridLayout(self)

        grid.addWidget(QLabel("Name:"), 0, 0)
        grid.addWidget(QLineEdit(), 0, 1)

        grid.addWidget(QLabel("Email:"), 1, 0)
        grid.addWidget(QLineEdit(), 1, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        grid.addWidget(buttons, 2, 0, 1, 2)


app = QApplication([])
dlg = FormDialog()
dlg.exec()  # blocks until dialog closed
