#!/usr/bin/env -S uv run --script

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QVBoxLayout,
)


class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login Dialog")

        layout = QVBoxLayout(self)
        layout.addWidget(QLineEdit("Username"))
        # Create password field with placeholder text and password echo mode
        self.password = QLineEdit()
        self.password.setPlaceholderText("Password")
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)


app = QApplication([])
dlg = LoginDialog()
dlg.exec()  # blocks until dialog closed
