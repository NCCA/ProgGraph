#!/usr/bin/env -S uv run --script

import sys

from PySide6.QtCore import QRect
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QWidget,
)


def main():
    # Create the main Qt app
    app = QApplication(sys.argv)

    # Create a main window widget
    mainwin = QMainWindow()
    mainwin.setObjectName("MainWindow")
    mainwin.resize(200, 200)
    mainwin.setWindowTitle("A MainWindow App")

    # Create a central widget. The main window will be its parent.
    centralwidget = QWidget()

    # Create a push button with the central widget as its parent
    button = QPushButton("Button", parent=centralwidget)
    button.setObjectName("button")
    button.setGeometry(QRect(10, 80, 100, 32))

    # Set the central widget for the main window
    mainwin.setCentralWidget(centralwidget)

    # Show the window
    mainwin.show()

    # Run the application's event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
