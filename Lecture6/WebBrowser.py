#!/usr/bin/env -S uv run --script


import sys

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QToolBar,
)


def main():
    # Make an instance of the QApplication
    app = QApplication(sys.argv)

    # Create a new MainWindow
    window = QMainWindow()

    # Create a toolbar and buttons
    toolbar = QToolBar("Main Toolbar")
    back_button = QPushButton("Back")
    fwd_button = QPushButton("Forward")
    toolbar.addWidget(back_button)
    toolbar.addWidget(fwd_button)

    # Add the toolbar to the main window
    window.addToolBar(toolbar)

    # Create the web view and load a page
    page = QWebEngineView()
    page.load(QUrl("http://www.google.co.uk"))

    # Connect the button's 'clicked' signal to the page's 'back' and 'forward' slots
    back_button.clicked.connect(page.back)
    fwd_button.clicked.connect(page.forward)

    # Set the web view as the central widget
    window.setCentralWidget(page)
    window.resize(1024, 720)

    # Show the window
    window.show()

    # Hand control over to the Qt framework
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
