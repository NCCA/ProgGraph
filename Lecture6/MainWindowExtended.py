#!/usr/bin/env -S uv run --script

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMainWindow


class MainWindow(QMainWindow):
    """
    A custom main window that overrides event handlers for key presses and resizing.
    """

    def __init__(self, parent=None):
        """
        Constructor for the MainWindow.
        """
        super().__init__(parent)
        self.resize(1024, 720)
        self.setWindowTitle("Extending a MainWindow Class")

    def keyPressEvent(self, event):
        """
        Overrides the keyPressEvent to handle key presses.
        This method is called every time the window receives a key event.
        """
        # If the 'Escape' key is pressed, exit the application.
        if event.key() == Qt.Key.Key_Escape:
            QApplication.instance().quit()
        # Otherwise, pass the event to the base class
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """
        Overrides the resizeEvent to handle window resizing.
        This method is called every time the window size changes.
        """
        # Get the new size from the event
        size = event.size()
        # Create a new title string using an f-string for formatting
        title = f"Extending a MainWindow Class size is {size.width()} {size.height()}"
        # Set the new window title
        self.setWindowTitle(title)
        # Call the base class's resizeEvent
        super().resizeEvent(event)


if __name__ == "__main__":
    # Create the application instance
    app = QApplication(sys.argv)
    # Create and show our custom main window
    window = MainWindow()
    window.show()
    # Start the application's event loop
    sys.exit(app.exec())
