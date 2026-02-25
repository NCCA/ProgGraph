from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QResizeEvent
from PySide6.QtWidgets import QWidget


class QWidgetABCMeta(ABCMeta, type(QWidget)):
    """
    A metaclass that combines the functionality of ABCMeta and QWidget's metaclass.

    This allows the creation of abstract base classes that are also QWidgets.
    """

    pass


class NumpyBufferWidget(QWidget, metaclass=QWidgetABCMeta):
    """
    An abstract base class for widgets that render a NumPy buffer.

    This class provides a simple way to generate a numpy buffer and render it to the screen
    using PySide6. It's designed to be subclassed for specific rendering implementations,
    such as WebGPU or Vulkan.

    Attributes:
        initialized (bool): A flag indicating whether the widget has been initialized.
                            This ensures that initialization logic is run only once.
        text_buffer (List[Tuple[int, int, str, int, str, QColor]]): A buffer to hold text
                                                                    to be rendered on the widget.
        buffer (Optional[np.ndarray]): The numpy array that holds the image data to be displayed.
    """

    def __init__(self) -> None:
        """
        Initialize the NumpyBufferWidget.

        This constructor initializes the QWidget, sets up internal state, and prepares
        the update timer.
        """
        super().__init__()
        self.initialized: bool = False
        self.text_buffer: List[Tuple[int, int, str, int, str, QColor]] = []
        self.buffer: Optional[np.ndarray] = None
        self._update_timer: QTimer = QTimer(self)
        self._update_timer.timeout.connect(self.update)
        self._flipped: bool = False

    def start_update_timer(self, interval_ms: int) -> None:
        """
        Starts a timer which triggers a repaint of the widget periodically.

        Args:
            interval_ms (int): The interval in milliseconds at which to update the widget.
        """
        self._update_timer.start(interval_ms)

    def stop_update_timer(self) -> None:
        """Stops the periodic update timer."""
        self._update_timer.stop()

    @abstractmethod
    def initialize_buffer(self) -> None:
        """
        Abstract method to initialize the rendering buffer and context.

        This method must be implemented in subclasses. It is called once, before the first
        paint event, to set up any necessary resources, such as a WebGPU context or
        the numpy buffer itself.
        """
        pass

    @abstractmethod
    def paint(self) -> None:
        """
        Abstract method for custom painting logic.

        This method must be implemented in subclasses. It is called on every paint event
        and should contain the core rendering logic that updates the content of `self.buffer`.
        """
        pass

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Handles the widget's paint event.

        This method orchestrates the rendering process. It ensures that the buffer is
        initialized, calls the custom `paint` method, and then draws the resulting
        numpy buffer and any buffered text onto the widget.

        Args:
            event (QPaintEvent): The paint event object.
        """
        # On the first paint event, initialize the buffer.
        if not self.initialized:
            self.initialize_buffer()
            self.initialized = True

        # Call the subclass's painting logic to update the buffer.
        self.paint()

        # Create a QPainter to draw on the widget.
        painter = QPainter(self)

        # If the buffer has been created, draw it.
        if self.buffer is not None:
            self._present_image(painter, self.buffer)

        # Draw any text that has been added to the text buffer.
        for x, y, text, size, font, colour in self.text_buffer:
            painter.setPen(colour)
            painter.setFont(QFont(font, size))
            painter.drawText(x, y, text)
        # Clear the text buffer after drawing so it's not drawn again on the next frame.
        self.text_buffer.clear()

        super().paintEvent(event)

    def render_text(
        self,
        x: int,
        y: int,
        text: str,
        size: int = 10,
        font: str = "Arial",
        colour: QColor = Qt.GlobalColor.black,
    ) -> None:
        """
        Adds text to a buffer to be rendered on the widget during the next paint event.

        Args:
            x (int): The x-coordinate for the text's top-left corner.
            y (int): The y-coordinate for the text's top-left corner.
            text (str): The text to render.
            size (int, optional): The font size. Defaults to 10.
            font (str, optional): The font family. Defaults to "Arial".
            colour (QColor, optional): The color of the text. Defaults to black.
        """
        self.text_buffer.append((x, y, text, size, font, colour))

    def resizeEvent(self, event: QResizeEvent) -> None:
        """
        Handles the widget's resize event.

        This can be overridden in subclasses to handle changes in widget size,
        for example, to resize the rendering buffer.

        Args:
            event (QResizeEvent): The resize event object.
        """
        # The default implementation calls the parent class's method.
        super().resizeEvent(event)

    def _present_image(self, painter: QPainter, image_data: np.ndarray) -> None:
        """
        Presents the numpy array image data on the widget.

        This internal method converts the numpy array into a QImage and draws it
        onto the widget, scaling it to fit the widget's dimensions.

        Args:
            painter (QPainter): The QPainter to use for drawing.
            image_data (np.ndarray): The image data to render, expected to be in
                                     a format compatible with QImage.Format_RGBX8888
                                     (e.g., a 3D array with shape (height, width, 4)).
        """
        height, width, _ = image_data.shape
        # Create a QImage from the numpy array data.
        # The data is not copied, so the numpy array must not be changed
        # while the QImage is in use.
        image = QImage(
            image_data.data,
            width,
            height,
            width * 4,  # Bytes per line
            QImage.Format.Format_RGBX8888,
        )
        # Flip the image horizontally if the flipped flag is set.
        if self._flipped:
            image.mirrored(horizontal=True, vertical=False)

        # Define source and target rectangles for drawing.
        source_rect = QRect(0, 0, width, height)
        target_rect = self.rect()  # The entire widget area.

        # Draw the image, scaling it to fit the target rectangle.
        painter.drawImage(target_rect, image, source_rect)
