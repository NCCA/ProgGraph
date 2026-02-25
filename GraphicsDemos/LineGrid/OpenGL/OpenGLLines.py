#!/usr/bin/env -S uv run --script

"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import argparse
import sys
import traceback

import numpy as np
import OpenGL.GL as gl
from ncca.ngl import Mat4, ShaderLib, Vec3, look_at, perspective
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication


class MainWindow(QOpenGLWindow):
    """
    The main window for the OpenGL application.

    Inherits from QOpenGLWindow to provide a canvas for OpenGL rendering within a PySide6 GUI.
    It handles user input (mouse, keyboard) for camera control and manages the OpenGL context.
    """

    def __init__(self, parent: object = None) -> None:
        """
        Initializes the main window and sets up default scene parameters.
        """
        super().__init__()
        # --- Camera and Transformation Attributes ---
        self.setTitle("Render Lines OpenGL (Core Profile)")
        self.window_width = 1024
        self.window_height = 720
        self.rotation = 0.0
        self.view = look_at(Vec3(0, 6, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))

    def initializeGL(self) -> None:
        """
        Called once when the OpenGL context is first created.
        This is the place to set up global OpenGL state, load shaders, and create geometry.
        """
        self.makeCurrent()  # Make the OpenGL context current in this thread
        # Set the background colour to a dark grey
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        # Enable depth testing, which ensures that objects closer to the camera obscure those further away
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Enable multisampling for anti-aliasing, which smooths jagged edges
        gl.glEnable(gl.GL_MULTISAMPLE)
        # Set up the camera's view matrix.
        ShaderLib.load_shader("LineShader", "LineVertex.glsl", "LineFragment.glsl")
        ShaderLib.use("LineShader")
        self._create_lines(10, 10, 30, 30)
        self.startTimer(16)

    def _create_lines(self, width: float, depth: float, rows: int, cols: int) -> None:
        points = []
        # get step values for x,z
        col_step = width / cols
        row_step = depth / rows
        x = width * 0.5  # half extents
        z = -(depth * 0.5)
        for _ in range(rows + 1):
            points.append(-x)
            points.append(0.0)
            points.append(z)
            points.append(x)
            points.append(0.0)
            points.append(z)
            z += row_step
        z = depth * 0.5
        x = -(width * 0.5)
        for _ in range(rows + 1):
            points.append(x)
            points.append(0.0)
            points.append(-z)
            points.append(x)
            points.append(0.0)
            points.append(z)
            x += col_step

        # 1. Create and bind a Vertex Array Object (VAO) to store all the state
        #    for our geometry.
        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)
        # 3. Create a Vertex Buffer Objects (VBOs) to hold the data on the GPU.
        vbo_id = gl.glGenBuffers(1)
        # 4. Configure the first VBO for vertex positions.
        points = np.array(points, dtype=np.float32)
        self.line_vertex_count = len(points) // 4 * 3  # 4 vertices per line and 3 components
        print(self.line_vertex_count)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, points.nbytes, points, gl.GL_STATIC_DRAW)
        # Set up vertex attribute pointer 0 (for "inPosition" in the vertex shader).
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        gl.glEnableVertexAttribArray(0)

    def paintGL(self) -> None:
        """
        Called every time the window needs to be redrawn.
        This is the main rendering loop where all drawing commands are issued.
        """
        self.makeCurrent()
        # Set the viewport to cover the entire window
        gl.glViewport(0, 0, self.window_width, self.window_height)
        # Clear the colour and depth buffers from the previous frame
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        rotation_matrix = Mat4.rotate_y(self.rotation)
        mvp = self.projection @ self.view @ rotation_matrix

        ShaderLib.set_uniform("MVP", mvp)
        gl.glBindVertexArray(self.vao_id)
        gl.glDrawArrays(gl.GL_LINES, 0, self.line_vertex_count)

    def resizeGL(self, w: int, h: int) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
            w: The new width of the window.
            h: The new height of the window.
        """
        # Update the stored width and height, considering high-DPI displays
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())
        self.projection = perspective(45.0, self.window_width / self.window_height, 0.1, 100.0)
        # Update the projection matrix to match the new aspect ratio.
        # This creates a perspective projection with a 45-degree field of view.

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()  # Exit the application

        self.update()

        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def timerEvent(self, event) -> None:
        """
        Handles timer events.

        Args:
            event: The QTimerEvent object containing information about the timer event.
        """
        self.rotation += 0.5
        self.update()


class DebugApplication(QApplication):
    """
    A custom QApplication subclass for improved debugging.

    By default, Qt's event loop can suppress exceptions that occur within event handlers
    (like paintGL or mouseMoveEvent), making it very difficult to debug as the application
    may simply crash or freeze without any error message. This class overrides the `notify`
    method to catch these exceptions, print a full traceback to the console, and then
    re-raise the exception to halt the program, making the error immediately visible.
    """

    def __init__(self):
        super().__init__()

    def notify(self, receiver, event):
        """
        Overrides the central event handler to catch and report exceptions.
        """
        try:
            # Attempt to process the event as usual
            return super().notify(receiver, event)
        except Exception:
            # If an exception occurs, print the full traceback
            traceback.print_exc()
            # Re-raise the exception to stop the application
            raise


if __name__ == "__main__":
    # --- Application Entry Point ---

    # Create a QSurfaceFormat object to request a specific OpenGL context
    format: QSurfaceFormat = QSurfaceFormat()
    # Request 4x multisampling for anti-aliasing
    format.setSamples(4)
    # Request OpenGL version 4.1 as this is the highest supported on macOS
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    # Request a Core Profile context, which removes deprecated, fixed-function pipeline features
    format.setProfile(QSurfaceFormat.CoreProfile)
    # Request a 24-bit depth buffer for proper 3D sorting
    format.setDepthBufferSize(24)
    # Set default format for all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    # Apply this format to all new OpenGL contexts
    QSurfaceFormat.setDefaultFormat(format)

    """
    Main function to run the application.
    Parses command line arguments and initializes the WebGPUScene.
    """
    parser = argparse.ArgumentParser(description="A WebGPU Lines demo")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )
    args = parser.parse_args()

    # Check for a "--debug" command-line argument to run the DebugApplication
    if args.debug:
        print("Debug mode enabled")
        app = DebugApplication()
    else:
        print("Running in normal mode")
        app = QApplication()

    # Create the main window
    window = MainWindow()
    window.resize(1024, 720)
    # Show the window
    window.show()
    # Start the application's event loop
    sys.exit(app.exec())
