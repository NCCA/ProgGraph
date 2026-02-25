#!/usr/bin/env -S uv run --script

"""
A template for creating a PySide6 application with an OpenGL viewport using py-ngl.

This script sets up a basic window, initializes an OpenGL context, and provides
standard mouse and keyboard controls for interacting with a 3D scene (rotate, pan, zoom).
It is designed to be a starting point for more complex OpenGL applications.
"""

import argparse
import ctypes
import math
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
        self.setTitle("Render Points OpenGL (Core Profile)")
        self.window_width = 1024
        self.window_height = 1024
        self.rotation = 0.0
        self.offset = 0.0
        self.view = look_at(Vec3(0, 10, 20), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.animate = True
        self.rotate = True

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
        self._create_triangle_plane(20, 20, 60, 60, Vec3(0, 1, 0))
        self.startTimer(16)

    def _create_triangle_plane(self, width: float, depth: float, width_step: int, depth_step: int, normal: Vec3):
        # create a structure for a vertex data layout
        vert_dtype = np.dtype([
            ("x", np.float32, (1)),
            ("y", np.float32, (1)),
            ("z", np.float32, (1)),
            ("nx", np.float32, (1)),
            ("ny", np.float32, (1)),
            ("nz", np.float32, (1)),
            ("u", np.float32, (1)),
            ("v", np.float32, (1)),
        ])
        # create a vert structure and set the normal which in this case will always be the same
        vert = np.array((0.0, 0.0, 0.0, normal.x, normal.y, normal.z, 0.0, 0.0), dtype=vert_dtype)
        # create an empty vertex array for our data
        # We can pre-calculate how many vertex entries we need as it will be width_step  * depth_step * 6 as we
        # have two triangles for each plane segment. this is quicker than appending to a buffer
        vertex_data = np.empty(width_step * depth_step * 6, dtype=vert_dtype)
        w2 = width / 2
        d2 = depth / 2
        u = 0.0
        v = 0.0
        du = 1.0 / width_step
        dv = 1.0 / depth_step
        w_step = width / width_step
        d_step = depth / depth_step
        vertex_index = 0

        for d in np.arange(-d2, d2, d_step):
            for w in np.arange(-w2, w2, w_step):
                #       // counter clock wise
                #       3
                #       | \
                #       |  \
                #       |   \
                #       1____2
                vert["u"] = u
                vert["v"] = v + dv
                vert["x"] = w
                vert["z"] = d + d_step
                vertex_data[vertex_index] = vert
                vertex_index += 1
                # 2
                vert["u"] = u + du
                vert["v"] = v + dv
                vert["x"] = w + w_step
                vert["z"] = d + d_step
                vertex_data[vertex_index] = vert
                vertex_index += 1
                # 3
                vert["u"] = u
                vert["v"] = v
                vert["x"] = w
                vert["z"] = d
                vertex_data[vertex_index] = vert
                vertex_index += 1

                #       /* tri 2 w,0,d
                #       // counter clock wise
                #       3_____2
                #       \    |
                #       \  |
                #       \ |
                #       \|
                #       1
                vert["u"] = u + du
                vert["v"] = v + dv
                vert["x"] = w + w_step
                vert["z"] = d + d_step
                vertex_data[vertex_index] = vert
                vertex_index += 1
                # 4
                vert["u"] = u + du
                vert["v"] = v
                vert["x"] = w + w_step
                vert["z"] = d
                vertex_data[vertex_index] = vert
                vertex_index += 1
                # 5
                vert["u"] = u
                vert["v"] = v
                vert["x"] = w
                vert["z"] = d
                vertex_data[vertex_index] = vert
                vertex_index += 1
                u += du
            u = 0.0
            v += dv

        # 1. Create and bind a Vertex Array Object (VAO) to store all the state
        #    for our geometry.
        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)
        # 3. Create a Vertex Buffer Objects (VBOs) to hold the data on the GPU.
        self.vbo_id = gl.glGenBuffers(1)
        # 4. Configure the first VBO for vertex positions.
        self.tri_count = len(vertex_data)
        self.buffer_size = vertex_data.nbytes
        self.vert_dtype = vert_dtype
        print(self.tri_count)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        vertex_data = vertex_data.flatten()
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.buffer_size, vertex_data, gl.GL_DYNAMIC_DRAW)
        # Set up vertex attribute pointer 0 (for "inPosition" in the vertex shader).
        # In this setup we are using vert.nbytes to indicate the stride between vertices.
        # We then offset from a void the number of floats into the structure.
        # In C void is the same size as a float32 so this makes the calculation easier for us
        # we need to calculate the size of a float float_size = ctypes.sizeof(ctypes.c_float())
        # ctypes.c_void_p(0) is the start of the current buffer.
        # ctypes.c_void_p(3) * float_size is 3 floats in so in this case nx.
        # ctypes.c_void_p(6) * float_size is 6 floats in so the u value.
        float_size = ctypes.sizeof(ctypes.c_float())
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, vert.nbytes, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, vert.nbytes, ctypes.c_void_p(3 * float_size))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, vert.nbytes, ctypes.c_void_p(6 * float_size))
        gl.glEnableVertexAttribArray(2)

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
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.tri_count)

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

        elif key == Qt.Key_W:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        elif key == Qt.Key_S:
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        elif key == Qt.Key_Space:
            self.animate ^= True
        elif key == Qt.Key_R:
            self.rotate ^= True

        self.update()

        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def timerEvent(self, event) -> None:
        """
        Handles timer events.

        Args:
            event: The QTimerEvent object containing information about the timer event.
        """
        if self.rotate:
            self.rotation += 0.5
        # Now map the OpenGL buffer to the CPU memory
        if self.animate:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
            ptr = gl.glMapBuffer(gl.GL_ARRAY_BUFFER, gl.GL_WRITE_ONLY)
            if ptr:
                # We need to know the number of vertices to create the array view
                # Create a ctypes pointer to the structured data
                # c_array = (ctypes.c_byte * self.buffer_size).from_address(ptr)
                c_array = (ctypes.c_byte * int(self.buffer_size)).from_address(ptr)

                # Create a numpy array that shares the memory with the ctypes array
                buffer = np.frombuffer(c_array, dtype=self.vert_dtype)

                # Now you can access the buffer data as a structured numpy array
                # in this case we will update the y value as .y=sin(( ptr[i].x + offset)) + cos( ptr[i].x - offset);
                offset = self.offset
                # update the y values using numpy vectorisation
                buffer["y"] = np.sin(buffer["x"] + offset) + np.cos(buffer["z"] - offset)

                # Unmap the buffer after you're done
                gl.glUnmapBuffer(gl.GL_ARRAY_BUFFER)
                self.offset += 0.05
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
    parser = argparse.ArgumentParser(description="A WebGPU points demo")
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
