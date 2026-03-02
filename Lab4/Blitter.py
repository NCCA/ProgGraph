#!/usr/bin/env -S uv run



import OpenGL.GL as gl
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

from DLA import DLA


class SimWorker(QObject):
    updated = Signal()  # Signal to notify when walk() is done

    def __init__(self, sim):
        super().__init__()
        self.sim = sim
        self.running = True

    @Slot()
    def run(self):
        while self.running:
            self.sim.walk()
            self.updated.emit()
            QThread.msleep(16)  # ~60 FPS, adjust as needed

    def stop(self):
        self.running = False


class Blitter(QOpenGLWindow):
    def __init__(self, simulation, parent=None):
        super().__init__()
        self.window_width = simulation.width
        self.window_height = simulation.height
        self.setTitle("NumPy Array Blit Example")
        self.rgba_data = simulation.image
        self.sim = simulation

    def initializeGL(self):
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        self.create_texture_from_numpy()
        # Create framebuffer object
        self.create_framebuffer()
        self.worker_thread = QThread()
        self.worker = SimWorker(self.sim)
        self.worker.moveToThread(self.worker_thread)
        self.worker.updated.connect(self.update_numpy_data)
        self.worker.updated.connect(self.update)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
        self.startTimer(20)

    def create_texture_from_numpy(self):
        """Create an OpenGL texture from the numpy RGBA data"""
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        # Upload the numpy array data to the texture
        height, width, channels = self.rgba_data.shape
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,  # target
            0,  # level
            gl.GL_RGBA8,  # internal format
            width,  # width
            height,  # height
            0,  # border (must be 0)
            gl.GL_RGBA,  # format
            gl.GL_UNSIGNED_BYTE,  # type
            self.rgba_data.pixels,  # data - numpy array
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def create_framebuffer(self):
        """Create a framebuffer object with our texture as color attachment"""
        self.fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)

        # Attach our texture as the color attachment
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.texture_id,
            0,
        )

        # Check framebuffer completeness
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("Error: Framebuffer not complete!")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def paintGL(self):
        # Set viewport and clear
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Blit from our FBO to the default framebuffer (screen)
        self.blit_numpy_data_to_screen()

    def blit_numpy_data_to_screen(self):
        """Blit the numpy array data from FBO to screen using glBlitFramebuffer"""
        height, width, _ = self.rgba_data.shape

        # Bind our FBO as the read framebuffer
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo_id)

        # Bind the default framebuffer as the draw framebuffer
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())

        # Set which color attachment to read from
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)

        # Perform the blit operation
        # This copies from the source rectangle to destination rectangle
        gl.glBlitFramebuffer(
            0,
            0,
            width,
            height,  # source rectangle (x0, y0, x1, y1)
            0,
            0,
            self.window_width,
            self.window_height,  # dest rectangle
            gl.GL_COLOR_BUFFER_BIT,  # which buffers to copy
            gl.GL_NEAREST,  # interpolation method
        )

        # Unbind framebuffers
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

    def timerEvent(self, event):
        self.sim.walk()
        self.update_numpy_data()
        self.update()

    def update_numpy_data(self):
        """Update the texture with new numpy RGBA data"""
        # Bind the texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        # Update texture data with new numpy array
        height, width, channels = self.rgba_data.shape
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,  # target
            0,  # level
            0,
            0,  # x and y offset
            width,
            height,  # width and height
            gl.GL_RGBA,  # format
            gl.GL_UNSIGNED_BYTE,  # type
            self.rgba_data.pixels,  # new data
        )

        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.update()  # Trigger a repaint

    def resizeGL(self, w, h):
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()
        super().closeEvent(event)


if __name__ == "__main__":
    # Set up OpenGL format
    format = QSurfaceFormat()
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    app = QApplication([])
    sim = DLA(400, 400)
    for _ in range(200):
        sim.random_seed()
    window = Blitter(sim)
    window.resize(400, 400)
    window.show()
    app.exec()

print("hello")
