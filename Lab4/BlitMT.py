#!/usr/bin/env -S uv run

import time

import numpy as np
import OpenGL.GL as gl
from PySide6.QtCore import QRunnable, Qt, QThreadPool
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import QOpenGLWindow
from PySide6.QtWidgets import QApplication

from DLA import DLA


class WalkTask(QRunnable):
    def __init__(self, sim):
        super().__init__()
        self.sim = sim

    def run(self):
        while True:
            self.sim.walk()
            time.sleep(0.01)


class Blitter(QOpenGLWindow):
    def __init__(self, simulation, n_cores=1, parent=None):
        super().__init__()
        self.window_width = 1024
        self.window_height = 720
        self.setTitle("NumPy Array Blit Example")
        self.rgba_data = simulation.image
        self.sim = simulation
        self.n_cores = n_cores
        self.thread_pool = QThreadPool.globalInstance()
        self.completed_tasks = 0

    def initializeGL(self):
        gl.glClearColor(0.4, 0.4, 0.4, 1.0)
        self.create_texture_from_numpy()
        self.create_framebuffer()
        self.startTimer(20)
        for _ in range(self.n_cores):
            task = WalkTask(self.sim)
            self.thread_pool.start(task)

    def create_texture_from_numpy(self):
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        height, width, channels = self.rgba_data.shape
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            self.rgba_data.pixels,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def create_framebuffer(self):
        self.fbo_id = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo_id)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            self.texture_id,
            0,
        )
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            print("Error: Framebuffer not complete!")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def paintGL(self):
        gl.glViewport(0, 0, self.window_width, self.window_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.blit_numpy_data_to_screen()

    def blit_numpy_data_to_screen(self):
        height, width, _ = self.rgba_data.shape
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, self.fbo_id)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, self.defaultFramebufferObject())
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBlitFramebuffer(
            0,
            0,
            width,
            height,
            0,
            0,
            self.window_width,
            self.window_height,
            gl.GL_COLOR_BUFFER_BIT,
            gl.GL_NEAREST,
        )
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, 0)

    def timerEvent(self, event):
        self.rgba_data.pixels = np.random.randint(
            0, 255, (self.window_height, self.window_width, 4)
        )
        self.update_numpy_data()
        self.update()

    def update_numpy_data(self):
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        height, width, channels = self.rgba_data.shape
        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            self.rgba_data.pixels,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self.update()

    def resizeGL(self, w, h):
        self.window_width = int(w * self.devicePixelRatio())
        self.window_height = int(h * self.devicePixelRatio())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        super().keyPressEvent(event)


if __name__ == "__main__":
    format = QSurfaceFormat()
    format.setMajorVersion(4)
    format.setMinorVersion(1)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(format)

    app = QApplication([])
    sim = DLA(400, 400)
    for _ in range(20):
        sim.random_seed()
    window = Blitter(sim, n_cores=4)  # Change n_cores as needed
    window.resize(1024, 720)
    window.show()
    app.exec()

print("hello")
