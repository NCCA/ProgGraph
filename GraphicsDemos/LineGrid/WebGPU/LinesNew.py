#!/usr/bin/env -S uv run --active --script
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from NumpyBufferWidget import NumpyBufferWidget
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from wgpu.utils import get_default_device

logging.basicConfig(level=logging.INFO)


class WebGPUScene(NumpyBufferWidget):
    """
    A concrete implementation of NumpyBufferWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    MSAA_SAMPLE_COUNT = 4
    SHADER_FILENAME = "LineShader.wgsl"
    VERTEX_STRIDE = 12
    BUFFER_ALIGNMENT = 256

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebGPU Lines")
        self.device: Optional[wgpu.GPUDevice] = None
        self.pipeline: Optional[wgpu.GPURenderPipeline] = None
        self.vertex_buffer: Optional[wgpu.GPUBuffer] = None

        ratio = self.devicePixelRatio()
        self.texture_size = (int(self.width() * ratio), int(self.height() * ratio))

        self.msaa_sample_count = self.MSAA_SAMPLE_COUNT
        self.rotation = 0.0
        self.view = look_at(Vec3(0, 6, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.animate = True

        aspect = self.texture_size[0] / self.texture_size[1] if self.texture_size[1] > 0 else 1
        self.project = perspective(45.0, aspect, 0.1, 100.0)

        self._initialize_web_gpu()
        self.update()

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        logging.info("Initializing WebGPU")
        try:
            self.device = get_default_device()
            self._init_buffers()
            self._create_render_buffer()
            self._create_render_pipeline()
            self.startTimer(16)
        except Exception as e:
            logging.error(f"Failed to initialize WebGPU: {e}", exc_info=True)

    def _create_lines(self, width: float, depth: float, rows: int, cols: int) -> np.ndarray:
        """Creates vertex data for a grid of lines using numpy."""
        half_w, half_d = width / 2, depth / 2

        # Horizontal lines
        z = np.linspace(-half_d, half_d, rows + 1)
        h_lines = np.zeros((len(z), 2, 3), dtype=np.float32)
        h_lines[:, 0, 0] = -half_w  # start x
        h_lines[:, 1, 0] = half_w  # end x
        h_lines[:, :, 2] = z[:, np.newaxis]  # z for start and end

        # Vertical lines
        x = np.linspace(-half_w, half_w, cols + 1)
        v_lines = np.zeros((len(x), 2, 3), dtype=np.float32)
        v_lines[:, :, 0] = x[:, np.newaxis]  # x for start and end
        v_lines[:, 0, 2] = -half_d  # start z
        v_lines[:, 1, 2] = half_d  # end z

        # Combine and flatten
        points = np.concatenate([h_lines, v_lines]).reshape(-1, 3)
        self.line_vertex_count = len(points)
        return points

    def _create_render_buffer(self) -> None:
        """Create the render buffers for the scene."""
        # This is the texture that the multisampled texture will be resolved to
        colour_buffer_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=1,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        self.colour_buffer_texture = colour_buffer_texture
        self.colour_buffer_texture_view = self.colour_buffer_texture.create_view()

        # This is the multisampled texture that will be rendered to
        self.multisample_texture = self.device.create_texture(
            size=self.texture_size,
            sample_count=self.msaa_sample_count,
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.multisample_texture_view = self.multisample_texture.create_view()

        # Now create a depth buffer
        depth_texture = self.device.create_texture(
            size=self.texture_size,
            format=wgpu.TextureFormat.depth24plus,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            sample_count=self.msaa_sample_count,
        )
        self.depth_buffer_view = depth_texture.create_view()

        # Calculate aligned buffer size for texture copy
        buffer_size = self._calculate_aligned_buffer_size()
        self.readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

    def _init_buffers(self) -> None:
        """Initialize the vertex buffers for the scene."""
        vertex_data = self._create_lines(10, 10, 30, 30)
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        shader_code = Path(self.SHADER_FILENAME).read_text()
        shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="line_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": self.VERTEX_STRIDE,  # 3 floats x 4 bytes per float
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.line_list},
            depth_stencil={
                "format": wgpu.TextureFormat.depth24plus,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": self.msaa_sample_count,
            },
        )

        # Create a uniform buffer
        self.uniform_data = np.zeros((), dtype=[("MVP", "float32", (16))])

        self.uniform_buffer = self.device.create_buffer_with_data(
            data=self.uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_uniform_buffer",
        )

        bind_group_layout = self.pipeline.get_bind_group_layout(0)
        # Create the bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": self.uniform_buffer},
                }
            ],
        )

    def paint(self) -> None:
        """
        Paint the WebGPU content.
        """
        try:
            command_encoder = self.device.create_command_encoder()
            render_pass = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": self.multisample_texture_view,
                        "resolve_target": self.colour_buffer_texture_view,
                        "load_op": wgpu.LoadOp.clear,
                        "store_op": wgpu.StoreOp.store,
                        "clear_value": (0.4, 0.4, 0.4, 1.0),
                    }
                ],
                depth_stencil_attachment={
                    "view": self.depth_buffer_view,
                    "depth_load_op": wgpu.LoadOp.clear,
                    "depth_store_op": wgpu.StoreOp.store,
                    "depth_clear_value": 1.0,
                },
            )
            self.update_uniform_buffers()
            render_pass.set_viewport(0, 0, self.texture_size[0], self.texture_size[1], 0, 1)
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_bind_group(0, self.bind_group, [], 0, 999999)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(self.line_vertex_count)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer(self.colour_buffer_texture)
        except Exception as e:
            logging.error(f"Failed to paint WebGPU content: {e}", exc_info=True)

    def resizeEvent(self, event) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.
        """
        ratio = self.devicePixelRatio()
        size = event.size()
        width = int(size.width() * ratio)
        height = int(size.height() * ratio)

        self.texture_size = (width, height)
        self.project = perspective(45.0, width / height if height > 0 else 1, 0.1, 100.0)

        self._create_render_buffer()

        if self.buffer is not None:
            self.buffer = np.zeros([height, width, 4], dtype=np.uint8)

        self.update()

    def update_uniform_buffers(self) -> None:
        """
        update the uniform buffers for the line pipeline.
        """
        rotation = Mat4.rotate_y(self.rotation)
        mvp_matrix = (self.project @ self.view @ rotation).to_numpy().astype(np.float32)
        self.uniform_data["MVP"] = mvp_matrix.flatten()
        self.device.queue.write_buffer(
            buffer=self.uniform_buffer,
            buffer_offset=0,
            data=self.uniform_data.tobytes(),
        )

    def _calculate_aligned_row_size(self) -> int:
        """
        Calculate the aligned row size for texture copy operations.
        Many GPUs require row alignment to 256 bytes.
        """
        bytes_per_pixel = 4  # RGBA8 = 4 bytes per pixel
        raw_row_size = self.texture_size[0] * bytes_per_pixel
        alignment = 256
        return (raw_row_size + alignment - 1) & -alignment

    def _calculate_aligned_buffer_size(self) -> int:
        """
        Calculate the aligned buffer size needed for texture copy operations.
        """
        aligned_row_size = self._calculate_aligned_row_size()
        return aligned_row_size * self.texture_size[1]

    def _update_colour_buffer(self, texture) -> None:
        """
        Update the colour buffer with the rendered texture data.
        """
        bytes_per_row = self._calculate_aligned_row_size()

        try:
            command_encoder = self.device.create_command_encoder()
            command_encoder.copy_texture_to_buffer(
                {"texture": texture},
                {
                    "buffer": self.readback_buffer,
                    "bytes_per_row": bytes_per_row,
                    "rows_per_image": self.texture_size[1],
                },
                self.texture_size + (1,),
            )
            self.device.queue.submit([command_encoder.finish()])

            self.readback_buffer.map_sync(mode=wgpu.MapMode.READ)
            raw_data = self.readback_buffer.read_mapped()

            height, width = self.texture_size

            strided_view = np.lib.stride_tricks.as_strided(
                np.frombuffer(raw_data, dtype=np.uint8),
                shape=(height, width, 4),
                strides=(bytes_per_row, 4, 1),
            )
            self.buffer = np.copy(strided_view)

            self.readback_buffer.unmap()
        except Exception as e:
            logging.error(f"Failed to update colour buffer: {e}", exc_info=True)
            if self.buffer is not None:
                self.buffer.fill(128)

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering.
        """
        logging.info("Initializing numpy buffer")
        ratio = self.devicePixelRatio()
        width = int(self.width() * ratio)
        height = int(self.height() * ratio)
        self.buffer = np.zeros([height, width, 4], dtype=np.uint8)

    def _toggle_animation(self) -> None:
        """Toggle the animation state."""
        self.animate = not self.animate

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.
        """
        key_handlers = {
            Qt.Key_Escape: self.close,
            Qt.Key_Space: self._toggle_animation,
        }
        handler = key_handlers.get(event.key())
        if handler:
            handler()
        self.update()
        super().keyPressEvent(event)

    def timerEvent(self, event) -> None:
        """
        Handle timer events to update the scene.
        """
        if self.animate:
            self.rotation += 0.5
        self.update()


def main():
    """
    Main function to run the application.
    Initializes and runs the WebGPUScene.
    """
    app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
