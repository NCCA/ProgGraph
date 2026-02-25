#!/usr/bin/env -S uv run --active --script
import argparse
import sys

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from WebGPUWidget import WebGPUWidget
from wgpu.utils import get_default_device


class WebGPUScene(WebGPUWidget):
    """
    A concrete implementation of NumpyBufferWidget for a WebGPU scene.

    This class implements the abstract methods to provide functionality for initializing,
    painting, and resizing the WebGPU context.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("WebGPU Lines")
        self.device = None
        self.pipeline = None
        self.vertex_buffer = None
        self.msaa_sample_count = 4
        self.rotation = 0.0
        self.view = look_at(Vec3(0, 6, 15), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.animate = True

        self._initialize_web_gpu()
        self.update()

    def _initialize_web_gpu(self) -> None:
        """
        Initialize the WebGPU context.

        This method sets up the WebGPU context for the scene.
        """
        print("initializeWebGPU")
        try:
            self.device = get_default_device()
            self._init_buffers()
            self._create_render_buffer()
            self._create_render_pipeline()
            self.startTimer(16)
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")

    def _create_triangle_plane_vectorized(
        self, width: float, depth: float, width_step: int, depth_step: int, normal: Vec3
    ):
        """
        Generates a flat list of vertices for a plane using vectorized numpy operations.
        """
        # A more convenient dtype for vertex data
        vert_dtype = np.dtype([("position", np.float32, 3), ("normal", np.float32, 3), ("uv", np.float32, 2)])
        self.tri_vertex_count = width_step * depth_step * 6
        # Create grid of quad corner coordinates
        x = np.linspace(-width / 2, width / 2, width_step + 1, dtype=np.float32)
        z = np.linspace(-depth / 2, depth / 2, depth_step + 1, dtype=np.float32)
        u = np.linspace(0, 1, width_step + 1, dtype=np.float32)
        v = np.linspace(1, 0, depth_step + 1, dtype=np.float32)  # Flipped

        # Get coordinates for the 4 corners of each quad
        pos_tl = np.array(np.meshgrid(x[:-1], z[:-1])).T.reshape(-1, 2)
        pos_tr = np.array(np.meshgrid(x[1:], z[:-1])).T.reshape(-1, 2)
        pos_bl = np.array(np.meshgrid(x[:-1], z[1:])).T.reshape(-1, 2)
        pos_br = np.array(np.meshgrid(x[1:], z[1:])).T.reshape(-1, 2)

        uv_tl = np.array(np.meshgrid(u[:-1], v[:-1])).T.reshape(-1, 2)
        uv_tr = np.array(np.meshgrid(u[1:], v[:-1])).T.reshape(-1, 2)
        uv_bl = np.array(np.meshgrid(u[:-1], v[1:])).T.reshape(-1, 2)
        uv_br = np.array(np.meshgrid(u[1:], v[1:])).T.reshape(-1, 2)

        num_quads = width_step * depth_step
        vertex_data = np.empty(num_quads * 6, dtype=vert_dtype)

        # Interleave the corner data to form two triangles per quad
        # Triangle 1: top-left, bottom-left, top-right
        vertex_data[0::6]["position"] = np.insert(pos_tl, 1, 0, axis=1)
        vertex_data[1::6]["position"] = np.insert(pos_bl, 1, 0, axis=1)
        vertex_data[2::6]["position"] = np.insert(pos_tr, 1, 0, axis=1)
        vertex_data[0::6]["uv"] = uv_tl
        vertex_data[1::6]["uv"] = uv_bl
        vertex_data[2::6]["uv"] = uv_tr

        # Triangle 2: top-right, bottom-left, bottom-right
        vertex_data[3::6]["position"] = np.insert(pos_tr, 1, 0, axis=1)
        vertex_data[4::6]["position"] = np.insert(pos_bl, 1, 0, axis=1)
        vertex_data[5::6]["position"] = np.insert(pos_br, 1, 0, axis=1)
        vertex_data[3::6]["uv"] = uv_tr
        vertex_data[4::6]["uv"] = uv_bl
        vertex_data[5::6]["uv"] = uv_br

        # Assign normals to all vertices
        vertex_data["normal"] = np.tile(normal.to_list(), (num_quads * 6, 1))

        return vertex_data

    def _init_buffers(self):
        vertex_data = self._create_triangle_plane_vectorized(10, 10, 30, 30, Vec3(0, 1, 0))
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=vertex_data.tobytes(), usage=wgpu.BufferUsage.VERTEX
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("TriShader.wgsl", "r") as f:
            shader_code = f.read()
            shader_module = self.device.create_shader_module(code=shader_code)

        self.pipeline = self.device.create_render_pipeline(
            label="particle_pipeline",
            layout="auto",
            vertex={
                "module": shader_module,
                "entry_point": "vertex_main",
                "buffers": [
                    {
                        "array_stride": 8 * 4,  # 8 floats x 4 bytes per float
                        "step_mode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                            {"format": "float32x2", "offset": 24, "shader_location": 2},
                        ],
                    }
                ],
            },
            fragment={
                "module": shader_module,
                "entry_point": "fragment_main",
                "targets": [{"format": wgpu.TextureFormat.rgba8unorm}],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
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
        self.uniform_data = np.zeros(
            (),
            dtype=[
                ("MVP", "float32", (16)),
            ],
        )

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

    def paintWebGPU(self) -> None:
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
            render_pass.draw(self.tri_vertex_count)
            render_pass.end()
            self.device.queue.submit([command_encoder.finish()])
            self._update_colour_buffer()
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def resizeWebGPU(self, width, height) -> None:
        """
        Called whenever the window is resized.
        It's crucial to update the viewport and projection matrix here.

        Args:
        """

        # Update projection matrix
        self.project = perspective(45.0, width / height if height > 0 else 1, 0.1, 100.0)

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

    def initialize_buffer(self) -> None:
        """
        Initialize the numpy buffer for rendering .

        """
        print("initialize numpy buffer")
        width = int(self.width() * self.ratio)
        height = int(self.height() * self.ratio)
        self.frame_buffer = np.zeros([height, width, 4], dtype=np.uint8)

    def keyPressEvent(self, event) -> None:
        """
        Handles keyboard press events.

        Args:
            event: The QKeyEvent object containing information about the key press.
        """
        key = event.key()
        if key == Qt.Key_Escape:
            self.close()  # Exit the application
        elif key == Qt.Key_Space:
            self.animate = not self.animate

        self.update()

        # Call the base class implementation for any unhandled events
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
    Parses command line arguments and initializes the WebGPUScene.
    """
    parser = argparse.ArgumentParser(description="A WebGPU Line demo")
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Run in debug mode",
    )

    args = parser.parse_args()
    app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
