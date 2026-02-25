#!/usr/bin/env -S uv run --active --script
import argparse
import sys

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat4, PerspMode, Vec3, look_at, perspective
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QApplication, QCheckBox, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget
from wgpu.utils import get_default_device

from WebGPUWidget import WebGPUWidget


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
        self.vertex_data = None
        self.msaa_sample_count = 4
        self.rotation = 0.0
        self.view = look_at(Vec3(0, 10, 20), Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.animate = True
        self.rotate = True
        self.offset = 0.0
        self.line_width = 0.02
        self.is_wireframe = False

        self._initialize_web_gpu()
        self._addUI()
        self.update()

    def _addUI(self):
        # --- Create UI Controls ---
        controls_container = QWidget()
        controls_layout = QHBoxLayout()
        controls_container.setLayout(controls_layout)

        # Checkbox for wireframe
        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setChecked(self.is_wireframe)
        self.wireframe_checkbox.stateChanged.connect(
            lambda state: setattr(self, "is_wireframe", state == Qt.CheckState.Checked.value)
        )
        controls_layout.addWidget(self.wireframe_checkbox)

        self.animate_checkbox = QCheckBox("Animate")
        self.animate_checkbox.setChecked(self.animate)
        self.animate_checkbox.stateChanged.connect(
            lambda state: setattr(self, "animate", state == Qt.CheckState.Checked.value)
        )
        controls_layout.addWidget(self.animate_checkbox)

        self.rotate_checkbox = QCheckBox("Rotate")
        self.rotate_checkbox.setChecked(self.rotate)
        self.rotate_checkbox.stateChanged.connect(
            lambda state: setattr(self, "rotate", state == Qt.CheckState.Checked.value)
        )
        controls_layout.addWidget(self.rotate_checkbox)

        # Slider for line width
        controls_layout.addWidget(QLabel("Line Width"))
        self.line_slider = QSlider(Qt.Orientation.Horizontal)
        self.line_slider.setRange(1, 100)
        self.line_slider.setValue(int(self.line_width * 100))
        self.line_slider.valueChanged.connect(self._set_line_width)
        controls_layout.addWidget(self.line_slider)
        # --- Add controls to the main window layout ---
        # The WebGPUWidget itself is the central widget, we can give it a layout
        main_layout = QVBoxLayout()
        # Add the rendering widget first, making it stretch
        # Note: This part is tricky as self is the widget. We add a container for controls.
        self.setLayout(main_layout)
        # We need a dummy widget to take the place of the renderer in the layout
        dummy_widget = QWidget()
        # this way round controls are at top (if we want at bottom just swap with below)
        main_layout.addWidget(controls_container)
        main_layout.addWidget(dummy_widget, 1)  # This will stretch

    @Slot(bool)
    def _set_wireframe_mode(self, state):
        self.is_wireframe = state == Qt.CheckState.Checked.value
        self.update()

    @Slot(int)
    def _set_line_width(self, value):
        self.line_width = value / 100.0
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
        self.vertex_data = self._create_triangle_plane_vectorized(20, 20, 50, 50, Vec3(0, 1, 0))
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=self.vertex_data.tobytes(), usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        with open("TriShaderWithWireFrame.wgsl", "r") as f:
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
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba8unorm,
                        # Add this "blend" configuration to enable transparency
                        "blend": {
                            "color": {
                                "src_factor": wgpu.BlendFactor.src_alpha,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                            "alpha": {
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one_minus_src_alpha,
                                "operation": wgpu.BlendOperation.add,
                            },
                        },
                    }
                ],
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
                ("line_width", "float32", (1)),
                ("is_wireframe", "float32", (1)),
                ("padding", "float32", (2)),
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

    def _update_vertex_buffer(self):
        """
        Update the vertex buffer per frame based on the animation offset.
        """
        if self.vertex_data is None:
            return

        updated_data = self.vertex_data.copy()
        positions = updated_data["position"]

        x = positions[:, 0]
        z = positions[:, 2]

        # Apply the algorithm to modify the y-coordinate
        positions[:, 1] = np.sin(x + self.offset) + np.cos(z - self.offset)

        # Recalculate normals for correct lighting
        # For a surface y = f(x, z), the normal is (-df/dx, 1, -df/dz)
        # f(x,z) = sin(x + offset) + cos(z - offset)
        # df/dx = cos(x + offset)
        # df/dz = -sin(z - offset)
        # Normal = (-cos(x + offset), 1, sin(z - offset))
        nx = -np.cos(x + self.offset)
        ny = np.ones_like(x)
        nz = np.sin(z - self.offset)

        normals = np.stack((nx, ny, nz), axis=-1)
        # Normalize the normals
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        # prevent division by zero
        norm[norm == 0] = 1.0
        normals = normals / norm

        updated_data["normal"] = normals

        # Write the updated data to the GPU buffer
        self.device.queue.write_buffer(self.vertex_buffer, 0, updated_data.tobytes())

    def paintWebGPU(self) -> None:
        """
        Paint the WebGPU content.
        """
        try:
            if self.animate:
                self._update_vertex_buffer()
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
        self.uniform_data["line_width"] = self.line_width
        self.uniform_data["is_wireframe"] = self.is_wireframe

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
        elif key == Qt.Key_W:
            self.is_wireframe = 1
        elif key == Qt.Key_S:
            self.is_wireframe = 0.1

        self.update()

        # Call the base class implementation for any unhandled events
        super().keyPressEvent(event)

    def timerEvent(self, event) -> None:
        """
        Handle timer events to update the scene.
        """
        if self.animate:
            self.offset += 0.05
        if self.rotate:
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
