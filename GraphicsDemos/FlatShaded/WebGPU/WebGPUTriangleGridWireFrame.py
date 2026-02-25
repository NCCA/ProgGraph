#!/usr/bin/env -S uv run --active --script
import argparse
import os
import sys
import traceback

import numpy as np
import wgpu
import wgpu.utils
from ncca.ngl import Mat3, Mat4, PerspMode, Vec3, look_at, perspective
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
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
        self.vertex_data = None
        self.msaa_sample_count = 4
        self.rotation = 0.0
        self.view_pos = Vec3(0, 10, 20)
        self.view = look_at(self.view_pos, Vec3(0, 0, 0), Vec3(0, 1, 0))
        self.animate = True
        self.rotate = True
        self.offset = 0.0
        # Material properties
        self.line_width = 0.02
        self.is_wireframe = False
        self.objectColour = Vec3(0.8, 0.8, 0.1)
        self.lightColour = Vec3(1.0, 1.0, 1.0)
        self.lightPos = Vec3(5.0, 5.0, 5.0)
        self.specularStrength = 0.5
        self.shininess = 32.0

        self._initialize_web_gpu()
        self._addUI()
        self.update()

    def _addUI(self):
        # Main container for all controls
        self.controls_container = QWidget()
        main_controls_layout = QVBoxLayout(self.controls_container)

        # Grid layout for all controls
        grid_layout = QGridLayout()
        main_controls_layout.addLayout(grid_layout)

        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setChecked(self.is_wireframe)
        self.wireframe_checkbox.stateChanged.connect(self._set_wireframe_mode)
        grid_layout.addWidget(self.wireframe_checkbox, 0, 0)

        self.animate_checkbox = QCheckBox("Animate")
        self.animate_checkbox.setChecked(self.animate)
        self.animate_checkbox.stateChanged.connect(
            lambda state: setattr(self, "animate", state == Qt.CheckState.Checked.value)
        )
        grid_layout.addWidget(self.animate_checkbox, 0, 1)

        self.rotate_checkbox = QCheckBox("Rotate")
        self.rotate_checkbox.setChecked(self.rotate)
        self.rotate_checkbox.stateChanged.connect(
            lambda state: setattr(self, "rotate", state == Qt.CheckState.Checked.value)
        )
        grid_layout.addWidget(self.rotate_checkbox, 0, 2)

        self.object_colour_button = QPushButton("Set Object Colour")
        self.object_colour_button.clicked.connect(self._set_object_colour)
        grid_layout.addWidget(self.object_colour_button, 1, 0, 1, 2)

        self.light_colour_button = QPushButton("Set Light Colour")
        self.light_colour_button.clicked.connect(self._set_light_colour)
        grid_layout.addWidget(self.light_colour_button, 1, 2, 1, 2)

        def add_slider(row, col, name, min_val, max_val, initial_val, callback):
            label = QLabel(name)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(initial_val)
            slider.valueChanged.connect(callback)
            grid_layout.addWidget(label, row, col)
            grid_layout.addWidget(slider, row, col + 1)
            return slider

        # Add sliders
        add_slider(2, 0, "Line Width", 1, 100, int(self.line_width * 100), self._set_line_width)
        add_slider(2, 2, "Shininess", 1, 256, int(self.shininess), self._set_shininess)
        add_slider(3, 0, "Specular", 0, 100, int(self.specularStrength * 100), self._set_specular)
        add_slider(3, 2, "Light X", -20, 20, int(self.lightPos.x), lambda v: self._update_light_pos("x", v))
        add_slider(4, 0, "Light Y", 0, 20, int(self.lightPos.y), lambda v: self._update_light_pos("y", v))
        add_slider(4, 2, "Light Z", -20, 20, int(self.lightPos.z), lambda v: self._update_light_pos("z", v))

        # Main window layout
        main_layout = QVBoxLayout(self)
        self.toggle_button = QPushButton("Show/Hide Controls")
        self.toggle_button.clicked.connect(self._toggle_controls)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.controls_container)
        main_layout.addStretch(1)

    @Slot()
    def _toggle_controls(self):
        self.controls_container.setVisible(not self.controls_container.isVisible())

    @Slot()
    def _set_object_colour(self):
        colour = QColorDialog.getColor(
            QColor(
                int(self.objectColour.x * 255),
                int(self.objectColour.y * 255),
                int(self.objectColour.z * 255),
            )
        )
        if colour.isValid():
            self.objectColour.x = colour.redF()
            self.objectColour.y = colour.greenF()
            self.objectColour.z = colour.blueF()
            self.update()

    @Slot()
    def _set_light_colour(self):
        colour = QColorDialog.getColor(
            QColor(
                int(self.lightColour.x * 255),
                int(self.lightColour.y * 255),
                int(self.lightColour.z * 255),
            )
        )
        if colour.isValid():
            self.lightColour.x = colour.redF()
            self.lightColour.y = colour.greenF()
            self.lightColour.z = colour.blueF()
            self.update()

    @Slot(int)
    def _set_wireframe_mode(self, state):
        self.is_wireframe = state == Qt.CheckState.Checked.value
        self.update()

    @Slot(int)
    def _set_line_width(self, value):
        self.line_width = value / 100.0
        self.update()

    @Slot(int)
    def _set_shininess(self, value):
        self.shininess = float(value)
        self.update()

    @Slot(int)
    def _set_specular(self, value):
        self.specularStrength = value / 100.0
        self.update()

    @Slot(str, int)
    def _update_light_pos(self, component, value):
        setattr(self.lightPos, component, float(value))
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
        self.vertex_data = self._create_triangle_plane_vectorized(50, 50, 250, 250, Vec3(0, 1, 0))
        self.vertex_buffer = self.device.create_buffer_with_data(
            data=self.vertex_data.tobytes(), usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST
        )

    def _create_render_pipeline(self) -> None:
        """
        Create a render pipeline.
        """
        shader_file = os.path.join(os.path.dirname(__file__), "PhongShaderWithWireFrame.wgsl")
        with open(shader_file, "r") as f:
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
                ("M", "float32", (16)),
                ("normal_matrix", "float32", (16)),
                ("padding", "float32", (16)),
            ],
        )

        material_dtype = np.dtype([
            ("objectColour", ("float32", 4)),
            ("lightColour", ("float32", 4)),
            ("lightPos", ("float32", 4)),
            ("viewPos", ("float32", 4)),
            ("params", ("float32", 4)),
        ])

        self.material_data = np.zeros(
            (),
            dtype=material_dtype,
        )

        self.uniform_buffer = self.device.create_buffer_with_data(
            data=self.uniform_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_uniform_buffer",
        )
        self.material_buffer = self.device.create_buffer_with_data(
            data=self.material_data.tobytes(),
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            label="line_pipeline_material_buffer",
        )

        bind_group_layout = self.pipeline.get_bind_group_layout(0)
        # Create the bind group
        self.bind_group = self.device.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,  # Matches @binding(0) in the shader
                    "resource": {"buffer": self.uniform_buffer},
                },
                {
                    "binding": 1,  # Matches @binding(1) in the shader
                    "resource": {"buffer": self.material_buffer},
                },
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
        # For a surface y = f(x, z), the normal is (-df/dx, 1, -df/dz) .
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
        normal_matrix = rotation.copy()
        normal_matrix.inverse().transpose()
        self.uniform_data["MVP"] = mvp_matrix.flatten()
        self.uniform_data["M"] = rotation.to_numpy().flatten()
        self.uniform_data["normal_matrix"] = normal_matrix.to_numpy().flatten()

        self.device.queue.write_buffer(
            buffer=self.uniform_buffer,
            buffer_offset=0,
            data=self.uniform_data.tobytes(),
        )
        # Material properties
        self.material_data["objectColour"] = self.objectColour.to_list() + [0.0]
        self.material_data["lightColour"] = self.lightColour.to_list() + [0.0]
        self.material_data["lightPos"] = self.lightPos.to_list() + [0.0]
        self.material_data["viewPos"] = self.view_pos.to_list() + [0.0]
        self.material_data["params"] = (
            self.specularStrength,
            self.shininess,
            self.line_width,
            float(self.is_wireframe),
        )

        self.device.queue.write_buffer(
            buffer=self.material_buffer,
            buffer_offset=0,
            data=self.material_data.tobytes(),
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

    # Check for a "--debug" command-line argument to run the DebugApplication
    if args.debug:
        print("Debug mode enabled")
        app = DebugApplication()
    else:
        print("Running in normal mode")
        app = QApplication()

    win = WebGPUScene()
    win.resize(1024, 720)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
