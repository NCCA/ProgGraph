#!/usr/bin/env -S uv run --active --script
import sys
from typing import Optional

import numpy as np
import wgpu
import wgpu.utils
from NumpyBufferWidget import NumpyBufferWidget
from PySide6.QtCore import Qt, QTimerEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication


class WebGPUScene(NumpyBufferWidget):
    """
    A concrete implementation of NumpyBufferWidget for rendering a simple WebGPU scene.

    This class sets up a WebGPU device, creates a render pipeline for a coloured
    triangle, and handles painting and animation updates.
    """

    def __init__(self) -> None:
        """Initializes the WebGPU scene widget."""
        super().__init__()
        self.setWindowTitle("WebGPU Triangle")

        # WebGPU-specific attributes
        self.device: Optional[wgpu.GPUDevice] = None
        self.pipeline: Optional[wgpu.GPURenderPipeline] = None
        self.vertex_buffer: Optional[wgpu.GPUBuffer] = None

        # Scene attributes
        self.angle: float = 0.0
        # Vertex data: 3 vertices, each with a 3D position (x, y, z) and a 3D colour (r, g, b)
        # fmt: off
        size= 0.5
        self.vertices: np.ndarray = np.array([
            -size, -size, 0.0, 1.0, 0.0, 0.0,  # Bottom-left vertex (red)
            0.0,   size, 0.0, 0.0, 1.0, 0.0,  # Top vertex (green)
            size, -size, 0.0, 0.0, 0.0, 1.0,  # Bottom-right vertex (blue)
        ], dtype=np.float32)
        # fmt: on

        # Buffer dimensions
        self.buffer_width: int = 1024
        self.buffer_height: int = 1024

        # Trigger the first paint event, which will initialize the WebGPU context.
        self.update()

    def initialize_buffer(self) -> None:
        """
        Initialize the WebGPU context, buffers, and render pipeline.

        This method is called once by the base class before the first paint event.
        """
        print("Initializing WebGPU context...")
        try:
            # Get the default WebGPU device
            self.device = wgpu.utils.get_default_device()
            if self.device is None:
                raise RuntimeError("Could not get a WebGPU device.")

            # Initialize GPU buffers and the render pipeline
            self._init_buffers()
            self._create_render_pipeline()

            # Start a timer to trigger animation updates
            self.startTimer(16)  # ~60 FPS
        except Exception as e:
            print(f"Failed to initialize WebGPU: {e}")
            # Create a dummy buffer to avoid errors in paintEvent
            self.buffer = np.zeros([self.buffer_height, self.buffer_width, 4], dtype=np.uint8)

    def _init_buffers(self) -> None:
        """Initializes the GPU buffers required for rendering."""
        if self.device is None:
            return
        # Create the main vertex buffer on the GPU.
        # This buffer will hold the vertex data for the triangle.
        self.vertex_buffer = self.device.create_buffer(
            size=self.vertices.nbytes,
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )

        # Create a temporary staging buffer for updating the vertex buffer from the CPU.
        # This buffer is mappable, allowing the CPU to write data to it.
        self.vertex_buffer.copy_buffer = self.device.create_buffer(
            size=self.vertices.nbytes,
            usage=wgpu.BufferUsage.MAP_WRITE | wgpu.BufferUsage.COPY_SRC,
        )

    def _create_render_pipeline(self) -> None:
        """Creates the WebGPU render pipeline, including shaders."""
        if self.device is None:
            return
        # WGSL shader code for the vertex shader.
        # It takes vertex position and colour, and outputs them to the fragment shader.
        vertex_shader_code = """
        struct VertexIn {
            @location(0) position: vec3<f32>,
            @location(1) colour: vec3<f32>,
        };

        struct VertexOut {
            @builtin(position) position: vec4<f32>,
            @location(0) fragColour: vec3<f32>,
        };

        @vertex
        fn main(input: VertexIn) -> VertexOut {
            var output: VertexOut;
            output.position = vec4<f32>(input.position, 1.0);
            output.fragColour = input.colour;
            return output;
        }
        """

        # WGSL shader code for the fragment shader.
        # It receives the interpolated colour from the vertex shader and outputs it.
        fragment_shader_code = """
        @fragment
        fn main(@location(0) fragColour: vec3<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(fragColour, 1.0); // Output the colour with full alpha
        }
        """

        # Create a pipeline layout (no bind groups needed for this simple example)
        pipeline_layout = self.device.create_pipeline_layout(bind_group_layouts=[])

        # Create the render pipeline
        self.pipeline = self.device.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self.device.create_shader_module(code=vertex_shader_code),
                "entry_point": "main",
                "buffers": [
                    {
                        # Define the structure of our vertex buffer
                        "array_stride": 6 * 4,  # 6 floats (pos+colour) * 4 bytes/float
                        "step_mode": "vertex",
                        "attributes": [
                            # Attribute 0: Position (vec3<f32>)
                            {"format": "float32x3", "offset": 0, "shader_location": 0},
                            # Attribute 1: Colour (vec3<f32>)
                            {"format": "float32x3", "offset": 12, "shader_location": 1},
                        ],
                    }
                ],
            },
            fragment={
                "module": self.device.create_shader_module(code=fragment_shader_code),
                "entry_point": "main",
                "targets": [{"format": "rgba8unorm"}],
            },
            primitive={"topology": "triangle-list"},
        )

    def paint(self) -> None:
        """
        Renders a single frame of the WebGPU scene.

        This method is called by the paintEvent of the widget.
        """
        if self.device is None or self.pipeline is None:
            self.render_text(10, 20, "WebGPU not initialized.", size=20, colour=Qt.GlobalColor.red)
            return

        self.render_text(10, 20, "First Triangle WebGPU", size=20, colour=Qt.GlobalColor.black)
        try:
            # Create a texture to render to
            texture: wgpu.GPUTexture = self.device.create_texture(
                size=(self.buffer_width, self.buffer_height, 1),
                format="rgba8unorm",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
            )
            texture_view: wgpu.GPUTextureView = texture.create_view()

            # Create a command encoder to record rendering commands
            command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder()

            # Begin a render pass
            render_pass: wgpu.GPURenderPassEncoder = command_encoder.begin_render_pass(
                color_attachments=[
                    {
                        "view": texture_view,
                        "resolve_target": None,
                        "load_op": "clear",
                        "store_op": "store",
                        "clear_value": (0.4, 0.4, 0.4, 1.0),  # Background color
                    }
                ]
            )

            # Set the pipeline and vertex buffer, then draw the triangle
            render_pass.set_pipeline(self.pipeline)
            render_pass.set_vertex_buffer(0, self.vertex_buffer)
            render_pass.draw(3)  # Draw 3 vertices
            render_pass.end()

            # Submit the commands to the GPU queue
            self.device.queue.submit([command_encoder.finish()])

            # Copy the rendered texture to the numpy buffer for display
            self._update_colour_buffer(texture)
        except Exception as e:
            print(f"Failed to paint WebGPU content: {e}")

    def _update_colour_buffer(self, texture: wgpu.GPUTexture) -> None:
        """
        Copies the rendered texture data from the GPU to a numpy array.

        Args:
            texture (wgpu.GPUTexture): The source texture to copy from.
        """
        if self.device is None:
            return

        buffer_size = self.buffer_width * self.buffer_height * 4  # Width * Height * 4 bytes/pixel (RGBA8)
        try:
            # Create a readback buffer on the GPU that the CPU can read from
            readback_buffer: wgpu.GPUBuffer = self.device.create_buffer(
                size=buffer_size,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
            )

            # Encode a command to copy the texture to the readback buffer
            command_encoder: wgpu.GPUCommandEncoder = self.device.create_command_encoder()
            command_encoder.copy_texture_to_buffer(
                source={
                    "texture": texture,
                    "mip_level": 0,
                    "origin": (0, 0, 0),
                },
                destination={
                    "buffer": readback_buffer,
                    "offset": 0,
                    "bytes_per_row": self.buffer_width * 4,
                    "rows_per_image": self.buffer_height,
                },
                copy_size=(self.buffer_width, self.buffer_height, 1),
            )
            self.device.queue.submit([command_encoder.finish()])

            # Map the buffer to CPU-accessible memory
            readback_buffer.map_sync(mode=wgpu.MapMode.READ)
            # Get the mapped data
            raw_data = readback_buffer.read_mapped()

            # Create a numpy array view of the raw data and assign it to self.buffer
            self.buffer = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.buffer_height, self.buffer_width, 4))

            # Unmap the buffer
            readback_buffer.unmap()
        except Exception as e:
            print(f"Failed to update color buffer: {e}")

    def timerEvent(self, event: QTimerEvent) -> None:
        """
        Handles timer events to update and animate the scene.

        Args:
            event (QTimerEvent): The timer event object.
        """
        if self.device is None or self.vertex_buffer is None:
            return

        # Simple animation: rotate the triangle
        self.angle += 0.01
        rotation_matrix = np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ],
            dtype=np.float32,
        )

        # Create a copy of original vertices to apply rotation
        rotated_vertices = self.vertices.copy()
        # Apply rotation to each vertex position
        for i in range(3):
            xy = self.vertices[i * 6 : i * 6 + 2]
            rotated_xy = xy @ rotation_matrix
            rotated_vertices[i * 6 : i * 6 + 2] = rotated_xy

        # Update the vertex buffer with the new rotated vertex data
        try:
            tmp_buffer = self.vertex_buffer.copy_buffer
            # Map the staging buffer to write the new vertex data
            tmp_buffer.map_sync(wgpu.MapMode.WRITE)
            tmp_buffer.write_mapped(rotated_vertices.tobytes())
            tmp_buffer.unmap()

            # Encode and submit a command to copy from the staging buffer to the main vertex buffer
            command_encoder = self.device.create_command_encoder()
            command_encoder.copy_buffer_to_buffer(tmp_buffer, 0, self.vertex_buffer, 0, self.vertices.nbytes)
            self.device.queue.submit([command_encoder.finish()])
        except Exception as e:
            print(f"Error updating vertex buffer: {e}")

        # Trigger a repaint of the widget
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Handles key press events.

        Args:
            event (QKeyEvent): The key event object.
        """
        if event.key() == Qt.Key.Key_Escape:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = WebGPUScene()
    win.resize(800, 800)
    win.show()
    sys.exit(app.exec())
