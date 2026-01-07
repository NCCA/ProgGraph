#!/usr/bin/env -S uv run --script
"""
This script demonstrates the Simple Factory design pattern.

The Simple Factory pattern provides a way to create objects without exposing the
instantiation logic to the client. A factory function or class is responsible
for creating and returning an instance of a specific class based on some input,
usually a string or an enum.

This pattern promotes loose coupling, as the client code only needs to know about
the abstract interface (the `Renderer` ABC) and the factory function, not the
concrete implementation classes (`OpenGL`, `DirectX`).
"""

from abc import ABC, abstractmethod
from typing import Type

# --- The Abstract Product ---


class Renderer(ABC):
    """
    An abstract base class (ABC) that defines the common interface for all
    renderer products. The factory will produce objects that conform to this
    interface.
    """

    @abstractmethod
    def load_scene(self, filename: str) -> bool:
        """Loads a 3D scene from a file."""
        pass

    @abstractmethod
    def set_viewport_size(self, width: int, height: int) -> None:
        """Sets the size of the rendering viewport."""
        pass

    @abstractmethod
    def set_camera_pos(self, x: float, y: float, z: float) -> None:
        """Sets the position of the camera in the 3D world."""
        pass

    @abstractmethod
    def set_look_at(self, x: float, y: float, z: float) -> None:
        """Sets the point the camera is looking at."""
        pass

    @abstractmethod
    def render(self) -> None:
        """Renders the scene to the screen."""
        pass


# --- Concrete Products ---


class OpenGLRenderer(Renderer):
    """A concrete renderer implementation using an OpenGL-like API."""

    def load_scene(self, filename: str) -> bool:
        print(f"OpenGL: Loading scene from {filename}")
        return True

    def set_viewport_size(self, width: int, height: int) -> None:
        print(f"OpenGL: Setting viewport to {width}x{height}")

    def set_camera_pos(self, x: float, y: float, z: float) -> None:
        print(f"OpenGL: Setting camera position to ({x}, {y}, {z})")

    def set_look_at(self, x: float, y: float, z: float) -> None:
        print(f"OpenGL: Setting camera look-at to ({x}, {y}, {z})")

    def render(self) -> None:
        print("OpenGL: Rendering scene... Done.")


class DirectXRenderer(Renderer):
    """A concrete renderer implementation using a DirectX-like API."""

    def load_scene(self, filename: str) -> bool:
        print(f"DirectX: Loading scene from {filename}")
        return True

    def set_viewport_size(self, width: int, height: int) -> None:
        print(f"DirectX: Setting viewport to {width}x{height}")

    def set_camera_pos(self, x: float, y: float, z: float) -> None:
        print(f"DirectX: Setting camera position to ({x}, {y}, {z})")

    def set_look_at(self, x: float, y: float, z: float) -> None:
        print(f"DirectX: Setting camera look-at to ({x}, {y}, {z})")

    def render(self) -> None:
        print("DirectX: Rendering scene... Done.")


# --- The Simple Factory ---

RENDERER_CATALOG: dict[str, Type[Renderer]] = {
    "OpenGL": OpenGLRenderer,
    "DirectX": DirectXRenderer,
}


def render_factory(kind: str) -> Renderer:
    """
    The factory function that creates and returns a renderer instance.

    Args:
        kind: The type of renderer to create (e.g., "OpenGL", "DirectX").

    Returns:
        An instance of a class that conforms to the Renderer interface.

    Raises:
        ValueError: If the requested renderer `kind` is not supported.
    """
    if kind in RENDERER_CATALOG:
        return RENDERER_CATALOG[kind]()
    else:
        raise ValueError(f"Unknown renderer kind: {kind}")


# --- Client Code Example ---

print("--- Using the factory to get an OpenGL renderer ---")
# The client asks the factory for an OpenGL renderer.
# Note: The client does not need to know about the `OpenGLRenderer` class itself.
renderer = render_factory("OpenGL")
renderer.load_scene("my_scene.obj")
renderer.set_camera_pos(0, 0, 5)
renderer.render()

print("\n--- Using the factory to get a DirectX renderer ---")
# Now, the client asks for a DirectX renderer.
renderer = render_factory("DirectX")
renderer.load_scene("another_scene.fbx")
renderer.set_camera_pos(10, 5, 3)
renderer.render()

print("\n--- Trying to get an unsupported renderer ---")
# The factory handles cases where an unknown renderer is requested.
try:
    renderer = render_factory("WebGPU")
except ValueError as e:
    print(f"Caught expected error: {e}")
