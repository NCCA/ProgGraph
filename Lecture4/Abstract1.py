#!/usr/bin/env -S uv run --script
"""
This script provides an example of Abstract Base Classes (ABCs) in Python.

ABCs are a way to define interfaces in Python. An ABC can define a set of methods
and properties that a concrete class must implement. This helps in designing a
clear and maintainable class hierarchy.
"""

import math
from abc import ABC, abstractmethod


class Shape(ABC):
    """
    An abstract base class for geometric shapes.

    This class defines the common interface that all shapes must implement.
    It includes abstract methods for calculating the area and perimeter.
    Classes that inherit from Shape must provide their own implementation for these
    methods.
    """

    @abstractmethod
    def area(self) -> float:
        """
        Calculates the area of the shape.

        This is an abstract method and must be implemented by any concrete subclass.

        Returns:
            The area of the shape as a float.
        """
        pass

    @abstractmethod
    def perimeter(self) -> float:
        """
        Calculates the perimeter of the shape.

        This is an abstract method and must be implemented by any concrete subclass.

        Returns:
            The perimeter of the shape as a float.
        """
        pass


class Circle(Shape):
    """
    Represents a circle, a concrete implementation of the Shape ABC.
    """

    def __init__(self, radius: float):
        """
        Initializes a Circle with a given radius.

        Args:
            radius: The radius of the circle.
        """
        self.radius = radius

    def area(self) -> float:
        """
        Calculates the area of the circle.

        Returns:
            The area of the circle.
        """
        return math.pi * self.radius**2

    def perimeter(self) -> float:
        """
        Calculates the circumference (perimeter) of the circle.

        Returns:
            The circumference of the circle.
        """
        return 2 * math.pi * self.radius


class Square(Shape):
    """
    Represents a square, a concrete implementation of the Shape ABC.
    """

    def __init__(self, side: float):
        """
        Initializes a Square with a given side length.

        Args:
            side: The length of the side of the square.
        """
        self.side = side

    def area(self) -> float:
        """
        Calculates the area of the square.

        Returns:
            The area of the square.
        """
        return self.side**2

    def perimeter(self) -> float:
        """
        Calculates the perimeter of the square.

        Returns:
            The perimeter of the square.
        """
        return 4 * self.side


class Rectangle(Shape):
    """
    Represents a rectangle. NOTE: This class is intentionally incomplete.

    It inherits from the Shape ABC but does not implement the abstract methods
    `area` and `perimeter`. Therefore, it is also considered an abstract class,
    and you cannot create an instance of it.
    """

    def __init__(self, width: float, height: float):
        """
        Initializes a Rectangle with a given width and height.

        Args:
            width: The width of the rectangle.
            height: The height of the rectangle.
        """
        self.width = width
        self.height = height


# --- Example Usage ---

# Create an instance of a Circle
c = Circle(5)
print("Circle with radius 5:")
print(f"  Area: {c.area():.2f}")
print(f"  Perimeter: {c.perimeter():.2f}")

# Create an instance of a Square
s = Square(4)
print("\nSquare with side 4:")
print(f"  Area: {s.area():.2f}")
print(f"  Perimeter: {s.perimeter():.2f}")

# The following lines will cause a TypeError because the `Rectangle` class does
# not implement the abstract methods `area` and `perimeter` from the `Shape` ABC.
# This demonstrates that Python prevents you from instantiating a class that
# has not fulfilled the contract of its abstract base class.
try:
    r = Rectangle(5, 2)
except TypeError as e:
    print("\nAttempting to create a Rectangle instance...")
    print(f"  Caught expected error: {e}")

# Similarly, you cannot instantiate the ABC `Shape` itself.
try:
    shape = Shape()
except TypeError as e:
    print("\nAttempting to create a Shape instance...")
    print(f"  Caught expected error: {e}")
