#!/usr/bin/env -S uv run --script
"""
This script demonstrates the use of Mixin classes in Python.

Mixins are a form of multiple inheritance where a class provides a specific,
self-contained piece of functionality that can be "mixed in" with other classes.
They are not meant to be instantiated on their own but are designed to add
reusable behaviors to various classes without creating complex inheritance hierarchies.

In this example, we have three mixins:
- `LoggingMixin`: Adds a simple logging method.
- `JsonMixin`: Adds a method to serialize the object to a JSON string.
- `ReprMixin`: Provides a developer-friendly string representation of the object.

These are then combined into the `Product` class.
"""

import json
import logging

# Configure basic logging to print messages to the console.
logging.basicConfig(level=logging.INFO, format="%(message)s")


# --- Mixin Classes ---


class LoggingMixin:
    """
    A mixin class that provides a `log` method.

    Any class that inherits from this mixin will have access to this method.
    It assumes the inheriting class will have a `__class__.__name__` attribute.
    """

    def log(self, message: str) -> None:
        """Logs a message with the class name prefixed."""
        logging.info(f"{self.__class__.__name__}: {message}")


class JsonMixin:
    """
    A mixin class that provides a `to_json` method.

    This mixin serializes the instance's `__dict__` (its attributes) into a
    JSON formatted string. It relies on the `json` standard library.
    """

    def to_json(self) -> str:
        """Converts the object's attributes to a JSON string."""
        return json.dumps(self.__dict__)


class ReprMixin:
    """
    A mixin class that provides a custom `__repr__` method.

    This provides a more informative string representation for an object,
    which is useful for debugging and logging.
    """

    def __repr__(self) -> str:
        """Returns a developer-friendly representation of the object."""
        attributes = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"<{self.__class__.__name__}({attributes})>"


# --- Main Business Class ---


class Product(LoggingMixin, JsonMixin, ReprMixin):
    """
    A simple class representing a product.

    This class inherits from all three mixins, thereby gaining their functionality.
    The order of inheritance matters for method resolution, but in this case,
    the mixins provide non-overlapping methods (except for `__repr__` which
    overrides the default `object.__repr__`).
    """

    def __init__(self, name: str, price: float):
        """
        Initializes a Product instance.

        Args:
            name: The name of the product.
            price: The price of the product.
        """
        self.name = name
        self.price = price


# --- Example Usage ---

# Create an instance of the Product class.
p = Product("Super Widget", 9.99)

# 1. Using the `ReprMixin` functionality
# When we print the object, the custom `__repr__` from ReprMixin is called.
print(f"Object representation: {p}")
# Expected output: <Product(name='Super Widget', price=9.99)>

# 2. Using the `JsonMixin` functionality
# We can now call `to_json` because Product inherited it from JsonMixin.
json_output = p.to_json()
print(f"JSON representation: {json_output}")
# Expected output: {"name": "Super Widget", "price": 9.99}

# 3. Using the `LoggingMixin` functionality
# We can call `log` because Product inherited it from LoggingMixin.
p.log("Product instance created successfully.")
# Expected output: Product: Product instance created successfully.
