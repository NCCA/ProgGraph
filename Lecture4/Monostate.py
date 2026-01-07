#!/usr/bin/env -S uv run --script
"""
This script demonstrates the Monostate (also known as the Borg) design pattern.

The Monostate pattern ensures that all instances of a class share the same state.
Unlike the Singleton pattern, which ensures there is only one instance of a class,
the Monostate pattern allows multiple instances to be created, but they all
manipulate the same underlying dictionary of attributes.

This is achieved by overriding the special `__dict__` attribute of each instance
to point to a single, shared, class-level dictionary (`_shared_state`).
"""

from typing import Any, Dict


class Monostate:
    """
    A class that implements the Monostate pattern.

    All instances of this class will share the same state. Any modification
    to one instance's attributes will be instantly reflected in all others.
    """

    # A class attribute that will be shared across all instances.
    _shared_state: Dict[str, Any] = {}

    def __init__(self):
        """
        Initializes a new instance of Monostate.

        The magic happens here: we replace the instance's private dictionary
        (`__dict__`) with the shared class-level dictionary. From this point on,
        any attribute access (e.g., `self.x = 10`) will read from or write to
        the `_shared_state` dictionary.
        """
        self.__dict__ = self._shared_state


# --- Example Usage ---

print("--- Initial State ---")
# At the beginning, the shared state is empty.
print(f"Monostate._shared_state is initially: {Monostate._shared_state}")

# Create two instances of the Monostate class.
# Note that these are two separate objects in memory.
a = Monostate()
b = Monostate()

print(f"\nAre 'a' and 'b' the same object? {a is b}")  # False

# Assign an attribute to the first instance, 'a'.
a.x = 5
print("\n--- After `a.x = 5` ---")

# Because they share state, the change is reflected in both instances.
print(f"a.x = {a.x}")
print(f"b.x = {b.x}")

# The underlying shared state dictionary now contains 'x'.
print(f"Monostate._shared_state is now: {Monostate._shared_state}")

# Assign a different attribute to the second instance, 'b'.
b.y = 99
print("\n--- After `b.y = 99` ---")

# The change is again reflected in both instances.
print(f"a.y = {a.y}")
print(f"b.y = {b.y}")

# The shared state now contains both 'x' and 'y'.
print(f"Monostate._shared_state is now: {Monostate._shared_state}")

# You can also see the instance dictionaries are the same object.
print(f"\nAre the __dict__ of 'a' and 'b' the same? {a.__dict__ is b.__dict__}")  # True
