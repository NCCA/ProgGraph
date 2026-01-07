#!/usr/bin/env -S uv run --script
"""
This script demonstrates the Singleton design pattern using a metaclass.

The Singleton pattern ensures that a class has only one instance and provides a
global point of access to it. This is useful for managing shared resources like
a database connection, a configuration manager, or a hardware interface, where
having multiple instances could lead to conflicts or inconsistent state.

This implementation uses a metaclass, which is a class for creating classes.
By defining the Singleton logic in the metaclass, we can make any class a
Singleton simply by specifying it as its metaclass.
"""

from typing import Any, Dict, Type


class SingletonMeta(type):
    """
    A metaclass to implement the Singleton pattern.

    This metaclass overrides the class creation process. When a class using this
    metaclass is instantiated for the first time, the instance is created and
    stored in a private dictionary. On subsequent instantiation attempts, the
    existing instance is returned instead of a new one.
    """

    # A private dictionary to store the single instance of each class.
    _instances: Dict[Type, Any] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        This method is called when you try to create an instance of a class,
        e.g., `Config()`.

        If the class (`cls`) is not already in our `_instances` dictionary, it means
        this is the first time it's being instantiated. We create the instance
        using `super().__call__` and store it.

        If the class is already in the dictionary, we simply return the stored instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ConfigManager(metaclass=SingletonMeta):
    """
    A configuration manager class implemented as a Singleton.

    No matter how many times you try to create an instance of `ConfigManager`,
    you will always get the same object. This ensures that all parts of an
    application access the same configuration state.
    """

    def __init__(self) -> None:
        """
        Initializes the configuration settings.

        IMPORTANT: The `__init__` method is called *every time* you attempt to
        create an instance (e.g., `ConfigManager()`), even though a new instance
        is not created after the first time. Therefore, it's best used for
        initial setup that is safe to run multiple times, or to include checks
        to prevent re-initialization if necessary.
        """
        print("(Running __init__...)")
        self.database_url = "postgres://user:pass@host/db"
        self.api_key = "default_api_key"


# --- Example Usage ---

print("--- First attempt to create a ConfigManager instance ---")
c1 = ConfigManager()

print("\n--- Second attempt to create a ConfigManager instance ---")
c2 = ConfigManager()

# 1. Verify that both variables point to the exact same object.
# The `is` operator checks if two variables refer to the same object in memory.
print(f"\nAre c1 and c2 the same object? {c1 is c2}")  # Expected: True

# 2. Check their memory addresses.
# The `id()` function returns the memory address of an object. They should be identical.
print(f"ID of c1: {hex(id(c1))}")
print(f"ID of c2: {hex(id(c2))}")

# 3. Modify the state through one variable and check the other.
# This demonstrates that they share the same state because they are the same instance.
print("\n--- Modifying state ---")
print(f"Original API key from c1: {c1.api_key}")

# Modify the api_key using the c2 variable
c2.api_key = "new_production_api_key"
print("Changed api_key using c2.")

# The change is reflected when accessing the attribute via c1.
print(f"New API key from c1: {c1.api_key}")  # Expected: new_production_api_key
