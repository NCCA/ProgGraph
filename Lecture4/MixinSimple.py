#!/usr/bin/env -S uv run --script
"""
This script demonstrates the Mixin pattern in Python.

A Mixin is a class that provides a specific piece of functionality, intended
to be "mixed in" with other classes through inheritance. Mixins are a way to
compose classes and add reusable behavior without creating deep, complex
inheritance trees.

Key characteristics of a Mixin:
- It is not meant to be instantiated on its own.
- It provides a set of methods that other classes can use.
- A class can inherit from one or more mixins to gain different functionalities.

In this example, `LoggingMixin` provides a `log` method that any other
class can use simply by inheriting from it.
"""

import logging
from typing import Any

# 0. Configure Logging
# By default, the logging module only shows messages of level WARNING or higher.
# We configure it to show INFO level messages so we can see our log output.
logging.basicConfig(level=logging.INFO, format="%(message)s")


# 1. Define the Mixin Class
class LoggingMixin:
    """
    This is a Mixin class that provides logging functionality.
    It is not intended to be used alone. Notice it doesn't have an __init__
    method for its own state. It acts on the state of the class it's mixed into.
    The `self` in this class will refer to an instance of the class that
    inherits this mixin (e.g., an instance of `User`).
    """

    # The `__class__` attribute is available on `self`.
    # We add a type hint here to indicate that any class using this mixin
    # is expected to have this attribute, which all classes do.
    __class__: Any

    def log(self, message: str) -> None:
        """Logs a message with the name of the class that is using the mixin."""
        # `self.__class__.__name__` gets the name of the actual class instance
        # that is calling this method, for example, 'User'.
        logging.info(f"LOG - {self.__class__.__name__}: {message}")


# 2. Define a Class that Uses the Mixin
class User(LoggingMixin):
    """
    This is a regular class representing a user.
    By inheriting from `LoggingMixin`, it gains access to the `log` method
    as if it were its own. This is a form of code reuse.
    """

    def __init__(self, name: str) -> None:
        """Initializes the User with a name."""
        self.name = name
        # We can use the log method right from the start!
        # `self` here is the instance of User.
        self.log(f"User object '{self.name}' created.")

    def greet(self) -> str:
        """
        A method that performs a greeting and uses the mixed-in log method.
        """
        # Here we call `self.log`, which is defined in `LoggingMixin`.
        self.log(f"Greeting user '{self.name}'")
        return f"Hello, {self.name}"


# 3. Client Code
if __name__ == "__main__":
    print("Creating a User instance...")
    # When User("Jon") is created, its __init__ method is called, which
    # in turn calls the `log` method inherited from LoggingMixin.
    u = User("Jon")
    print("-" * 20)

    print("Calling the greet method...")
    # The greet method also uses the `log` method from the mixin.
    greeting_message = u.greet()
    print(f'Greet method returned: "{greeting_message}"')
