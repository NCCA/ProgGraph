#!/usr/bin/env -S uv run --script
"""
A decorator is a design pattern in Python that allows you to add new
functionality to an existing function or class without modifying its
source code.

In this example, we create a simple decorator called `uppercase` that
takes the string output of a function and converts it to uppercase.
"""

from typing import Any, Callable


# 1. Defining the Decorator
# A decorator is a function that takes another function as an argument,
# adds some functionality, and then returns another function.
def uppercase(func: Callable[..., str]) -> Callable[..., str]:
    """
    This is the decorator function.
    It takes a function `func` that returns a string and returns a new
    function that wraps `func` and converts its result to uppercase.

    Args:
        func: The function to be decorated. It is expected to return a string.

    Returns:
        The wrapper function.
    """

    # 2. Defining the Wrapper Function
    # This inner function "wraps" the original function. It's the place
    # where you can run code before and after the original function executes.
    def wrapper(*args: Any, **kwargs: Any) -> str:
        """
        This wrapper function is what gets executed instead of the original
        `greet` function. It can accept any arguments that `greet` would.
        """
        # Call the original function that was passed into the decorator.
        original_result = func(*args, **kwargs)
        # Modify the result of the original function.
        modified_result = original_result.upper()
        # Return the new, modified result.
        return modified_result

    # 3. Returning the Wrapper
    # The decorator must return the wrapper function. This returned function
    # will replace the original decorated function.
    return wrapper


# 4. Applying the Decorator
# The `@` symbol is Python's special syntax for applying a decorator to a
# function. The line `@uppercase` directly above `def greet(...)` is
# syntactic sugar that is equivalent to writing:
# greet = uppercase(greet)
#
# This means the `greet` variable no longer holds a reference to the original
# greet function, but to the `wrapper` function returned by `uppercase`.
@uppercase
def greet(name: str) -> str:
    """
    A simple function that returns a greeting message.
    This function is decorated by `uppercase`.
    """
    return f"Hello, {name}"


# 5. Calling the Decorated Function
# When we call `greet("World")`, we are not calling the original `greet`
# function. Instead, we are invoking the `wrapper` function. The `wrapper`
# then calls the original `greet` function internally and modifies its result.
print(greet("World"))
