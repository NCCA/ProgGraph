#!/usr/bin/env -S uv run --script
"""
This script demonstrates the Strategy design pattern in Python.

The Strategy pattern is a behavioral design pattern that enables selecting
an algorithm at runtime. Instead of implementing a single algorithm directly,
code receives runtime instructions as to which in a family of algorithms
to use.

In this example:
- `add` and `multiply` are two distinct algorithms (strategies).
- `calculate` is the context that uses one of the strategies.
- We can change the behavior of `calculate` by passing it a different
  strategy function, effectively changing the algorithm it uses.
"""

from typing import Callable

# 1. Define a Type for the Strategy
# For clarity and reusability, we can define a type alias for our
# strategy function signature. A strategy in this case is any function
# that takes two floats and returns a float.
Strategy = Callable[[float, float], float]


# 2. Define Concrete Strategies
# These are the specific implementations of the algorithm (the "family of algorithms").
# Each function adheres to the `Strategy` signature.


def add(x: float, y: float) -> float:
    """A concrete strategy that performs addition."""
    return x + y


def multiply(x: float, y: float) -> float:
    """A concrete strategy that performs multiplication."""
    return x * y


# 3. Define the Context
# The context is the object or function that uses a strategy. It is configured
# with a concrete strategy object and maintains a reference to it.
def calculate(a: float, b: float, strategy: Strategy) -> float:
    """
    This is the context function. It executes a calculation based on the
    provided strategy without knowing the details of the strategy itself.

    Args:
        a: The first number.
        b: The second number.
        strategy: The strategy function to use for the calculation.

    Returns:
        The result of applying the strategy to the numbers.
    """
    print(f"Calculating {a} and {b} using strategy: {strategy.__name__}")
    return strategy(a, b)


# 4. Client Code: Using the Strategies
# The client code selects a concrete strategy and passes it to the context.
# This allows the behavior of the `calculate` function to be changed at runtime.

# Use the 'add' strategy
print("--- Using the 'add' strategy ---")
result_add = calculate(2, 3, add)
print(f"Result: {result_add}\n")  # Expected output: 5

# Use the 'multiply' strategy
print("--- Using the 'multiply' strategy ---")
result_multiply = calculate(2, 3, multiply)
print(f"Result: {result_multiply}\n")  # Expected output: 6
