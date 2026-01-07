#!/usr/bin/env -S uv run --script
"""
This script demonstrates the Adapter design pattern.

The Adapter pattern allows objects with incompatible interfaces to collaborate.
It acts as a wrapper between two objects, catching calls for one object and
transforming them into a format and interface recognizable by the other.

In this example, we have an `OldPrinter` class that we cannot or do not want to
change. We want to use it in a new system that expects a `request` method, not
a `print_text` method. The `NewPrinterAdapter` class makes this possible.
"""


class OldPrinter:
    """
    The Adaptee class (also known as the Service).

    This is the class that has an existing, specific interface that our client
    code cannot directly use.
    """

    def print_text(self, text: str) -> None:
        """
        The original method of the adaptee that prints a given text.

        Args:
            text: The string to be printed.
        """
        print(f"OldPrinter prints: {text}")


class NewPrinterAdapter:
    """
    The Adapter class.

    This class wraps an instance of the `OldPrinter` (the adaptee) and exposes
    a new, more modern interface that the client code expects to use.
    """

    def __init__(self, old_printer: OldPrinter):
        """
        Initializes the adapter with an instance of the adaptee.

        Args:
            old_printer: An instance of the OldPrinter class to be adapted.
        """
        self._old_printer = old_printer

    def request(self, text: str) -> None:
        """
        The target interface method that the client code will call.

        This method translates the call into a call to the original `print_text`
        method on the wrapped `OldPrinter` instance.

        Args:
            text: The string to be processed.
        """
        print("Adapter's 'request' method called...")
        # The adapter forwards the call to the adaptee's method.
        self._old_printer.print_text(text)


# --- Client Code Example ---

# 1. We have an instance of the old class, which we cannot change.
old_printer_instance = OldPrinter()

# 2. We create an adapter and pass the old object to it.
# The adapter now 'wraps' the old printer, giving it a new interface.
new_adapter = NewPrinterAdapter(old_printer_instance)

# 3. The client code can now use the adapter's `request` method.
# Internally, the adapter calls the original `print_text` method, but the
# client doesn't need to know about that complexity.
print("Client making a request...")
new_adapter.request("Hello, World!")
