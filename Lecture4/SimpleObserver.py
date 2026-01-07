#!/usr/bin/env -S uv run --script
"""
This script demonstrates the Observer design pattern.

The Observer pattern is a behavioral design pattern where an object, called the
subject, maintains a list of its dependents, called observers, and notifies
them automatically of any state changes, usually by calling one of their methods.

It is used to establish a one-to-many dependency between objects without
making the objects tightly coupled.

Components in this example:
- Observer (ABC): An interface for objects that should be notified of changes.
- Subject: The object that maintains a list of observers and notifies them.
- PrintObserver: A concrete implementation of an Observer that reacts to changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


# 1. Define the Observer Interface (the "Subscriber")
class Observer(ABC):
    """
    The Observer interface declares the `update` method, which is what the
    subject will call to notify the observer of a change.
    """

    @abstractmethod
    def update(self, data: Any) -> None:
        """Receive update from the subject."""
        pass


# 2. Implement the Subject (the "Publisher")
class Subject:
    """
    The Subject owns some important state and notifies observers when the state
    changes. It is responsible for managing its list of observers.
    """

    def __init__(self):
        # This list holds the observer objects that are "subscribed".
        self._observers: List[Observer] = []

    def register(self, observer: Observer) -> None:
        """
        Adds an observer to the subject's list of subscribers.
        """
        print(f"Subject: Registered an observer: {observer.__class__.__name__}")
        self._observers.append(observer)

    def notify(self, message: Any) -> None:
        """
        Notifies all registered observers by calling their `update` method.
        """
        print(f'Subject: Notifying observers with message: "{message}"')
        for observer in self._observers:
            observer.update(message)


# 3. Implement a Concrete Observer
class PrintObserver(Observer):
    """
    A Concrete Observer implements the Observer interface and defines what
    action to take when it is updated by the subject.
    """

    def update(self, data: Any) -> None:
        """
        Processes the notification from the subject by printing the data.
        """
        print(f'PrintObserver: Received data: "{data}"')


# 4. Client Code: Using the Pattern
if __name__ == "__main__":
    # Create the subject object.
    subject = Subject()

    # Create a concrete observer object.
    print_observer = PrintObserver()

    # Register the observer with the subject. Now the subject knows who
    # to notify when something happens.
    subject.register(print_observer)

    # The subject's state changes or an event occurs, and it notifies
    # all its registered observers.
    subject.notify("Hello, World!")

    # You can create and register more observers of different types.
    class AnotherObserver(Observer):
        def update(self, data: Any) -> None:
            print(f"AnotherObserver: Got this data: '{data}'")

    another_observer = AnotherObserver()
    subject.register(another_observer)

    # When notify is called again, all registered observers receive the message.
    subject.notify("A second notification for everyone!")
