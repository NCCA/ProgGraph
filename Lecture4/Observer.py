#!/usr/bin/env -S uv run --script
"""
This script provides a more advanced example of the Observer design pattern.

This implementation demonstrates a "pull" style observer pattern. In this style,
the subject passes a reference to itself (`self`) to the observers. The observers
are then responsible for "pulling" the state they need from the subject. This
contrasts with a "push" style, where the subject "pushes" the updated data
directly to the observers in the notify call.

Pattern Components:
- Subject: Manages a list of observers and its own state. When its state
  changes, it notifies all registered observers.
- Observer (ABC): An interface for observer objects, defining the `update` method.
- DivObserver & ModObserver: Concrete observers that perform actions based on the
  subject's state, which they pull from the subject in their `update` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional


# 1. The Observer Interface
class Observer(ABC):
    """
    The Observer interface declares the update method. All concrete observers
    must implement this method.
    """

    @abstractmethod
    def update(self, subject: Subject) -> None:
        """
        Receive an update from the subject. The subject instance itself is passed
        to allow the observer to pull the necessary data.
        """
        pass


# 2. The Subject Class
class Subject:
    """
    The Subject owns the important state and notifies observers when it changes.
    """

    # The state of the subject. Observers will be notified when this changes.
    # Using Optional[int] because it is initialized to None.
    _state: Optional[int] = None

    # A list to hold all registered observer objects.
    _observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        """
        Subscribes an observer to receive updates.
        """
        print(f"Subject: Attached an observer: {observer.__class__.__name__}")
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        """
        Unsubscribes an observer so it no longer receives updates.
        """
        print(f"Subject: Detached an observer: {observer.__class__.__name__}")
        self._observers.remove(observer)

    def notify(self) -> None:
        """
        Triggers an update in each subscribed observer, passing a reference
        to this subject instance (the "pull" model).
        """
        print("Subject: Notifying observers...")
        for observer in self._observers:
            observer.update(self)

    # 3. Using a Property to Automatically Trigger Notifications
    # The @property decorator allows us to treat a method like an attribute.
    # The .setter below defines the behavior for when this attribute is assigned.
    @property
    def state(self) -> Optional[int]:
        """Getter for the state attribute."""
        return self._state

    @state.setter
    def state(self, value: int) -> None:
        """
        Setter for the state attribute. This is the core of this implementation.
        Whenever the subject's state is changed (e.g., `subject.state = 10`),
        this method is executed. It first updates the state, then it automatically
        calls `notify()` to alert all observers of the change.
        """
        self._state = value
        print(f"Subject: State has changed to: {self._state}")
        self.notify()


# 4. Concrete Observer Implementations
class DivObserver(Observer):
    """
    A concrete observer that performs integer division on the subject's state.
    """

    def __init__(self, divisor: int):
        if divisor == 0:
            raise ValueError("Divisor cannot be zero")
        self._divisor = divisor

    def update(self, subject: Subject) -> None:
        """
        This method is called by the subject's `notify` method.
        It "pulls" the state from the subject and performs its calculation.
        """
        if subject.state is not None:
            result = subject.state // self._divisor
            print(
                f"  -> DivObserver({self._divisor}): {subject.state} // {self._divisor} = {result}"
            )


class ModObserver(Observer):
    """
    A concrete observer that performs a modulo operation on the subject's state.
    """

    def __init__(self, divisor: int):
        if divisor == 0:
            raise ValueError("Divisor cannot be zero")
        self._divisor = divisor

    def update(self, subject: Subject) -> None:
        """
        Pulls the state from the subject and performs its calculation.
        """
        if subject.state is not None:
            result = subject.state % self._divisor
            print(
                f"  -> ModObserver({self._divisor}): {subject.state} % {self._divisor} = {result}"
            )


# 5. Client Code
if __name__ == "__main__":
    # Create the subject instance.
    subject = Subject()

    # Create several observer instances.
    div_obs1 = DivObserver(4)
    div_obs2 = DivObserver(3)
    mod_obs3 = ModObserver(3)

    # Attach (subscribe) the observers to the subject.
    subject.attach(div_obs1)
    subject.attach(div_obs2)
    subject.attach(mod_obs3)
    print("-" * 20)

    # Change the subject's state. This single action will trigger the `state`
    # setter, which calls `notify()`, and all three attached observers will
    # have their `update` method called.
    subject.state = 14
    print("-" * 20)

    # Detach one of the observers.
    subject.detach(div_obs2)
    print("-" * 20)

    # Change the state again. Now, only the remaining two observers will be notified.
    subject.state = 25
    print("-" * 20)
