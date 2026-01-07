#!/usr/bin/env -S uv run --script

"""
This script provides a Python implementation of the Command Design Pattern.

The Command Pattern is a behavioral design pattern that turns a request into a
stand-alone object containing all information about the request. This
transformation lets you parameterize methods with different requests, delay or
queue a request's execution, and support undoable operations.

This example, based on the provided C++ code, demonstrates the pattern by
simulating agents that can move around in a 2D space.

The key components are:
- Command (ABC): An interface for executing an operation.
- ConcreteCommand: Implements the Command interface, binding an action to a Receiver.
- Receiver (Agent): The object that performs the actual work.
- Invoker (InputProcessor): Asks the command to carry out the request.
- Client: Creates a ConcreteCommand object and sets its receiver.
"""

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Receiver: The object that will receive and act upon commands.
# -----------------------------------------------------------------------------


@dataclass
class Point2:
    """A simple 2D point to represent a location."""

    x: float = 0.0
    y: float = 0.0


class Agent:
    """
    The Receiver class. It knows how to perform the operations associated
    with carrying out a request. Any class can serve as a Receiver.
    """

    def __init__(self, pos: Point2, name: str):
        """
        Initializes an Agent.

        Args:
            pos: The starting position of the agent.
            name: The name of the agent for identification.
        """
        self._pos = pos
        self._name = name
        self._speed = 1.0  # Default speed is 'walk'
        print(f"{self._name} created at position ({self._pos.x}, {self._pos.y})")

    def run(self) -> None:
        """Sets the agent's speed to 'run' (2.0)."""
        print(f"{self._name} started running.")
        self._speed = 2.0

    def walk(self) -> None:
        """Sets the agent's speed to 'walk' (1.0)."""
        print(f"{self._name} started walking.")
        self._speed = 1.0

    def up(self) -> None:
        """Moves the agent up."""
        self._pos.y += self._speed

    def down(self) -> None:
        """Moves the agent down."""
        self._pos.y -= self._speed

    def left(self) -> None:
        """Moves the agent left."""
        self._pos.x -= self._speed

    def right(self) -> None:
        """Moves the agent right."""
        self._pos.x += self._speed

    def debug(self) -> None:
        """Prints the current state of the agent."""
        print(
            f"DEBUG - {self._name} is at ({self._pos.x:.1f}, {self._pos.y:.1f}) with speed {self._speed:.1f}"
        )


# -----------------------------------------------------------------------------
# Command Interface and Concrete Commands
# -----------------------------------------------------------------------------


class Command(ABC):
    """
    The Command interface declares a method for executing a command.
    """

    @abstractmethod
    def execute(self, agent: Agent) -> None:
        """
        This is the core method of the command pattern that, when implemented,
        will execute the specific action on the receiver.

        Args:
            agent: The receiver object to act upon.
        """
        pass


# The following are ConcreteCommand classes. They define a binding between an
# action and a Receiver. The Invoker calls execute() to issue the request.


class RunCommand(Command):
    """Command to make the agent run."""

    def execute(self, agent: Agent) -> None:
        agent.run()


class WalkCommand(Command):
    """Command to make the agent walk."""

    def execute(self, agent: Agent) -> None:
        agent.walk()


class LeftCommand(Command):
    """Command to move the agent left."""

    def execute(self, agent: Agent) -> None:
        agent.left()


class RightCommand(Command):
    """Command to move the agent right."""

    def execute(self, agent: Agent) -> None:
        agent.right()


class UpCommand(Command):
    """Command to move the agent up."""

    def execute(self, agent: Agent) -> None:
        agent.up()


class DownCommand(Command):
    """Command to move the agent down."""

    def execute(self, agent: Agent) -> None:
        agent.down()


# -----------------------------------------------------------------------------
# Invoker: The object that knows how to execute a command.
# -----------------------------------------------------------------------------


class Moves(Enum):
    """An enumeration for the possible moves, mapping to commands."""

    RUN = auto()
    WALK = auto()
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class InputProcessor:
    """
    The Invoker class. It is responsible for initiating requests.
    It holds a command object and at some point asks the command to carry out
    the request by calling its execute() method.
    """

    def __init__(self):
        """
        Initializes the InputProcessor, creating a mapping from moves to
        concrete command objects.
        """
        self._commands: Dict[Moves, Command] = {
            Moves.RUN: RunCommand(),
            Moves.WALK: WalkCommand(),
            Moves.LEFT: LeftCommand(),
            Moves.RIGHT: RightCommand(),
            Moves.UP: UpCommand(),
            Moves.DOWN: DownCommand(),
        }

    def handle_input(self, move: Moves) -> Optional[Command]:
        """
        Given a move, returns the corresponding command object.

        Args:
            move: The move to be processed.

        Returns:
            The command object associated with the move, or None if not found.
        """
        return self._commands.get(move)


# -----------------------------------------------------------------------------
# Client: The code that sets up and runs the simulation.
# -----------------------------------------------------------------------------


def main() -> int:
    """
    The main client code that wires everything together.
    """
    print("--- Setting up the simulation ---")

    # 1. Create a sequence of moves. This could come from user input, AI,
    #    or be read from a file.
    input_sequence: List[Moves] = [
        Moves.WALK,
        Moves.RIGHT,
        Moves.RUN,
        Moves.UP,
        Moves.RIGHT,
        Moves.DOWN,
        Moves.WALK,
        Moves.DOWN,
        Moves.LEFT,
        Moves.LEFT,
    ]
    print(f"Input sequence: {[move.name for move in input_sequence]}\n")

    # 2. Create the receivers (the agents that will be commanded).
    agents: List[Agent] = [
        Agent(Point2(0.0, 0.0), "Agent 1"),
        Agent(Point2(0.0, 2.0), "Agent 2"),
        Agent(Point2(3.0, 0.0), "Agent 3"),
    ]
    print("\n--- Starting simulation ---")

    # 3. Create the invoker.
    processor = InputProcessor()

    # 4. Loop through the inputs and execute commands on all agents.
    for i, move in enumerate(input_sequence):
        print(f"\nStep {i + 1}: Processing move '{move.name}'")
        command = processor.handle_input(move)
        if command:
            # The same command is executed on all agents.
            for agent in agents:
                command.execute(agent)
                agent.debug()
        else:
            print(f"Warning: No command found for move '{move.name}'")

    print("\n--- Simulation finished ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
