"""Nodes module"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(slots=True)
class Nodes:
    """Node representation with attributes like coordinates and demand."""
    _id: str
    _node_type: str
    _coordinates: Tuple[float, float]
    _demand: float
    isVisited: bool = False

    def __repr__(self):
        return self.id

    @property
    def id(self) -> str:
        return self._id

    @property
    def node_type(self) -> str:
        return self._node_type

    @property
    def coordinates(self) -> Tuple[float, float]:
        return self._coordinates

    @property
    def X(self) -> float:
        return self._coordinates[0]

    @property
    def Y(self) -> float:
        return self._coordinates[1]

    def __eq__(self, other):
        if isinstance(other, Nodes):
            return self.id == other.id
        return False

    def __hash__(self):
        return hash(self.id)