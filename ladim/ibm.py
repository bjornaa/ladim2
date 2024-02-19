"""Abstract Base Class for IBMs"""

from __future__ import annotations

from typing import Any


class IBM:
    """Base class for Individual Based Model"""

    def __init__(self, modules: dict[str, Any], **kwargs: dict[str, Any]):
        self.modules = modules
        self.opts = kwargs

    def update(self) -> None:
        """Updates the IBM to the next time step"""

    def close(self) -> None:
        """Perform cleanup procedures after simulation is finished"""
