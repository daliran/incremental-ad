from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import Self


class Configurable(ABC):

    @classmethod
    @abstractmethod
    def add_args(cls, parser: ArgumentParser, prefix: str | None = None) -> None:
        """Register this component's CLI arguments into the shared parser using ARG_PREFIX as namespace."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: Namespace, prefix: str | None = None) -> Self:
        """Construct an instance of this component from the parsed argument namespace."""
        ...
