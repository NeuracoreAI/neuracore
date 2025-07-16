"""Singleton metaclass for creating singleton classes.

This metaclass ensures that only one instance of a class can be created.
"""

from abc import ABCMeta
from typing import Any


class SingletonMetaclass(ABCMeta):
    """Metaclass for creating singleton classes.

    This metaclass ensures that only one instance of a class can be created.
    """

    _instances: dict[type, object] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> object:
        """Called when a new instance of the class is requested.

        If an instance of the class does not already exist, it creates one and
        stores it. Otherwise, it returns the existing instance.

        Args:
            *args: Positional arguments passed to the class constructor.
            **kwargs: Keyword arguments passed to the class constructor.

        Returns:
            The single instance of the class.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]
