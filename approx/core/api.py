"""
Contains stuff to be exported in the public API.
"""
from contextlib import contextmanager

from approx.core.backends.common import auto_select_backend


@contextmanager
def auto_cast_all() -> None:
    """
    Context manager that automatically casts and applies quantization to
    any tensors created within the context.

    Returns:
        Context manager
    """
    pass


def auto_set_backend() -> None:
    """
    Automatically selects an appropriate backend to utilize.
    Returns:
        None
    """
    auto_select_backend()
