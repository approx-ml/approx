import enum
import sys
import abc


class BackendNotSupported(Exception):
    pass


class BackendEngine(abc.ABC):
    pass


class BackendChoice(enum.Enum):
    TORCH = 0


def auto_select_backend() -> BackendEngine:
    """
    Automatically selects the backend necessary

    Returns:
        An instance of whatever backend is most appropriate.
    """
    # don't import the backends until it is actually selected
    if "torch" in sys.modules:
        return set_backend(BackendChoice.TORCH)
    raise BackendNotSupported(
        f"This backend is currently not supported. "
        f"Supported backends: {[b.name for b in BackendChoice]}"
    )


def set_backend(choice: BackendChoice) -> BackendEngine:
    if choice == BackendChoice.TORCH:
        from approx.core.backends.pytorch import PyTorchBackend

        return PyTorchBackend()
