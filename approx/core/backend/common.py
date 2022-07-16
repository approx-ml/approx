"""
Common backend handling
"""

import importlib.util

from approx.core.backend._backend import (
    BackendEngine,
    BackendNotSupported,
    BackendType,
)


def auto_select_backend() -> BackendEngine:
    """Automatically selects the backend necessary

    Raises:
        BackendNotSupported: Could not find a backend.

    Returns:
        BackendEngine: An instance of whatever backend is most appropriate.
    """
    backend_types = [
        attr
        for attr in dir(BackendType)
        if (
            not callable(getattr(BackendType, attr))
            and not attr.startswith("__")
        )
    ]

    preferred_backends = ["TORCH"]
    for backend_type in preferred_backends:
        if backend_type.lower() == "unknown":
            continue
        if importlib.util.find_spec(backend_type.lower()) is not None:
            return set_backend(BackendType[backend_type])

    # fallback
    for backend_type in backend_types:
        if backend_type.lower() == "unknown":
            continue
        if importlib.util.find_spec(backend_type.lower()) is not None:
            return set_backend(BackendType[backend_type])

    raise BackendNotSupported(
        f"This backend is currently not supported. "
        f"Supported backends: {[b.name for b in BackendType]}"
    )


def set_backend(choice: BackendType) -> BackendEngine:
    """Sets the backend to use.

    Args:
        choice (BackendType): The backend to use. (e.g. BackendType.TORCH)

    Raises:
        BackendNotSupported: When the backend is not supported.

    Returns:
        BackendEngine: An instance of whatever backend is most appropriate.
    """

    if choice == BackendType.TORCH:
        from approx.core.backend._backend import PyTorchBackend

        return PyTorchBackend()

    elif choice == BackendType.NUMPY:
        from approx.core.backend._backend import NumPyBackend

        return NumPyBackend()

    # if asked for a backend that doesn't exist, raise an error
    raise BackendNotSupported(str(choice))
