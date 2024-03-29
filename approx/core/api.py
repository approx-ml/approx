"""
Contains stuff to be exported in the public API.
"""

from typing import Any

from approx.core import _vars
from approx.core.backend.common import auto_select_backend
from approx.core.compare import CompareResult, EvalLoop, _CompareRunner
from approx.core.device.common import auto_select_device


def auto_quantize(model: Any, pretrained: bool = True) -> Any:
    """Turns a normal model into a quantized model, using an appropriate backend

    Args:
        model: The model to quantize.
        pretrained: Whether this model is pretrained
    """
    if _vars._APPROX_BACKEND is None:
        raise ValueError(
            "No backend has been set. "
            "Please call `approx.auto_set_backend()`."
        )
    qmodel = _vars._APPROX_BACKEND.auto_quantize(model, pretrained)
    return qmodel


def compare(
    model: Any, quantized_model: Any, test_loader: Any, *, eval_loop: EvalLoop
) -> CompareResult:
    """
    Compares your normal model with your quantized model

    Args:
        model: Your normal model
        quantized_model: Your quantized model
        test_loader: The "dataloader" to be used for testing.
        eval_loop: Your evaluation loop that operates on your model and it's data,
                   and returns a dictionary which maps each metric to its history

    Returns:
        Useful statistical information
    """
    runner = _CompareRunner([model, quantized_model], test_loader, eval_loop)
    return runner.run()


def auto_set_backend() -> None:
    """Automatically sets an appropriate backend for `approx` to use.

    Returns:
        None.
    """

    _vars._APPROX_BACKEND = auto_select_backend()


def auto_set_device() -> None:
    """Automatically sets an appropriate device for `approx` to use.

    Returns:
        None.
    """
    _vars._APPROX_DEVICE = auto_select_device()
