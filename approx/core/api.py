"""
Contains stuff to be exported in the public API.
"""

from typing import Any, Callable, List, Optional, Dict

from approx.core import _vars
from approx.core.backend.common import auto_select_backend
from approx.core.compare import CompareResult, Metric, _CompareRunner
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
    model: Any,
    quantized_model: Any,
    test_loader: Any,
    *,
    eval_loop: Callable[[Any], Dict[Metric, List[float]]]
) -> CompareResult:
    """
    Compares your normal model with your quantized model

    Args:
        model: Your normal model
        quantized_model: Your quantized model
        eval_loop: Your evaluation loop that operates on your model and returns
                   a dictionary which maps each metric to its history

    Notes:
        For certain backends, we support automatic generation of the eval loop given the
        model metrics.

    Returns:
        Useful statistical information
    """
    runner = _CompareRunner(
        [model, quantized_model], test_loader, eval_loop
    )
    return runner.run()


def auto_cast_all(*args: Any) -> None:
    """Automatically casts any tensors for automatic quantization

    Notes:
        This function is deprecated and planned for removal.

    Returns:
        None:
    """
    # todo: unimplemented
    pass


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
