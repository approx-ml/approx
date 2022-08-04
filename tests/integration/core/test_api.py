"""
Test Public API
"""
import torch.nn


def test_auto_set_backend():
    """Tests that the auto_set_backend function works as expected."""
    import approx
    from approx.core.backend._backend import BackendEngine

    approx.auto_set_backend()

    assert isinstance(approx.backend(), BackendEngine)
    assert str(approx.backend()) == "torch"


def test_set_backend():
    """Tests that the set_backend function works as expected."""
    from approx.core.backend import set_backend
    from approx.core.backend._backend import BackendType

    backend_types = [
        attr
        for attr in dir(BackendType)
        if (
            not callable(getattr(BackendType, attr))
            and not attr.startswith("__")
        )
    ]
    for backend_type in backend_types:
        if backend_type.lower() == "unknown":
            continue
        backend_engine = set_backend(BackendType[backend_type])
        assert backend_engine.type == BackendType[backend_type]
        assert str(backend_engine) == backend_type.lower()


def test_set_device():
    """Tests that the set_device function works as expected."""
    from approx.core.device import set_device
    from approx.core.device._device import DeviceType

    device_types = [
        attr
        for attr in dir(DeviceType)
        if (
            not callable(getattr(DeviceType, attr))
            and not attr.startswith("__")
        )
    ]
    for device_type in device_types:
        if device_type.lower() == "unknown":
            continue
        device_engine = set_device(DeviceType[device_type])
        assert device_engine.type == DeviceType[device_type]
        assert str(device_engine) == device_type.lower()


def test_compare(model_with_random_data):
    import approx

    model_with_random_data.train_model()
    compare_result = approx.compare(
        model_with_random_data.model,
        model_with_random_data.model,
        model_with_random_data.test_dl,
        eval_loop=model_with_random_data.eval_loop,
    )
    assert len(compare_result) == 2
    acc = compare_result.mean("accuracy")
    loss = compare_result.mean("loss")
    assert len(acc.values()) == 1
    assert len(loss.values()) == 1


def test_auto_quantize(model_with_random_data):
    import torch

    import approx

    # enforce pytorch
    approx.core._vars._APPROX_BACKEND = (
        approx.core.backend._backend.PyTorchBackend()
    )
    model_with_random_data.train_model()
    quantized_model = approx.auto_quantize(
        model_with_random_data.model, pretrained=True
    )
    assert isinstance(quantized_model, torch.nn.Module)
    layers = [l for l in quantized_model.net]
    quantized_layers = list(
        filter(lambda l: isinstance(l, torch.nn.quantized.Linear), layers)
    )
    assert len(quantized_layers) == 3
