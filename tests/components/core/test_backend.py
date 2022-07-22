"""
Test Backend
"""

import pytest
import torch

PYTORCH_QUANTIZED_MODULES = [
    # quantized modules
    "DynamicQuantizedLinear",
    "DynamicQuantizedRNNCell",
    "DynamicQuantizedLSTMCell",
    "DynamicQuantizedGRUCell",
    # quantized operations
    "QuantizedEmbeddingBag",
]


def assert_children(m: torch.nn.Module):
    children = dict(m.named_children())
    if children == {}:
        if (
            hasattr(m, "_get_name")
            and m._get_name() in PYTORCH_QUANTIZED_MODULES
        ):
            if hasattr(m, "get_weight"):
                for weight in m.get_weight().values():
                    assert weight.dtype == torch.qint8
                for bias in m.get_bias().values():
                    assert bias.dtype == torch.float32
            elif hasattr(m, "_packed_params"):
                if "Dynamic" in m._get_name():
                    assert m._packed_params.dtype == torch.qint8
                else:
                    assert m._packed_params.dtype == torch.quint8
            else:
                raise NotImplementedError(
                    "No _all_weight_values or _packed_params"
                )
        else:
            raise NotImplementedError(
                "Quantized modules are not supported yet", m
            )
    else:
        for name, child in children.items():
            if (
                hasattr(child, "_get_name")
                and child._get_name() in PYTORCH_QUANTIZED_MODULES
            ):
                if hasattr(child, "get_weight"):
                    for weight in child.get_weight().values():
                        assert weight.dtype == torch.qint8
                    for bias in child.get_bias().values():
                        assert bias.dtype == torch.float32
                elif hasattr(child, "_packed_params"):
                    if "Dynamic" in child._get_name():
                        assert child._packed_params.dtype == torch.qint8
                    else:
                        assert child._packed_params.dtype == torch.quint8
                else:
                    raise NotImplementedError(
                        "No _all_weight_values or _packed_params"
                    )
            else:
                raise NotImplementedError(
                    "Quantized modules are not supported yet", name, child
                )
    return True


class SingleLayer(torch.nn.Module):
    def __init__(self, type="linear"):
        super(SingleLayer, self).__init__()
        if type == "linear":
            self.l1 = torch.nn.Linear(12, 1)
        elif type == "conv2d":
            self.l1 = torch.nn.Conv2d(1, 1, 3, padding=1)
        elif type == "conv3d":
            self.l1 = torch.nn.Conv3d(1, 1, 3, padding=1)
        elif type == "convtranspose2d":
            self.l1 = torch.nn.ConvTranspose2d(1, 1, 3, padding=1)
        elif type == "convtranspose3d":
            self.l1 = torch.nn.ConvTranspose3d(1, 1, 3, padding=1)
        elif type == "batchnorm":
            self.l1 = torch.nn.BatchNorm2d(1)
        elif type == "instancenorm":
            self.l1 = torch.nn.InstanceNorm2d(1)
        elif type == "layernorm":
            self.l1 = torch.nn.LayerNorm(1)
        elif type == "embedding":
            self.l1 = torch.nn.Embedding(1, 1)
        elif type == "embeddingbag":
            self.l1 = torch.nn.EmbeddingBag(1, 1)
        elif type == "rnn":
            self.l1 = torch.nn.RNNCell(1, 1)
        elif type == "lstm":
            self.l1 = torch.nn.LSTMCell(1, 1)
        elif type == "gru":
            self.l1 = torch.nn.GRUCell(1, 1)
        else:
            raise NotImplementedError(f"type {type} not implemented")

    def forward(self, x):
        x = self.l1(x)
        return x


@pytest.mark.parametrize(
    "type, supported",
    [
        # Linear
        ["linear", True],
        # ConvXd
        ["conv2d", False],
        ["conv3d", False],
        ["convtranspose2d", False],
        ["convtranspose3d", False],
        # Norms
        ["batchnorm", False],
        ["layernorm", False],
        ["instancenorm", False],
        # Embeddings
        ["embedding", False],
        ["embeddingbag", True],
        # RNNs
        ["rnn", True],
        ["lstm", True],
        ["gru", True],
    ],
)
def test_quantize_on_single_layer(type, supported):
    """Tests that the auto_set_backend function works as expected."""
    import approx

    model_fp32 = SingleLayer(type=type)
    approx.auto_set_backend()
    model_int8 = approx.auto_quantize(model_fp32, pretrained=True)
    if supported:
        assert assert_children(model_int8) == True
    else:
        with pytest.raises(NotImplementedError):
            assert_children(model_int8)


class MultiLayers(torch.nn.Module):
    def __init__(self, type="linear"):
        super(MultiLayers, self).__init__()
        self._type = type
        if type == "linear":
            self.l1 = torch.nn.Linear(12, 8)
            self.l2 = torch.nn.Linear(8, 4)
            self.l3 = torch.nn.Linear(4, 2)
            self.l4 = torch.nn.Linear(2, 1)
        elif type == "conv2d":
            self.l1 = torch.nn.Conv2d(256, 128, 3, padding=1)
            self.l2 = torch.nn.Conv2d(128, 64, 3, padding=1)
            self.l3 = torch.nn.Conv2d(64, 32, 3, padding=1)
            self.l4 = torch.nn.Conv2d(32, 16, 3, padding=1)
        elif type == "conv3d":
            self.l1 = torch.nn.Conv3d(256, 128, 3, padding=1)
            self.l2 = torch.nn.Conv3d(128, 64, 3, padding=1)
            self.l3 = torch.nn.Conv3d(64, 32, 3, padding=1)
            self.l4 = torch.nn.Conv3d(32, 16, 3, padding=1)
        elif type == "convtranspose2d":
            self.l1 = torch.nn.ConvTranspose2d(16, 32, 3, padding=1)
            self.l2 = torch.nn.ConvTranspose2d(32, 64, 3, padding=1)
            self.l3 = torch.nn.ConvTranspose2d(64, 128, 3, padding=1)
            self.l4 = torch.nn.ConvTranspose2d(128, 256, 3, padding=1)
        elif type == "convtranspose3d":
            self.l1 = torch.nn.ConvTranspose3d(16, 32, 3, padding=1)
            self.l2 = torch.nn.ConvTranspose3d(32, 64, 3, padding=1)
            self.l3 = torch.nn.ConvTranspose3d(64, 128, 3, padding=1)
            self.l4 = torch.nn.ConvTranspose3d(128, 256, 3, padding=1)
        elif type == "batchnorm":
            self.l1 = torch.nn.Conv2d(256, 128, 3, padding=1)
            self.l2 = torch.nn.BatchNorm2d(1)
            self.l3 = torch.nn.Conv2d(128, 64, 3, padding=1)
            self.l4 = torch.nn.BatchNorm2d(1)
        elif type == "instancenorm":
            self.l1 = torch.nn.Conv2d(256, 128, 3, padding=1)
            self.l2 = torch.nn.InstanceNorm2d(1)
            self.l3 = torch.nn.Conv2d(128, 64, 3, padding=1)
            self.l4 = torch.nn.InstanceNorm2d(1)
        elif type == "layernorm":
            self.l1 = torch.nn.Conv2d(256, 128, 3, padding=1)
            self.l2 = torch.nn.LayerNorm(1)
            self.l3 = torch.nn.Conv2d(128, 64, 3, padding=1)
            self.l4 = torch.nn.LayerNorm(1)
        elif type == "embedding":
            self.l1 = torch.nn.Embedding(1, 1)
            self.l2 = torch.nn.Embedding(1, 1)
            self.l3 = torch.nn.Embedding(1, 1)
            self.l4 = torch.nn.Embedding(1, 1)
        elif type == "embeddingbag":
            self.l1 = torch.nn.EmbeddingBag(1, 1)
            self.l2 = torch.nn.EmbeddingBag(1, 1)
            self.l3 = torch.nn.EmbeddingBag(1, 1)
            self.l4 = torch.nn.EmbeddingBag(1, 1)
        elif type == "rnn":
            self.l1 = torch.nn.RNNCell(256, 128)
            self.l2 = torch.nn.RNNCell(128, 64)
            self.l3 = torch.nn.RNNCell(64, 32)
            self.l4 = torch.nn.RNNCell(32, 16)
        elif type == "lstm":
            self.l1 = torch.nn.LSTMCell(256, 128)
            self.l2 = torch.nn.LSTMCell(128, 64)
            self.l3 = torch.nn.LSTMCell(64, 32)
            self.l4 = torch.nn.LSTMCell(32, 16)
        elif type == "gru":
            self.l1 = torch.nn.GRUCell(256, 128)
            self.l2 = torch.nn.GRUCell(128, 64)
            self.l3 = torch.nn.GRUCell(64, 32)
            self.l4 = torch.nn.GRUCell(32, 16)
        else:
            raise NotImplementedError(f"type {type} not implemented")

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        return x


@pytest.mark.parametrize(
    "type, supported",
    [
        # Linear
        ["linear", True],
        # ConvXd
        ["conv2d", False],
        ["conv3d", False],
        ["convtranspose2d", False],
        ["convtranspose3d", False],
        # Norms
        ["batchnorm", False],
        ["layernorm", False],
        ["instancenorm", False],
        # Embeddings
        ["embedding", False],
        ["embeddingbag", True],
        # RNNs
        ["rnn", True],
        ["lstm", True],
        ["gru", True],
    ],
)
def test_quantize_on_multiple_layer(type, supported):
    """Tests that the auto_set_backend function works as expected."""
    import approx

    model_fp32 = MultiLayers(type=type)
    approx.auto_set_backend()
    model_int8 = approx.auto_quantize(model_fp32, pretrained=True)
    if supported:
        assert assert_children(model_int8) == True
    else:
        with pytest.raises(NotImplementedError):
            assert_children(model_int8)
