# API reference

## Main API
::: approx
    options:
        heading_level: 3

## Types


### `EvalLoop`

A function which accepts your model and the data,
and returns a dictionary mapping metrics to their histories.

For example
```python
from approx import CompareMetric as Metric

def eval_loop(model, data):
    return {
        Metric.LOSS: [1.0, 2.0],
        Metric.ACCURACY: [3.0, 4.0],
    }
```

::: approx.core.compare.EvalLoop
    options:
        heading_level: 4


### `CompareMetric`
Either `LOSS` or `ACCURACY`

::: approx.CompareMetric
    options:
        heading_level: 4
