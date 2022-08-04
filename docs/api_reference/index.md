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

def eval_loop(model, data):
    return {
        "loss": [1.0, 2.0],
        "accuracy": [3.0, 4.0],
    }
```

::: approx.core.compare.EvalLoop
    options:
        heading_level: 4
