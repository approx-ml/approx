import enum
import statistics
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table
from typing_extensions import Protocol


class Metric(enum.Enum):
    LOSS = 0
    ACCURACY = 1


class EvalLoop(Protocol):
    def __call__(self, model: Any, test_dl: Any) -> Dict[Metric, List[float]]:
        ...


class _EvalHistory:
    def __init__(self, model: Any, histories: Dict[Metric, List[float]]):
        try:
            str(model)
        except Exception:
            raise ValueError(
                "Currently does not support models which do not have `__str__`"
            )
        self.model = model
        self._histories = histories

    @property
    def metrics(self) -> List[Metric]:
        """Returns the metrics which have been recorded"""
        return list(self._histories.keys())

    def mean(self, metric: Metric) -> float:
        r"""
        Computes the mean (also known as the average)

        $$
        \frac{1}{N} \sum_{i=1}^N x_i
        $$
        where N is the number of batches

        Args:
            metric: The metric to compute the mean for

        Returns:
            The mean of that metric's history
        """
        # implementation note: this could be cached in case the history is very large
        # and computing may take significant time
        return statistics.mean(self[metric])

    def std(self, metric: Metric) -> float:
        r"""
        Computes the standard deviation of the metric history

        $$
        \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}
        $$
        where N is the number of batches

        Args:
            metric: The metric to compute the standard deviation for

        Returns:
            The standard deviation of the metric history
        """
        return statistics.pstdev(self[metric])

    def median(self, metric: Metric) -> float:
        r"""
        Computes the median of the metric history

        $$
        \frac{1}{2} \left[ \frac{1}{N} \sum_{i=1}^N x_i + \frac{1}{N} \sum_{i=1}^N x_i \right]
        $$
        where N is the number of batches

        Args:
            metric: The metric to compute the median for

        Returns:
            The median of the metric history
        """
        return statistics.median(self[metric])

    def __getitem__(self, metric: Metric) -> List[float]:
        """
        Retrieves the metric history for some metric. Useful if you need
        access to the actual history itself

        Args:
            metric: The metric to return the history for

        Returns:
            The history of the given metric
        """
        return self._histories.get(metric, [])


class CompareResult:
    """
    A result that is returned after calling `approx.compare`
    """

    def __init__(self, results: List[_EvalHistory]):
        self._results = results

    def mean(self, metric: Metric) -> Dict[str, float]:
        r"""
        Computes the mean for some particular metric

        $$
        \frac{1}{N} \sum_{i=1}^N x_i
        $$

        where N is the number of batches

        Args:
            metric: The metric to compute the mean for

        Returns:
            A dictionary that maps your model to the mean of that metric's history
        """
        return {
            str(result.model): result.mean(metric) for result in self._results
        }

    def std(self, metric: Metric) -> Dict[str, float]:
        r"""
        Compares the standard deviations for some particular metric

        $$
        \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2}
        $$

        where N is the number of batches

        Args:
            metric: The metric to compute the standard deviation for

        Returns:
            A dictionary that maps your model to the standard deviation of that metric's history
        """
        return {
            str(result.model): result.std(metric) for result in self._results
        }

    def __str__(self) -> str:
        # crude table printer, should be replaced with a more fancy one
        console = Console()
        for metric in self._results[0].metrics:
            console.rule(f"[bold red]{metric.name.lower()}")
            table = Table(
                show_header=True, header_style="bold magenta", show_lines=True
            )
            table.add_column("Model", style="bold green")
            table.add_column("Mean", justify="center")
            table.add_column("Std", justify="center")
            for result in self._results:
                table.add_row(
                    f"{result.model}",
                    f"{result.mean(metric):.4f}",
                    f"{result.std(metric):.4f}",
                )
            console.print(table)
        return ""


class _CompareRunner:
    """
    Runs the compare operation
    """

    def __init__(
        self,
        models: List[Any],
        test_loader: Any,
        eval_loop: EvalLoop,
    ):
        # todo: auto generate eval loop for certain backends
        self._models = models
        self._test_loader = test_loader
        self._eval_loop = eval_loop
        self._results: List[_EvalHistory] = []

    def run(self) -> CompareResult:
        # todo: investigate multiprocessing
        for model in self._models:
            metric_hist = self._eval_loop(model, self._test_loader)
            self._results.append(_EvalHistory(model, metric_hist))
        return CompareResult(self._results)
