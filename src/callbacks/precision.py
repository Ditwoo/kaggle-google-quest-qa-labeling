import numpy as np
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from numba import njit, prange


@njit(parallel=True)
def fast_precision(y_pred: np.ndarray,
                   y_true: np.ndarray) -> float:
    size = y_true.shape[0]
    tp = 0
    fp = 0
    for i in prange(size):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
    precision = 1.0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    return precision


@njit(parallel=True)
def fast_precision_by_row(y_pred: np.ndarray,
                          y_true: np.ndarray) -> float:
    num_rows = y_true.shape[0]
    num_cols = y_true.shape[1]
    score = 0.0
    for r in prange(num_rows):
        tp = 0
        fp = 0
        for i in prange(num_cols):
            if y_pred[r][i] == 1 and y_true[r][i] == 1:
                tp += 1
            if y_pred[r][i] == 1 and y_true[r][i] == 0:
                fp += 1
        precision = 1.0
        if tp + fp != 0:
            precision = tp / (tp + fp)
        score = score + precision / num_rows
    return score


class PrecisionCallback(Callback):
    def __init__(
            self,
            prefix: str = "precision",
            input_key: str = "targets",
            output_key: str = "logits",
            threshold: float = 0.5,
            how: str = "all",
            **metric_params,
    ):
        super().__init__(CallbackOrder.Metric)
        self.prefix: str = prefix
        self.input_key: str = input_key
        self.output_key: str = output_key
        self.threshold: float = threshold
        self.how = how
        self.metric_params = metric_params

    def _all(self, o, t):
        o = (o.flatten() >= self.threshold).astype(np.int8)
        t = t.flatten().astype(np.int8)

        score: float = fast_precision(o, t)
        return score

    def _elements(self, o, t):
        o = (o >= self.threshold).astype(np.int8)
        t = t.astype(np.int8)

        score: float = fast_precision_by_row(o, t)
        return score

    def on_batch_end(self, state: RunnerState) -> None:
        outputs = state.output[self.output_key].sigmoid()
        outputs = outputs.detach().cpu().numpy()

        targets = state.input[self.input_key]
        targets = targets.detach().cpu().numpy()

        if self.how == "elementwise":
            score = self._elements(outputs, targets)
        else:
            score = self._all(outputs, targets)

        state.metrics.add_batch_value(metrics_dict={f"{self.prefix}": score})


__all__ = ("PrecisionCallback",)
