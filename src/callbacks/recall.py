import numpy as np
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from numba import njit, prange


@njit(parallel=True)
def fast_recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    size = y_true.shape[0]
    tp = 0
    fn = 0
    for i in prange(size):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
    recall = 0.0
    if tp + fn != 0:
        recall = tp / (tp + fn)
    return recall


@njit(parallel=True)
def fast_recall_by_row(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    num_rows = y_true.shape[0]
    num_cols: int = y_true.shape[1]
    score: float = 0.0
    for r in prange(num_rows):
        tp: int = 0
        fn: int = 0
        for i in prange(num_cols):
            if y_pred[r][i] == 1 and y_true[r][i] == 1:
                tp += 1
            if y_pred[r][i] == 0 and y_true[r][i] == 1:
                fn += 1
        recall = 0.0
        if tp + fn != 0.0:
            recall = tp / (tp + fn)
        score = score + recall / num_rows
    return score


class RecallCallback(Callback):
    def __init__(
            self,
            prefix: str = "recall",
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

        score: float = fast_recall(o, t)
        return score

    def _elements(self, o, t):
        o = (o >= self.threshold).astype(np.int8)
        t = t.astype(np.int8)

        score: float = fast_recall_by_row(o, t)
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


__all__ = ("RecallCallback",)
