from typing import Tuple

import numpy as np
from catalyst.dl.core import Callback, CallbackOrder, RunnerState
from numba import njit, prange

from .precision import fast_precision_by_row
from .recall import fast_recall_by_row


@njit(parallel=True)
def fast_f_score_parts(y_pred: np.ndarray, y_true: np.ndarray) -> Tuple[int, int, int, int]:
    size = y_true.shape[0]
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in prange(size):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        if y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        if y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
        if y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
    return tp, tn, fp, fn


class FBetaCallback(Callback):
    def __init__(self,
                 prefix: str = "fbeta",
                 input_key: str = "targets",
                 output_key: str = "logits",
                 threshold: float = 0.5,
                 beta: float = 1.0,
                 how: str = "all",
                 **metric_params):
        """
        F_beta score callback.

        * To give more weight to Precision beta should be - `0 < beta < 1`
        * To give more weight to Recall, beta should be - `1 < beta < +inf`

        If `beta == 1` then will be computed simple F1 score.
        """
        super().__init__(CallbackOrder.Metric)

        self.prefix: str = prefix
        self.input_key: str = input_key
        self.output_key: str = output_key
        self.threshold: float = threshold
        self.b = beta
        self.b_2: float = beta ** 2
        self.metric_params = metric_params
        self.how = how

        if self.how == "elementwise":
            self.compute_foo = self._elements
        else:
            self.compute_foo = self._all

    def f_measure(self, p, r):
        try:
            return (1 + self.b_2) * p * r / (self.b_2 * p + r)
        except ZeroDivisionError:
            return 0.0

    def _all(self, o, t):
        o = (o.flatten() >= self.threshold).astype(np.int8)
        t = t.flatten().astype(np.int8)

        tp, _, fp, fn = fast_f_score_parts(o, t)

        precision = 1.0
        if tp + fp != 0:
            precision = tp / (tp + fp)

        recall = 0.0
        if tp + fn != 0:
            recall = tp / (tp + fn)

        return self.f_measure(precision, recall)

    def _elements(self, o, t):
        o = (o >= self.threshold).astype(np.int8)
        t = t.astype(np.int8)

        precision = fast_precision_by_row(o, t)
        recall = fast_recall_by_row(o, t)

        return self.f_measure(precision, recall)

    def on_batch_end(self, state: RunnerState) -> None:
        outputs = state.output[self.output_key].sigmoid()
        outputs = outputs.detach().cpu().numpy()

        targets = state.input[self.input_key]
        targets = targets.detach().cpu().numpy()

        f_beta = self.compute_foo(outputs, targets)
        state.metrics.add_batch_value(metrics_dict={f"{self.prefix}": f_beta})


class F1Callback(FBetaCallback):
    def __init__(self,
                 prefix: str = "f1",
                 input_key: str = "targets",
                 output_key: str = "logits",
                 threshold: float = 0.5,
                 how: str = "all",
                 **metric_params):
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            threshold=threshold,
            beta=1.0,
            how=how,
            **metric_params,
        )


__all__ = ("F1Callback", "FBetaCallback")
