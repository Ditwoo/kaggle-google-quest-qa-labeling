import numpy as np
import torch
from scipy.stats import spearmanr
from catalyst.dl.core import Callback, CallbackOrder, RunnerState


def spearman(input, target, classes: list) -> float:
    score = 0.0
    for c in classes:
        score += np.nan_to_num(spearmanr(target[:, c], input[:, c]).correlation) / len(classes)
    return score


class SpearmanScoreCallback(Callback):
    def __init__(self, 
                 prefix: str = "spearman",
                 input_key: str = "targets",
                 output_key: str = "logits",
                 classes: list = [0],
                 eps: float = 1e-7,
                 **metric_params):
        super().__init__(CallbackOrder.Metric)

        self.prefix: str = prefix
        self.input_key: str = input_key
        self.output_key: str = output_key
        self.eps = eps
        self.metric_params = metric_params
        self.classes = classes
        self._reset()

    def _reset(self):
        self.targets = None
        self.outputs = None

    def on_batch_end(self, state: RunnerState) -> None:
        # store batch predictions and targets
        outputs = torch.sigmoid(state.output[self.output_key])
        outputs = outputs.detach().cpu().numpy()
        if self.outputs is None:
            self.outputs = []
        self.outputs.append(outputs)

        targets = state.input[self.input_key]
        targets = targets.detach().cpu().numpy()
        if self.targets is None:
            self.targets = []
        self.targets.append(targets)

    def on_loader_end(self, state: RunnerState) -> None:
        # compute score based on data from whole loader
        spearman_score = spearman(
            np.vstack(self.outputs), 
            np.vstack(self.targets),
            self.classes
        )
        state.metrics.epoch_values[state.loader_name][f"{self.prefix}"] = spearman_score
        self._reset()

__all__ = ("SpearmanScoreCallback", )
