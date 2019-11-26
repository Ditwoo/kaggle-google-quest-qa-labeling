import torch
from catalyst.dl.core import Callback, CallbackOrder, RunnerState


def spearman(input, target, eps: float = 1e-7) -> float:
    n = input.size(0)
    input_mean = input.mean(dim=0, keepdim=True)
    target_mean = target.mean(dim=0, keepdim=True)
    input_std = input.std(dim=0)
    target_std = target.std(dim=0)
    # deal with the degenerate case when some variances are zero
    # this happens when some input or target columns are constant
    input_std = torch.clamp(input_std, eps)
    target_std = torch.clamp(target_std, eps)
    cov = ((input - input_mean) * (target - target_mean)).mean(dim=0)
    cov = cov * (n / (n - 1))   # Bessel's correction
    corr = cov / (input_std * target_std)
    return corr.mean()


class SpearmanScoreCallback(Callback):
    def __init__(self, 
                 prefix: str = "spearman",
                 input_key: str = "targets",
                 output_key: str = "logits",
                 eps: float = 1e-7,
                 **metric_params):
        super().__init__(CallbackOrder.Metric)

        self.prefix: str = prefix
        self.input_key: str = input_key
        self.output_key: str = output_key
        self.eps = eps
        self.metric_params = metric_params

    def on_batch_end(self, state: RunnerState) -> None:
        outputs = torch.sigmoid(state.output[self.output_key])
        targets = state.input[self.input_key]
        score = spearman(outputs, targets, self.eps)
        state.metrics.add_batch_value(
            metrics_dict={f"{self.prefix}": score}
        )

__all__ = ("SpearmanScoreCallback", )
