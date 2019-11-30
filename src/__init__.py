from catalyst.dl import registry
from .runner import Runner

# from .optimizers import RAdam
from .models import (
    LinearModel,
    LSTM_GRU,
    MultiInputLstm,
    MultiInputLstmGru,
    MultiInputLstmGruAttention,
    patch_model_with_embedding,
    model_from_checkpoint,
)
from .callbacks import (
    PrecisionCallback,
    RecallCallback,
    F1Callback,
    FBetaCallback,
    SpearmanScoreCallback,
)
from .experiment import Experiment


registry.Model(LinearModel)
registry.Model(LSTM_GRU)
registry.Model(MultiInputLstm)
registry.Model(MultiInputLstmGru)
registry.Model(MultiInputLstmGruAttention)
registry.Model(patch_model_with_embedding)
registry.Model(model_from_checkpoint)

registry.Callback(PrecisionCallback)
registry.Callback(RecallCallback)
registry.Callback(F1Callback)
registry.Callback(FBetaCallback)
registry.Callback(SpearmanScoreCallback)