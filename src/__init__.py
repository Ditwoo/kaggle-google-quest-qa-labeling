from catalyst.dl import registry
from .runner import Runner

# from .optimizers import RAdam
from .models import (
    LinearModel,
    LSTM_GRU,
    MultiInputLstm,
    MultiInputLstmGru,
    MultiInputLstmGruAttention,
    TransfModel,
    PooledTransfModel,
    PooledLstmTransfModel,
    PooledTransfModelWithCatericalFeatures,
    PTCFS,
    PTM,
    PTC,
    TwoSidedPooledTransformer,
    PCTCFS,
    patch_model_with_embedding,
    model_from_checkpoint,
    unfreezed_transf,
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
registry.Model(TransfModel)
registry.Model(PooledTransfModel)
registry.Model(PooledLstmTransfModel)
registry.Model(PooledTransfModelWithCatericalFeatures)
registry.Model(PTCFS)
registry.Model(PTM)
registry.Model(PTC)
registry.Model(TwoSidedPooledTransformer)
registry.Model(PCTCFS)
# functions
registry.Model(patch_model_with_embedding)
registry.Model(model_from_checkpoint)
registry.Model(unfreezed_transf)

registry.Callback(PrecisionCallback)
registry.Callback(RecallCallback)
registry.Callback(F1Callback)
registry.Callback(FBetaCallback)
registry.Callback(SpearmanScoreCallback)