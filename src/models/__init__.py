import numpy as np
import torch
import torch.nn as nn
from catalyst.dl import registry

from .vector import LinearModel
from .lstm_gru import LSTM_GRU


def patch_model_with_embedding(embedding_file: str, params: dict) -> torch.nn.Module:
    embedding = np.load(embedding_file, allow_pickle=True)

    model = registry.MODELS.get_from_params(**params)
    model.embedding.weight = nn.Parameter(torch.tensor(embedding, dtype=torch.float))
    model.embedding.requires_grad = False
    
    return model


def model_from_checkpoint(checkpoint: str, params: dict) -> torch.nn.Module:
    model = registry.MODELS.get_from_params(**params)

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
