import os
import numpy as np
import torch
import torch.nn as nn
from catalyst.dl import registry


def patch_model_with_embedding(embedding_file: str, params: dict) -> nn.Module:
    embedding = np.load(embedding_file, allow_pickle=True)

    model = registry.MODELS.get_from_params(**params)
    model.embedding.weight = nn.Parameter(torch.tensor(embedding, dtype=torch.float))
    model.embedding.requires_grad = False
    
    return model


def model_from_checkpoint(checkpoint: str, params: dict) -> nn.Module:
    # if checkpoint name is uppercase word -> load checkpoint from this env variable
    if checkpoint.isupper():
        checkpoint = os.environ[checkpoint]

    model = registry.MODELS.get_from_params(**params)

    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model


def unfreezed_transf(checkpoint: str, params: dict) -> nn.Module:
    model = model_from_checkpoint(checkpoint, params)
    require_grads(model.bert)
    return model


def do_not_require_grads(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def require_grads(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


class EmbeddingDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2).permute(0, 3, 2, 1)
        x = super(EmbeddingDropout, self).forward(x)
        x = x.permute(0, 3, 2, 1).squeeze(2)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class LstmPool(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(LstmPool, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=output_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(2)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x_max = self.max_pool(x)[:, -1]
        x_avg = self.avg_pool(x)[:, -1]
        x = torch.cat([x_max, x_avg], 1)
        return x


class LstmGruPool(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int,
                 output_size: int):
        super(LstmGruPool, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.gru = nn.GRU(
            input_size=2 * hidden_size,
            hidden_size=output_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(2)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        x_max = self.max_pool(x)[:, -1]
        x_avg = self.avg_pool(x)[:, -1]
        x = torch.cat([x_max, x_avg], 1)

        return x


class LSTM_GRU_Attention_Pool(nn.Module):
    def __init__(self,
                 input_size: int, 
                 hidden_size: int,
                 output_size: int):
        super(LSTM_GRU_Attention_Pool, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_attn = BiaslessAttention(
            feature_dim=2 * hidden_size
        )
        self.gru = nn.GRU(
            input_size=2 * hidden_size,
            hidden_size=output_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_attn = BiaslessAttention(
            feature_dim=2 * output_size
        )

    def forward(self, x, mask=None):
        
        self.lstm.flatten_parameters()
        x_lstm, _ = self.lstm(x)
        x_lstm_attn = self.lstm_attn(x_lstm, mask)

        self.gru.flatten_parameters()
        x_gru, _ = self.gru(x_lstm)
        x_gru_attn = self.gru_attn(x_gru, mask)

        x = torch.cat([
            torch.max(x_lstm, 1)[0], torch.max(x_gru, 1)[0], x_lstm_attn,
            torch.mean(x_lstm, 1), torch.mean(x_gru, 1), x_gru_attn,
        ], 1)

        return x


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class BiaslessAttention(nn.Module):
    def __init__(self, feature_dim, **kwargs):
        super(BiaslessAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.feature_dim = feature_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = x.size(1)

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            # mask should have shapes B*S
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
