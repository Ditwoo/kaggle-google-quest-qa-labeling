import torch
import torch.nn as nn


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
        self.gru = nn.GRU(
            input_size=2 * hidden_size,
            hidden_size=output_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.attn = BiaslessAttention(
            feature_dim=2 * output_size
        )

    def forward(self, x):
        self.lstm.flatten_parameters()
        x_lstm, _ = self.lstm(x)

        self.gru.flatten_parameters()
        x_gru, _ = self.gru(x_lstm)

        x_attn = self.attn(x_gru)

        x = torch.cat([
            torch.max(x_lstm, 1)[0], torch.max(x_gru, 1)[0], x_attn,
            torch.mean(x_lstm, 1), torch.mean(x_gru, 1), x_attn,
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
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)
