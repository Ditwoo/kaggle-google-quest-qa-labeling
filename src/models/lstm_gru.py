import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import EmbeddingDropout


class LSTM_GRU(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        embedding_dim: int,
        num_classes: int = 1,
        hidden_lstm_size: int = 64,
        hidden_gru_size: int = 64,
        dropout_rate: float = 0.5):

        super(LSTM_GRU, self).__init__()
        
        self.embedding = nn.Embedding(embedding_size, embedding_dim)
        self.embedding_dropout = EmbeddingDropout(dropout_rate)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_lstm_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.gru = nn.GRU(
            input_size=hidden_lstm_size * 2,
            hidden_size=hidden_gru_size,
            bias=True,
            batch_first=True,
            bidirectional=True,
        )
        self.max_pool = nn.MaxPool1d(2)
        self.avg_pool = nn.AvgPool1d(2)
        input_size = 2 * hidden_gru_size  # 2 pooling layers
        self.head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            nn.Linear(input_size, num_classes * 2),
                            nn.ReLU(True),
                            nn.BatchNorm1d(num_classes * 2),
                            nn.Dropout(dropout_rate),
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            nn.Linear(num_classes * 2, int(num_classes * 1.5)),
                            nn.ReLU(True),
                            nn.BatchNorm1d(int(num_classes * 1.5)),
                            nn.Dropout(dropout_rate),
                        ),
                    ),
                    ("head", nn.Linear(int(num_classes * 1.5), num_classes)),
                ]
            )
        )

    def forward(self, input_sequence):
        x = self.embedding(input_sequence)  # batch * seq -> batch * seq * emb_size
        x = self.embedding_dropout(x)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        self.gru.flatten_parameters()
        x, _ = self.gru(x)

        x_max = self.max_pool(x)[:, -1]
        x_avg = self.avg_pool(x)[:, -1]
        x = torch.cat([x_max, x_avg], 1)
        x = self.head(x)
        return x


__all__ = ("LSTM_GRU",)
