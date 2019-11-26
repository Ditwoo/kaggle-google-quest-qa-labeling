import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import EmbeddingDropout


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


class MultiInputLstm(nn.Module):
    def __init__(self, 
        embedding_size: int,
        embedding_dim: int,
        num_classes: int = 1,
        hidden_lstm_size: int = 64,
        dropout_rate: float = 0.5
    ):
        super(MultiInputLstm, self).__init__()

        self.embedding = nn.Embedding(embedding_size, embedding_dim)
        self.embedding_dropout = EmbeddingDropout(dropout_rate)

        self.title_lstm = LstmPool(embedding_dim, hidden_lstm_size)
        self.question_lstm = LstmPool(embedding_dim, hidden_lstm_size)
        self.answer_lstm = LstmPool(embedding_dim, hidden_lstm_size)
        input_size = 3 * 2 * hidden_lstm_size  # 2 pooling layers
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

    def forward(self, question_title, question_body, answer):
        # import pdb; pdb.set_trace()

        title = self.embedding(question_title)
        title = self.embedding_dropout(title)

        title = self.title_lstm(title)

        question = self.embedding(question_body)
        question = self.embedding_dropout(question)

        question = self.question_lstm(question)

        answer = self.embedding(answer)
        answer = self.embedding_dropout(answer)

        answer = self.answer_lstm(answer)

        x = torch.cat([title, question, answer], 1)
        x = self.head(x)

        return x
