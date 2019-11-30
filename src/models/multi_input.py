import torch
import torch.nn as nn
from collections import OrderedDict
from .utils import EmbeddingDropout, LstmPool, LstmGruPool, LSTM_GRU_Attention_Pool


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


class MultiInputLstmGru(nn.Module):
    def __init__(self, 
        embedding_size: int,
        embedding_dim: int,
        category_embedding_size: int,
        category_embedding_dim: int,
        host_embedding_size: int,
        host_embedding_dim: int,
        num_classes: int = 1,
        hidden_size: int = 64,
        out_rnn_size: int = 64,
        dropout_rate: float = 0.5
    ):
        super(MultiInputLstmGru, self).__init__()

        self.embedding = nn.Embedding(embedding_size, embedding_dim)
        self.embedding.requires_grad = False
        self.embedding_dropout = EmbeddingDropout(dropout_rate)
        self.category_embedding = nn.Embedding(category_embedding_size, category_embedding_dim)
        self.host_embedding = nn.Embedding(host_embedding_size, host_embedding_dim)

        self.title_lstm = LstmGruPool(embedding_dim, hidden_size, out_rnn_size)
        self.question_lstm = LstmGruPool(embedding_dim, hidden_size, out_rnn_size)
        self.answer_lstm = LstmGruPool(embedding_dim, hidden_size, out_rnn_size)
        
        input_size = 3 * 2 * out_rnn_size + category_embedding_dim + host_embedding_dim
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

    def forward(self, question_title, question_body, answer, category, host):
        
        title = self.embedding(question_title)
        title = self.embedding_dropout(title)

        title = self.title_lstm(title)

        question = self.embedding(question_body)
        question = self.embedding_dropout(question)

        question = self.question_lstm(question)

        answer = self.embedding(answer)
        answer = self.embedding_dropout(answer)

        answer = self.answer_lstm(answer)

        category = self.category_embedding(category).squeeze(1)
        host = self.host_embedding(host).squeeze(1)

        x = torch.cat([title, question, answer, category, host], 1)
        x = self.head(x)

        return x


class MultiInputLstmGruAttention(nn.Module):
    def __init__(self, 
        embedding_size: int,
        embedding_dim: int,
        category_embedding_size: int,
        category_embedding_dim: int,
        host_embedding_size: int,
        host_embedding_dim: int,
        num_classes: int = 1,
        hidden_size: int = 64,
        out_rnn_size: int = 64,
        dropout_rate: float = 0.5
    ):
        super(MultiInputLstmGruAttention, self).__init__()

        self.embedding = nn.Embedding(embedding_size, embedding_dim)
        self.embedding_dropout = EmbeddingDropout(dropout_rate)
        self.category_embedding = nn.Embedding(category_embedding_size, category_embedding_dim)
        self.host_embedding = nn.Embedding(host_embedding_size, host_embedding_dim)

        self.title_lstm = LSTM_GRU_Attention_Pool(embedding_dim, hidden_size, out_rnn_size)
        self.question_lstm = LSTM_GRU_Attention_Pool(embedding_dim, hidden_size, out_rnn_size)
        self.answer_lstm = LSTM_GRU_Attention_Pool(embedding_dim, hidden_size, out_rnn_size)
        
        input_size = 3*(2*(2*hidden_size + 2*out_rnn_size + 2*out_rnn_size)) + category_embedding_dim + host_embedding_dim
        self.head = nn.Sequential(
            OrderedDict(
                [
                    (
                        "block1",
                        nn.Sequential(
                            nn.Linear(input_size, num_classes * 10),
                            nn.ReLU(True),
                            nn.BatchNorm1d(num_classes * 10),
                            nn.Dropout(dropout_rate),
                        ),
                    ),
                    (
                        "block2",
                        nn.Sequential(
                            nn.Linear(num_classes * 10, int(num_classes * 5)),
                            nn.ReLU(True),
                            nn.BatchNorm1d(int(num_classes * 5)),
                            nn.Dropout(dropout_rate),
                        ),
                    ),
                    ("head", nn.Linear(int(num_classes * 5), num_classes)),
                ]
            )
        )

    def forward(self, question_title, question_body, answer, category, host):
        
        title = self.embedding(question_title)
        title = self.embedding_dropout(title)

        title = self.title_lstm(title)

        question = self.embedding(question_body)
        question = self.embedding_dropout(question)

        question = self.question_lstm(question)

        answer = self.embedding(answer)
        answer = self.embedding_dropout(answer)

        answer = self.answer_lstm(answer)

        category = self.category_embedding(category).squeeze(1)
        host = self.host_embedding(host).squeeze(1)

        x = torch.cat([title, question, answer, category, host], 1)

        # import pdb; pdb.set_trace()

        x = self.head(x)

        return x
