import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .utils import do_not_require_grads


class TransfModel(nn.Module):
    def __init__(self,
                 pretrain_dir: str,
                 num_classes: int = 1):
        super(TransfModel, self).__init__()

        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )
        # do_not_require_grads(self.bert)  # freeze bert parameters

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_classes)
        )

    def forward(self, sequences, segments=None, head_mask=None):
        mask = (sequences > 0).float()
        bert_output = self.bert(
            input_ids=sequences,
            attention_mask=mask,
            token_type_ids=segments,
            head_mask=head_mask
        )
        # we only need the hidden state here and don't need
        # transformer output, so index 0
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        # we take embeddings from the [CLS] token, so again index 0
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits


class PooledTransfModel(nn.Module):
    def __init__(self,
                 pretrain_dir: str,
                 num_classes: int = 1):
        super(PooledTransfModel, self).__init__()

        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )
        # do_not_require_grads(self.bert)  # freeze bert parameters

        self.pre_classifier = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            # nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, num_classes)
        )

    def forward(self, sequences, segments=None, head_mask=None):
        mask = (sequences > 0).float()
        bert_output = self.bert(
            input_ids=sequences,
            attention_mask=mask,
            token_type_ids=segments,
            head_mask=head_mask
        )
        # we only need the hidden state here and don't need
        # transformer output, so index 0
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        # we take embeddings from the [CLS] token, so again index 0
        pooled_output = torch.cat([
            torch.max(hidden_state, 1)[0],
            torch.mean(hidden_state, 1),
        ], 1)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits


class PooledLstmTransfModel(nn.Module):
    def __init__(self,
                 pretrain_dir: str,
                 num_classes: int = 1):
        super(PooledLstmTransfModel, self).__init__()

        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )
        
        self.rnns = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=config.hidden_dropout_prob,
        )

        self.pre_classifier = nn.Linear(config.hidden_size * 6, config.hidden_size)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_classes)
        )

    def forward(self, sequences, segments=None, head_mask=None):
        mask = (sequences > 0).float()
        bert_output = self.bert(
            input_ids=sequences,
            attention_mask=mask,
            token_type_ids=segments,
            head_mask=head_mask
        )
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        rnn_hidden_states, _ = self.rnns(hidden_state)  # (bs, seq_len, dim)
        pooled_output = torch.cat([
            torch.max(hidden_state, 1)[0],
            torch.mean(hidden_state, 1),
            torch.max(rnn_hidden_states, 1)[0],
            torch.mean(rnn_hidden_states, 1),
        ], 1)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits


class MultipleInputTransfModel(nn.Module):
    def __init__(self,
                 pretrain_dir: str,
                 num_classes: int = 1):
        super(MultipleInputTransfModel, self).__init__()

        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )
        do_not_require_grads(self.bert)  # freeze bert parameters

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_classes)
        )
    
    def forward(self, question_title, question_body, answer, head_mask=None):
        title = self.bert(
            input_ids=question_title,
            attention_mask=(question_title > 0).float(),
            head_mask=head_mask
        )[0][:, 0]

        body = self.bert(
            input_ids=question_body,
            attention_mask=(question_body > 0).float(),
            head_mask=head_mask
        )[0][:, 0]

        ans = self.bert(
            input_ids=question_title,
            attention_mask=(answer > 0).float(),
            head_mask=head_mask
        )[0][:, 0]

        concated = torch.cat([title, body, ans], 1)
        logits = self.classifier(concated)

        return logits