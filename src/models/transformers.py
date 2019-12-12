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
        do_not_require_grads(self.bert)  # freeze bert parameters

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_classes)
        )

    def forward(self, sequences, head_mask=None):
        mask = (sequences > 0).float()
        bert_output = self.bert(
            input_ids=sequences,
            attention_mask=mask,
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