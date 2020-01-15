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
        """
        Inputs:
            sequences - torch.LongTensor with tokens
            segments  - torch.LongTensor with segment indicators
        """
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
        """
        Inputs:
            sequences - torch.LongTensor with tokens
            segments  - torch.LongTensor with segment indicators
        """
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
        """
        Inputs:
            sequences - torch.LongTensor with tokens
            segments  - torch.LongTensor with segment indicators
        """
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


class PooledTransfModelWithCatericalFeatures(nn.Module):
    def __init__(self,
                 pretrain_dir: str,
                 num_categories: int,
                 num_hosts: int,
                 num_classes: int = 1):
        super(PooledTransfModelWithCatericalFeatures, self).__init__()

        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )

        self.bert = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )
        
        categories_emb_dim = 8
        self.categories_emb = nn.Embedding(num_categories, categories_emb_dim)

        hosts_emb_dim = 16
        self.hosts_emb = nn.Embedding(num_hosts, hosts_emb_dim)

        self.pre_classifier = nn.Linear(
            config.hidden_size * 2 + hosts_emb_dim + categories_emb_dim, 
            config.hidden_size * 4
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            # nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, num_classes)
        )

    def forward(self, sequences, category, host, segments=None, head_mask=None):
        """
        Inputs:
            sequences - torch.LongTensor with tokens
            category  - torch.LongTensor with 1 category for each element of batch
            host      - torch.LongTensor with 1 host for each element of batch
            segments  - torch.LongTensor with segment indicators
        """
        mask = (sequences > 0).float()
        bert_output = self.bert(
            input_ids=sequences,
            attention_mask=mask,
            token_type_ids=segments,
            head_mask=head_mask
        )
        ctg_emb = self.categories_emb(category).squeeze(1)
        host_emb = self.hosts_emb(host).squeeze(1)
        hidden_state = bert_output[0]  # (bs, seq_len, dim)
        pooled_output = torch.cat([
            torch.max(hidden_state, 1)[0],
            torch.mean(hidden_state, 1),
            ctg_emb,
            host_emb
        ], 1)  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits


class PTM(nn.Module):
    """
    Pooled Transformer model.
    """
    def __init__(self,
                 pretrain_dir: str,
                 num_classes: int = 1,
                 pad_token: int = 0):
        super(PTM, self).__init__()
        self.pad_token = pad_token
        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )
        if hasattr(config, "hidden_dropout_prob"):
            dropout = config.hidden_dropout_prob
        elif hasattr(config, "dropout"):
            dropout = config.dropout
        else:
            dropout = 0.1

        self.base_model = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )

        self.pre_classifier = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, num_classes)
        )
    
    def forward(self, sequences, segments=None, head_mask=None):
        """
        Inputs:
            sequences - torch.LongTensor with tokens
            segments  - torch.LongTensor with segment indicators
        """

        # import pdb; pdb.set_trace()

        mask = (sequences != self.pad_token).float()
        bm_output = self.base_model(
            input_ids=sequences,
            attention_mask=mask,
            token_type_ids=segments,
            # head_mask=head_mask
        )
        hidden_state = bm_output[0]  # (bs, seq_len, dim)

        pooled_output = torch.cat([
            torch.max(hidden_state, 1)[0],
            torch.mean(hidden_state, 1),
        ], 1) # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits


class PTC(nn.Module):
    """
    Pooled Transformer model with text statistics.
    """
    def __init__(self,
                 pretrain_dir: str,
                 stats_dim: int,
                 num_classes: int = 1,
                 pad_token: int = 0):
        super(PTC, self).__init__()
        self.pad_token = pad_token
        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )
        if hasattr(config, "hidden_dropout_prob"):
            dropout = config.hidden_dropout_prob
        elif hasattr(config, "dropout"):
            dropout = config.dropout
        else:
            dropout = 0.1

        self.base_model = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )

        stats_hidden_dim = 64
        self.stats_dense = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.ReLU(True),
        )

        self.pre_classifier = nn.Linear(
            config.hidden_size * 2 + stats_hidden_dim, 
            config.hidden_size
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, num_classes)
        )
    
    def forward(self, sequences, stats, segments=None, head_mask=None):
        """
        Inputs:
            sequences - torch.LongTensor with tokens
            category  - torch.LongTensor with 1 category for each element of batch
            host      - torch.LongTensor with 1 host for each element of batch
            stats     - torch.FloatTensor with values (text statistics)
            segments  - torch.LongTensor with segment indicators
        """

        # import pdb; pdb.set_trace()

        mask = (sequences != self.pad_token).float()
        bm_output = self.base_model(
            input_ids=sequences,
            attention_mask=mask,
            token_type_ids=segments,
            # head_mask=head_mask
        )
        hidden_state = bm_output[0]  # (bs, seq_len, dim)
        stats_feats = self.stats_dense(stats) # (bs, dim)

        pooled_output = torch.cat([
            torch.max(hidden_state, 1)[0],
            torch.mean(hidden_state, 1),
            stats_feats
        ], 1) # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits


class PTCFS(nn.Module):
    """
    Pooled Transformer model with categorical features (category & host features) and text statistics.
    """
    def __init__(self,
                 pretrain_dir: str,
                 num_categories: int,
                 num_hosts: int,
                 stats_dim: int,
                 num_classes: int = 1,
                 pad_token: int = 0):
        super(PTCFS, self).__init__()
        self.pad_token = pad_token
        config = AutoConfig.from_pretrained(
            pretrain_dir, 
            num_labels=num_classes
        )
        if hasattr(config, "hidden_dropout_prob"):
            dropout = config.hidden_dropout_prob
        elif hasattr(config, "dropout"):
            dropout = config.dropout
        else:
            dropout = 0.1

        self.base_model = AutoModel.from_pretrained(
            pretrain_dir, 
            config=config
        )

        categories_emb_dim = 8
        self.categories_emb = nn.Embedding(num_categories, categories_emb_dim)

        hosts_emb_dim = 16
        self.hosts_emb = nn.Embedding(num_hosts, hosts_emb_dim)

        stats_hidden_dim = 64
        self.stats_dense = nn.Sequential(
            nn.Linear(stats_dim, 64),
            nn.ReLU(True),
        )

        self.pre_classifier = nn.Linear(
            config.hidden_size * 2 + hosts_emb_dim + categories_emb_dim + stats_hidden_dim, 
            config.hidden_size
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(config.hidden_size, num_classes)
        )
    
    def forward(self, sequences, category, host, stats, segments=None, head_mask=None):
        """
        Inputs:
            sequences - torch.LongTensor with tokens
            category  - torch.LongTensor with 1 category for each element of batch
            host      - torch.LongTensor with 1 host for each element of batch
            stats     - torch.FloatTensor with values (text statistics)
            segments  - torch.LongTensor with segment indicators
        """

        # import pdb; pdb.set_trace()

        mask = (sequences != self.pad_token).float()
        bm_output = self.base_model(
            input_ids=sequences,
            attention_mask=mask,
            token_type_ids=segments,
            # head_mask=head_mask
        )
        hidden_state = bm_output[0]  # (bs, seq_len, dim)
        ctg_emb = self.categories_emb(category).squeeze(1) # (bs, dim)
        host_emb = self.hosts_emb(host).squeeze(1) # (bs, dim)
        stats_feats = self.stats_dense(stats) # (bs, dim)

        pooled_output = torch.cat([
            torch.max(hidden_state, 1)[0],
            torch.mean(hidden_state, 1),
            ctg_emb,
            host_emb,
            stats_feats
        ], 1) # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)
        return logits
