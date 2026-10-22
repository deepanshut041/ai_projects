"""An inspection-friendly bidirectional encoder with MLM and span prediction heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RobertaConfig:
    vocab_size: int = 301
    embedding_dimension: int = 32
    num_transformer_blocks: int = 2
    num_attention_heads: int = 4
    mlp_ratio: int = 4
    context_length: int = 64
    pad_token_id: int = 1
    hidden_dropout_p: float = 0.0
    attention_dropout_p: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.embedding_dimension % self.num_attention_heads:
            raise ValueError("Embedding width must be divisible by the number of heads")
        if self.context_length < 4 or self.num_transformer_blocks < 1:
            raise ValueError("The encoder needs positive depth and a usable context length")

    def to_dict(self):
        return asdict(self)


def position_ids(input_ids: torch.Tensor, pad_token_id: int = 1) -> torch.Tensor:
    real = input_ids.ne(pad_token_id).long()
    return real.cumsum(dim=1) * real + pad_token_id


def initialize(module: nn.Module, standard_deviation: float = 0.02):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=standard_deviation)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class RobertaEmbeddings(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dimension, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.context_length + config.pad_token_id + 1, config.embedding_dimension, padding_idx=config.pad_token_id)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_p)

    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[1] > self.config.context_length:
            raise ValueError("Input IDs must have shape [batch, time] within the configured context")
        positions = position_ids(input_ids, self.config.pad_token_id)
        values = self.word_embeddings(input_ids) + self.position_embeddings(positions)
        return self.dropout(self.layer_norm(values))


class RobertaAttention(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.k_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.v_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.out_proj = nn.Linear(config.embedding_dimension, config.embedding_dimension)

    def forward(self, x, attention_mask):
        batch, length, width = x.shape
        heads = self.config.num_attention_heads
        head_width = width // heads
        q = self.q_proj(x).reshape(batch, length, heads, head_width).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, length, heads, head_width).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, heads, head_width).transpose(1, 2)
        allowed_keys = attention_mask[:, None, None, :].bool()
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed_keys, dropout_p=self.config.attention_dropout_p if self.training else 0.0)
        merged = context.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.out_proj(merged)


class RobertaFeedForward(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        hidden = config.embedding_dimension * config.mlp_ratio
        self.intermediate = nn.Linear(config.embedding_dimension, hidden)
        self.output = nn.Linear(hidden, config.embedding_dimension)
        self.dropout = nn.Dropout(config.hidden_dropout_p)

    def forward(self, x):
        return self.dropout(self.output(F.gelu(self.intermediate(x))))


class RobertaEncoderLayer(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.attention_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.feed_forward = RobertaFeedForward(config)
        self.output_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)

    def forward(self, x, attention_mask):
        x = self.attention_norm(x + self.attention_dropout(self.attention(x, attention_mask)))
        return self.output_norm(x + self.feed_forward(x))


class RobertaModel(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.layers = nn.ModuleList([RobertaEncoderLayer(config) for _ in range(config.num_transformer_blocks)])
        self.apply(lambda module: initialize(module, config.initializer_range))

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("Attention mask must have the same [batch,time] shape as input IDs")
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class RobertaMLMHead(nn.Module):
    def __init__(self, config: RobertaConfig, word_embeddings: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(config.embedding_dimension, config.embedding_dimension)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.embedding_dimension, config.vocab_size)
        self.apply(lambda module: initialize(module, config.initializer_range))
        self.decoder.weight = word_embeddings.weight

    def forward(self, hidden):
        return self.decoder(self.layer_norm(F.gelu(self.dense(hidden))))


def masked_token_loss(logits, labels):
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Each token needs exactly one target or -100")
    count = int(labels.ne(-100).sum())
    loss_sum = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
    return loss_sum, count


class RobertaForMaskedLM(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.roberta = RobertaModel(config)
        self.mlm_head = RobertaMLMHead(config, self.roberta.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, labels=None):
        hidden = self.roberta(input_ids, attention_mask)
        logits = self.mlm_head(hidden)
        result = {"hidden_states": hidden, "logits": logits}
        if labels is not None:
            loss_sum, count = masked_token_loss(logits, labels)
            result.update(loss_sum=loss_sum, target_count=count, loss=loss_sum / max(count, 1))
        return result


class RobertaForQuestionAnswering(nn.Module):
    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.config = config
        self.roberta = RobertaModel(config)
        self.qa_head = nn.Linear(config.embedding_dimension, 2)
        initialize(self.qa_head, config.initializer_range)

    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        hidden = self.roberta(input_ids, attention_mask)
        start, end = self.qa_head(hidden).unbind(dim=-1)
        result = {"start_logits": start, "end_logits": end}
        if (start_positions is None) != (end_positions is None):
            raise ValueError("Provide both start and end targets")
        if start_positions is not None:
            if start_positions.shape != (input_ids.shape[0],) or end_positions.shape != start_positions.shape:
                raise ValueError("Span targets must have shape [batch]")
            valid = start_positions.ne(-100) & end_positions.ne(-100)
            if not torch.equal(start_positions.ne(-100), end_positions.ne(-100)):
                raise ValueError("Both endpoints must be ignored together")
            if valid.any() and ((start_positions[valid] < 0).any() or (end_positions[valid] >= input_ids.shape[1]).any() or (end_positions[valid] < start_positions[valid]).any()):
                raise ValueError("A target must be an ordered span within the sequence")
            start_sum = F.cross_entropy(start, start_positions, ignore_index=-100, reduction="sum")
            end_sum = F.cross_entropy(end, end_positions, ignore_index=-100, reduction="sum")
            loss_sum = (start_sum + end_sum) / 2
            count = int(valid.sum())
            result.update(loss_sum=loss_sum, target_count=count, loss=loss_sum / max(count, 1))
        return result

    def load_mlm_backbone(self, model: RobertaForMaskedLM):
        if self.config.to_dict() != model.config.to_dict():
            raise ValueError("MLM and QA architecture configurations must match exactly")
        self.roberta.load_state_dict(model.roberta.state_dict(), strict=True)


def optimizer_groups(model, weight_decay=0.01):
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def warmup_linear_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("Warmup must finish before training finishes")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
