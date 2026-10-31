"""A complete encoder-decoder Transformer with bounded greedy decoding."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from translation_attention import Attention


@dataclass(frozen=True)
class TransformerConfig:
    src_vocab_size: int = 10
    tgt_vocab_size: int = 12
    embedding_dimension: int = 32
    num_attention_heads: int = 4
    encoder_depth: int = 1
    decoder_depth: int = 1
    mlp_ratio: int = 4
    attention_dropout_p: float = 0.0
    hidden_dropout_p: float = 0.0
    max_src_len: int = 32
    max_tgt_len: int = 32
    src_pad_id: int = 0
    tgt_pad_id: int = 1
    fused_attn: bool = True

    def __post_init__(self):
        if self.embedding_dimension <= 0 or self.embedding_dimension % 2:
            raise ValueError("sinusoidal positions require a positive even embedding width")
        if self.num_attention_heads <= 0 or self.embedding_dimension % self.num_attention_heads:
            raise ValueError("embedding width must be divisible by positive head count")
        if min(self.encoder_depth, self.decoder_depth, self.mlp_ratio,
               self.max_src_len, self.max_tgt_len) < 1:
            raise ValueError("depths, expansion and lengths must be positive")
        if not 0 <= self.src_pad_id < self.src_vocab_size or not 0 <= self.tgt_pad_id < self.tgt_vocab_size:
            raise ValueError("padding ids must belong to their own vocabularies")


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super().__init__()
        if embed_dim % 2:
            raise ValueError("choose an even width for paired sine/cosine dimensions")
        position = torch.arange(max_len, dtype=torch.float32)[:, None]
        frequency = torch.exp(-math.log(10000.0) * torch.arange(0, embed_dim, 2) / embed_dim)
        encoding = torch.zeros(max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("encodings", encoding)

    def forward(self, x):
        if x.shape[1] > len(self.encodings):
            raise ValueError("sequence exceeds the positional-encoding capacity")
        return x + self.encodings[:x.shape[1]].to(dtype=x.dtype)


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.src_embeddings = nn.Embedding(config.src_vocab_size, config.embedding_dimension,
                                            padding_idx=config.src_pad_id)
        self.tgt_embeddings = nn.Embedding(config.tgt_vocab_size, config.embedding_dimension,
                                            padding_idx=config.tgt_pad_id)
        self.src_positional_encodings = PositionalEncoding(config.max_src_len, config.embedding_dimension)
        self.tgt_positional_encodings = PositionalEncoding(config.max_tgt_len, config.embedding_dimension)

    def forward_src(self, input_ids):
        return self.src_positional_encodings(self.src_embeddings(input_ids))

    def forward_tgt(self, input_ids):
        return self.tgt_positional_encodings(self.tgt_embeddings(input_ids))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.embedding_dimension * config.mlp_ratio
        self.intermediate_dense = nn.Linear(config.embedding_dimension, hidden)
        self.activation = nn.GELU()
        self.intermediate_dropout = nn.Dropout(config.hidden_dropout_p)
        self.output_dense = nn.Linear(hidden, config.embedding_dimension)
        self.output_dropout = nn.Dropout(config.hidden_dropout_p)

    def forward(self, x):
        x = self.intermediate_dropout(self.activation(self.intermediate_dense(x)))
        return self.output_dropout(self.output_dense(x))


def make_attention(config):
    return Attention(config.embedding_dimension, config.num_attention_heads,
                     config.attention_dropout_p, fused_attn=config.fused_attn)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.enc_attention = make_attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_p)
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self, x, attention_mask=None):
        x = self.layer_norm(x + self.dropout(self.enc_attention(x, attention_mask=attention_mask)))
        return self.final_layer_norm(x + self.feed_forward(x))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dec_attention = make_attention(config)
        self.dec_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.dec_attention_layernorm = nn.LayerNorm(config.embedding_dimension)
        self.cross_attention = make_attention(config)
        self.cross_attention_dropout = nn.Dropout(config.hidden_dropout_p)
        self.cross_attention_layernorm = nn.LayerNorm(config.embedding_dimension)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.embedding_dimension)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        update = self.dec_attention(tgt, attention_mask=tgt_mask, causal=True)
        tgt = self.dec_attention_layernorm(tgt + self.dec_attention_dropout(update))
        update = self.cross_attention(src, tgt, attention_mask=src_mask)
        tgt = self.cross_attention_layernorm(tgt + self.cross_attention_dropout(update))
        return self.final_layer_norm(tgt + self.feed_forward(tgt))


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encodings = Embeddings(config)
        self.encoder = nn.ModuleList([TransformerEncoderLayer(config) for _ in range(config.encoder_depth)])
        self.decoder = nn.ModuleList([TransformerDecoderLayer(config) for _ in range(config.decoder_depth)])
        self.head = nn.Linear(config.embedding_dimension, config.tgt_vocab_size)
        self.apply(_init_weights)

    def encode(self, src_ids, src_attention_mask=None):
        if src_attention_mask is None:
            src_attention_mask = src_ids != self.config.src_pad_id
        if not src_attention_mask.any(dim=-1).all():
            raise ValueError("each source must contain at least one usable token")
        memory = self.encodings.forward_src(src_ids)
        for layer in self.encoder:
            memory = layer(memory, src_attention_mask)
        return memory

    def decode(self, memory, tgt_ids, src_attention_mask, tgt_attention_mask=None):
        if tgt_attention_mask is None:
            tgt_attention_mask = tgt_ids != self.config.tgt_pad_id
        hidden = self.encodings.forward_tgt(tgt_ids)
        for layer in self.decoder:
            hidden = layer(memory, hidden, src_attention_mask, tgt_attention_mask)
        return self.head(hidden)

    def forward(self, src_ids, tgt_ids, src_attention_mask=None, tgt_attention_mask=None):
        if src_attention_mask is None:
            src_attention_mask = src_ids != self.config.src_pad_id
        memory = self.encode(src_ids, src_attention_mask)
        return self.decode(memory, tgt_ids, src_attention_mask, tgt_attention_mask)

    @torch.no_grad()
    def inference(self, src_ids, tgt_start_id=2, tgt_end_id=3, max_len=16,
                  src_attention_mask=None):
        """Return [batch, length], retaining BOS/EOS and padding completed rows."""
        if not 2 <= max_len <= self.config.max_tgt_len:
            raise ValueError("max_len includes BOS and must fit the target position capacity")
        if not (0 <= tgt_start_id < self.config.tgt_vocab_size and
                0 <= tgt_end_id < self.config.tgt_vocab_size):
            raise ValueError("generation ids must belong to the target vocabulary")
        if len({tgt_start_id, tgt_end_id, self.config.tgt_pad_id}) != 3:
            raise ValueError("BOS, EOS and padding must have different ids")
        was_training = self.training
        self.eval()
        try:
            source_mask = src_ids != self.config.src_pad_id if src_attention_mask is None else src_attention_mask
            memory = self.encode(src_ids, source_mask)
            generated = torch.full((src_ids.shape[0], 1), tgt_start_id, dtype=torch.long, device=src_ids.device)
            finished = torch.zeros(src_ids.shape[0], dtype=torch.bool, device=src_ids.device)
            for _ in range(max_len - 1):
                logits = self.decode(memory, generated, source_mask)[:, -1].clone()
                logits[:, self.config.tgt_pad_id] = -torch.inf
                logits[:, tgt_start_id] = -torch.inf
                next_id = logits.argmax(dim=-1)
                next_id = torch.where(finished, torch.full_like(next_id, self.config.tgt_pad_id), next_id)
                generated = torch.cat([generated, next_id[:, None]], dim=1)
                finished |= next_id == tgt_end_id
                if finished.all():
                    break
            return generated
        finally:
            self.train(was_training)


def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def main():
    from translation_data import make_fixture
    src, tgt, batch = make_fixture()
    torch.manual_seed(7)
    model = Transformer(TransformerConfig(src_vocab_size=src.vocab_size, tgt_vocab_size=tgt.vocab_size))
    logits = model(batch["src_input_ids"], batch["tgt_input_ids"], batch["src_pad_mask"], batch["tgt_pad_mask"])
    print("logits", tuple(logits.shape), "parameters", sum(p.numel() for p in model.parameters()))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["tgt_outputs"].flatten())
    loss.backward()
    print("finite loss", bool(torch.isfinite(loss)), "head gradient", bool(model.head.weight.grad.abs().sum() > 0))


if __name__ == "__main__":
    main()
