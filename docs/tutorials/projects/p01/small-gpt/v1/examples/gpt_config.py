"""The educational P01 model contract; no pretrained compatibility claim."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab: int = 260
    context: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    hidden: int = 256
    dropout: float = 0.0

    def __post_init__(self):
        if min(self.vocab, self.context, self.width, self.heads,
               self.layers, self.hidden) < 1 or self.vocab < 260:
            raise ValueError("Positive dimensions and the structural vocabulary required")
        if self.width % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Width must divide into heads; dropout must be in [0,1)")
