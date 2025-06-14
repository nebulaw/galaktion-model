from dataclasses import dataclass


@dataclass
class Config:
    train_steps: int = 5_000
    valid_steps: int = 100

    train_batch_size: int = 128
    valid_batch_size: int = 128

    logging_steps: int = 100
    learning_rate: float = 3e-4

@dataclass
class ModelConfig:
    block_size: int = 128
    d_model: int = 256
    head_dim: int = 256
    n_head: int = 8
    layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.2


