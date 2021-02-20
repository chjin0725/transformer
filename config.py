from dataclasses import dataclass


@dataclass
class Config:
    device: str
    emb_dim: int
    ffn_dim: int
    num_attention_heads: int
    attention_drop_out: float
    drop_out: float
    max_position: int
    num_encoder_layers: int
    num_decoder_layers: int
    batch_size: int
    learning_rate: float
    n_epochs: int
    gradient_clip: int