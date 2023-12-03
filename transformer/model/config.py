import json
import os
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    architecture = "transformer"
    tokenizer: str = None
    vocab_size: int = 16000,
    hidden_size: int = 1024
    encoder_hidden_layers: int = 6
    decoder_hidden_layers: int = 6
    num_attn_heads: int = 16
    intermediate_size: int = 2048
    hidden_act: str = "gelu"
    shared_embeddings: bool = True
    embed_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    attn_dropout_prob: float = 0.1
    mlm_probability: float = 0.15
    label_smoothing: float = 0.1
    max_position_embeddings: int = 256
    layer_norm_eps: float = 1e-6
    pad_token_id: int = 0
    unk_token_id: int = 1
    bos_token_id: int = 2
    eos_token_id: int = 3
    mask_token_id: int = 4
    seed: int = 291
    
    @classmethod
    def load(cls, file: str):
        with open(file, mode="r") as f:
            dict_data = json.load(f)
        
        if dict_data["architecture"] != cls.architecture:
            return ValueError("Model type is not matched!")
        
        config = cls(**dict_data["params"])
        return config
    
    def __str__(self):
        return f"TransformerConfig(architecture={self.architecture}, params={str(self.__dict__)})"


@dataclass
class GenerationConfig:
    beam_width: int = 5
    n_best: int = 1
    max_length: int | None = None
    