import torch
from dataclasses import dataclass
from enum import Enum
import warnings


@dataclass
class Language(Enum):
    ENGLISH: int = 5
    VIETNAMESE: int = 6


@dataclass
class BatchEncoding:
    ids: list[list[int]] | torch.Tensor = None
    mask: list[list[int]] | torch.Tensor | None = None
    
    def to(self, device: str | torch.device | int) -> "BatchEncoding":
        if isinstance(device, str) or isinstance(device, torch.device) or isinstance(device, int):
            if self.ids is not None:
                self.ids = self.ids.to(device=device)
            if self.mask is not None:
                self.mask = self.mask.to(device=device)
        else:
            warnings.warn(f"Attempting to cast a BatchEncoding to type {str(device)}. This is not supported.")
        return self
    
    def __getitem__(self, index) -> torch.Tensor:
        if index == 0:
            return self.ids
        elif index == 1:
            return self.mask
        else:
            raise IndexError("Index out of range")


@dataclass
class PairEncoding:
    input: BatchEncoding = None
    target: BatchEncoding = None
    
    def to(self, device: str | torch.device | int) -> "PairEncoding":
        if isinstance(device, str) or isinstance(device, torch.device) or isinstance(device, int):
            if self.input is not None:
                self.input = self.input.to(device=device)
            if self.target is not None:
                self.target = self.target.to(device=device)
        else:
            warnings.warn(f"Attempting to cast a PairEncoding to type {str(device)}. This is not supported.")
        return self
    
    def __getitem__(self, index) -> BatchEncoding:
        if index == 0:
            return self.input
        elif index == 1:
            return self.target
        else:
            raise IndexError("Index out of range")


@dataclass
class ModelInputEncoding:
    input: BatchEncoding = None
    target_in: BatchEncoding = None
    target_out: BatchEncoding = None
    
    def to(self, device: str | torch.device | int) -> "ModelInputEncoding":
        if isinstance(device, str) or isinstance(device, torch.device) or isinstance(device, int):
            if self.input is not None:
                self.input = self.input.to(device=device)
            if self.target_in is not None:
                self.target_in = self.target_in.to(device=device)
            if self.target_out is not None:
                self.target_out = self.target_out.to(device=device)
        else:
            warnings.warn(f"Attempting to cast a PairEncoding to type {str(device)}. This is not supported.")
        return self
    
    def __getitem__(self, index) -> BatchEncoding:
        if index == 0:
            return self.input
        elif index == 1:
            return self.target_in
        elif index == 2:
            return self.target_out
        else:
            raise IndexError("Index out of range")
    

@dataclass
class CorpusFile:
    url: str
    language: Language
    

@dataclass
class EncoderOutput:
    last_hidden_states: torch.FloatTensor = None
    attention_weight: torch.FloatTensor = None


@dataclass
class DecoderOutput:
    last_hidden_states: torch.FloatTensor = None
    self_attention_weight: torch.FloatTensor = None
    cross_attention_weight: torch.FloatTensor = None
    
    
@dataclass
class TransformerOutput:
    logits: torch.FloatTensor = None
    encoder_attention_weight: torch.FloatTensor = None
    decoder_self_attention_weight: torch.FloatTensor = None
    decoder_cross_attention_weight: torch.FloatTensor = None
