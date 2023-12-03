import torch
from torch.nn import functional as tf
from dataclasses import dataclass
from enum import Enum
from .dataclasses_utils import Language


@dataclass
class GenerateMethod(Enum):
    GREEDY = 1
    BEAM_SEARCH = 2
    
    @classmethod
    def from_str(cls, value) -> "GenerateMethod":
        if value == "greedy":
            return cls.GREEDY
        elif value == "beam":
            return cls.BEAM_SEARCH
        else:
            raise NotImplementedError
    

def linear_act(x):
    return x


ACT2FN = {
    "gelu": tf.gelu,
    "leaky_relu": tf.leaky_relu,
    "linear": linear_act,
    "mish": tf.mish,
    "relu": tf.relu,
    "relu6": tf.relu6,
    "sigmoid": tf.sigmoid,
    "silu": tf.silu,
    "swish": tf.silu,
    "tanh": tf.tanh
}


extension2lang = {"en": Language.ENGLISH, "vi": Language.VIETNAMESE}
lang2extension = {Language.ENGLISH.value: "en", Language.VIETNAMESE.value: "vi"}
symbol2lang = {"english": Language.ENGLISH,
               "vietnamese": Language.VIETNAMESE,
               "en": Language.ENGLISH,
               "vi": Language.VIETNAMESE}
