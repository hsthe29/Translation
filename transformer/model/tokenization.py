import torch
from .config import TransformerConfig
from sentencepiece import SentencePieceProcessor
from ..dataclasses_utils import Language, PairEncoding, BatchEncoding
import os
from concurrent.futures import ThreadPoolExecutor


def _load_processor(path) -> SentencePieceProcessor:
    processor = SentencePieceProcessor()
    processor.Load(model_file=path)
    
    print(
        f"\033[92mLoaded processor\033[00m {path} \033[92mand found\033[00m {processor.piece_size()} \033[92munique "
        f"tokens.\033[00m")
    return processor


class BilingualTokenizer:
    def __init__(self, config: TransformerConfig):
        super(BilingualTokenizer, self).__init__()
        
        self.config = config
        self.valid_language_id = [Language.ENGLISH, Language.VIETNAMESE]
        if config.tokenizer is not None:
            self.__processor = _load_processor(config.tokenizer)
        else:
            raise ValueError("Not specific tokenizer!")
        
        assert config.vocab_size == self.__processor.vocab_size()
        assert config.pad_token_id == self.__processor.pad_id()
        assert config.unk_token_id == self.__processor.unk_id()
        assert config.bos_token_id == self.__processor.bos_id()
        assert config.eos_token_id == self.__processor.eos_id()
        
        self.all_special_ids = [self.__processor.bos_id(), 5, 6, self.__processor.eos_id(), self.__processor.unk_id(), 4]

    def __call__(self, input_texts: str | list[str],
                 target_texts: str | list[str] = None,
                 input_language: Language | None = None,
                 target_language: Language | None = None,
                 return_tensors: bool = False,
                 return_attention_mask: bool = True,
                 padding: bool = False,
                 pad_direction: str = "right",
                 max_input_length: int = None,
                 max_target_length: int = None,
                 pad_id: int = None) -> PairEncoding:
        """
        :param input_texts: str | list[str]
        :param target_texts: str | list[str]
        :param return_tensors:
        :param padding:
        :param max_input_length:
        :param max_target_length:
        :param pad_id:
        :return PairEncoding
        """
        if pad_id is None:
            pad_id = self.__processor.pad_id()
            
        if not padding and return_tensors:
            raise ValueError("Return tensor must apply padding")
        
        if max_input_length is None:
            max_input_length = (1 << 16)
            
        if max_target_length is None:
            max_target_length = (1 << 16)
        
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        batch_size = len(input_texts)
        
        input_ids = self.encode(input_texts,
                                bos_id=input_language.value if input_language else None,
                                max_length=max_input_length)
        
        if target_texts is not None:
            """Use for training"""
            if isinstance(target_texts, str):
                target_texts = [target_texts]
                assert batch_size == len(target_texts)
                
            target_ids = self.encode(target_texts,
                                     bos_id=target_language.value if target_language else None,
                                     max_length=max_target_length)
        else:
            """Use for inference"""
            target_ids = [[target_language.value if target_language else self.__processor.bos_id()]] * batch_size
        
        if padding:
            pair_encoding = PairEncoding(input=BatchEncoding(ids=input_ids,
                                                             mask=None),
                                         target=BatchEncoding(ids=target_ids,
                                                              mask=None))
            
            pair_encoding = self.pad(pair_encoding, pad_id, pad_direction, return_attention_mask, return_tensors)
            
        else:
            input_attention_mask = None
            target_attention_mask = None
            if return_attention_mask:
                input_attention_mask = []
                target_attention_mask = []
        
                for i in range(batch_size):
                    input_attention_mask.append([int(token_id != pad_id) for token_id in input_ids[i]])
                    target_attention_mask.append([int(token_id != pad_id) for token_id in target_ids[i]])
            
            pair_encoding = PairEncoding(input=BatchEncoding(ids=input_ids,
                                                             mask=input_attention_mask),
                                         target=BatchEncoding(ids=target_ids,
                                                              mask=target_attention_mask))
                
        return pair_encoding
    
    def tokenize(self, text: str | list[str]):
        return self.__processor.EncodeAsPieces(text)

    def encode(self, text: str | list[str], bos_id: int | None = None, eos_id: int | None = None, max_length: int = None):
        if bos_id is None:
            bos_id = self.__processor.bos_id()
        if eos_id is None:
            eos_id = self.__processor.eos_id()

        if max_length is None:
            max_length = 1 << 16
            
        list_ids = self.__processor.EncodeAsIds(text)
        if isinstance(text, str):
            if max_length - 2 < len(list_ids):
                list_ids = list_ids[:max_length - 2]
            list_ids = [bos_id] + list_ids + [eos_id]
        
        else:
            for i, ids in enumerate(list_ids):
                if max_length - 2 < len(ids):
                    ids = ids[:max_length - 2]
                list_ids[i] = [bos_id] + ids + [eos_id]
                
        return list_ids
    
    def pad(self, pair_encodings: PairEncoding | list[PairEncoding],
            pad_id: int = None,
            pad_direction: str = "right",
            return_attention_mask: bool = True,
            return_tensors=False):
        if isinstance(pair_encodings, PairEncoding):
            input_ids = pair_encodings.input.ids
            target_ids = pair_encodings.target.ids
        elif isinstance(pair_encodings, list):
            input_ids = []
            target_ids = []
            for _, pair_encoding in enumerate(pair_encodings):
                input_ids = input_ids + pair_encoding.input.ids
                target_ids = target_ids + pair_encoding.target.ids
        else:
            raise ValueError()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            padded_input = list(
                executor.map(
                    lambda x: self._pad_ids(x, pad_id, pad_direction, return_attention_mask),
                    [input_ids, target_ids]))
        
        _input_ids = padded_input[0][0]
        _target_ids = padded_input[1][0]
        input_attention_mask = padded_input[0][1]
        target_attention_mask = padded_input[1][1]
        
        if return_tensors:
            _input_ids = torch.tensor(padded_input[0][0], dtype=torch.int64)
            _target_ids = torch.tensor(padded_input[1][0], dtype=torch.int64)
            if return_attention_mask:
                input_attention_mask = torch.tensor(padded_input[0][1], dtype=torch.int64)
                target_attention_mask = torch.tensor(padded_input[1][1], dtype=torch.int64)
        
        return PairEncoding(input=BatchEncoding(ids=_input_ids,
                                                mask=input_attention_mask),
                            target=BatchEncoding(ids=_target_ids,
                                                 mask=target_attention_mask))
    
    def _pad_ids(self,
                 list_ids,
                 pad_id=None,
                 pad_direction="right",
                 return_attention_mask=True):
        if pad_id is None:
            pad_id = self.__processor.pad_id()
        max_length = 0
        for _, ids in enumerate(list_ids):
            max_length = max(max_length, len(ids))
        
        new_ids = []
        
        if pad_direction == "right":
            for _, ids in enumerate(list_ids):
                new_ids.append(ids + [pad_id] * (max_length - len(ids)))
        elif pad_direction == "left":
            for _, ids in enumerate(list_ids):
                new_ids.append([pad_id] * (max_length - len(ids)) + ids)
        else:
            raise ValueError("\'pad_direction\' must be either \'right\' or \'left\'")
        if return_attention_mask:
            attention_mask = []
            for _, ids in enumerate(new_ids):
                attention_mask.append([int(token_id != pad_id) for token_id in ids])
            return new_ids, attention_mask
        
        return new_ids, None
    
    def get_special_tokens_mask(self, input_ids: list[int]) -> list[int]:
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` or `encode_plus` methods.

        Args:
            input_ids (`list[int]`):
                List of ids of the input sequence.

        Returns:
            A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        
        all_special_ids = self.all_special_ids  # cache the property
        
        special_tokens_mask = [1 if token in all_special_ids else 0 for token in input_ids]
        
        return special_tokens_mask
    
    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens[0], list):
            list_ids = []
            for _, toks in enumerate(tokens):
                list_ids.append(self.__processor.PieceToId(toks))
            return list_ids
        return self.__processor.PieceToId(tokens)

    def convert_ids_to_tokens(self, input_ids):
        if isinstance(input_ids[0], list):
            list_tokens = []
            for _, ids in enumerate(input_ids):
                list_tokens.append(self.__processor.IdToPiece(ids))
            return list_tokens
        return self.__processor.IdToPiece(input_ids)

    def decode_tokens(self, tokens):
        return self.__processor.DecodePieces(tokens)
    
    def decode_ids(self, ids):
        return self.__processor.DecodeIds(ids)

    def vocab_size(self):
        return self.__processor.vocab_size()

    def pad_id(self):
        return self.__processor.pad_id()

    def unk_id(self):
        return self.__processor.unk_id()

    def bos_id(self):
        return self.__processor.bos_id()

    def eos_id(self):
        return self.__processor.eos_id()
    
    def mask_id(self):
        return self.config.mask_token_id
    
    def pad_token(self):
        return self.__processor.IdToPiece(self.__processor.pad_id())
    
    def unk_token(self):
        return self.__processor.IdToPiece(self.__processor.unk_id())
    
    def bos_token(self):
        return self.__processor.IdToPiece(self.__processor.bos_id())
    
    def eos_token(self):
        return self.__processor.IdToPiece(self.__processor.eos_id())
    
    def mask_token(self):
        return self.__processor.IdToPiece(self.config.mask_token_id)
    
    def get_token(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError()
        
        return self.__processor.IdToPiece(token_id)
