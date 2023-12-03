import torch
from ..model.tokenization import BilingualTokenizer
from ..dataclasses_utils import PairEncoding, ModelInputEncoding, BatchEncoding
from dataclasses import dataclass


@dataclass
class DataCollator:
    tokenizer: BilingualTokenizer
    mlm_probability: float | None
    
    def _merge_list(self, list_pair_ids: list[tuple[list[int], list[int]]]):
        input_ids = []
        target_ids = []
        input_len = 0
        target_len = 0
        
        for _, (inp_ids, tar_ids) in enumerate(list_pair_ids):
            input_len = max(input_len, len(inp_ids))
            target_len = max(target_len, len(tar_ids))
            input_ids.append(inp_ids)
            target_ids.append(tar_ids)
        
        return (input_ids, input_len), (target_ids, target_len)
    
    def _pad(self, list_ids, length, pad_id):
        for i, ids in enumerate(list_ids):
            list_ids[i] = ids + [pad_id]*(length - len(ids))
        
        return torch.tensor(list_ids, dtype=torch.int64)
    
    def __call__(self, list_pair_ids: list[tuple[list[int], list[int]]]) -> ModelInputEncoding:
        """Pair ids: [(input_ids, target_ids)]"""
        pad_id = self.tokenizer.pad_id()
        
        input_info, target_info = self._merge_list(list_pair_ids)
        
        input_ids = self._pad(input_info[0], input_info[1], pad_id)
        target_ids = self._pad(target_info[0], target_info[1], pad_id)
        
        target_out_ids = torch.roll(target_ids, shifts=-1, dims=-1)
        target_out_ids[:, -1] = pad_id
        
        input_mask = input_ids.ne(pad_id).to(dtype=torch.int64)
        target_mask = target_ids.ne(pad_id).to(dtype=torch.int64)
        
        if self.mlm_probability is not None:
            target_ids = self.apply_masked_tokens(target_ids)
        
        return ModelInputEncoding(input=BatchEncoding(input_ids, input_mask),
                                  target_in=BatchEncoding(target_ids, target_mask),
                                  target_out=BatchEncoding(target_out_ids, target_mask))
    
    def apply_masked_tokens(self, input_ids: torch.Tensor):
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val) for val in input_ids.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        input_ids[masked_indices] = self.tokenizer.mask_id()
        
        return input_ids
