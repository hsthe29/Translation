import torch
from .model import BilingualTokenizer
import collections
import math
from torchtext.data.utils import ngrams_iterator


def _compute_ngram_counter(tokens, max_n):
    """Create a Counter with a count of unique n-grams in the tokens list

    Args:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted

    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count

    Examples:
        >>> from torchtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['me', 'me', 'you']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1})
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n))
    
    return ngrams_counter


class Metric:
    
    def cumulate(self, logits, y_trues, mask):
        return NotImplementedError
    
    def get_score(self):
        return NotImplementedError
    
    def reset(self):
        return NotImplementedError
    

class MaskedAccuracyScore(Metric):
    def __init__(self):
        self.true_ids = 0.0
        self.total_ids = 0.0
    
    @torch.no_grad()
    def cumulate(self,
                 logits: torch.FloatTensor,
                 y_trues: torch.LongTensor,
                 mask: torch.LongTensor):
        pred_ids = torch.argmax(logits, dim=-1)
        accuracy = y_trues.eq(pred_ids).to(dtype=torch.float32)
        
        mask = mask.to(dtype=torch.float32)
        accuracy = accuracy * mask
        
        self.true_ids += accuracy.sum().item()
        self.total_ids += mask.sum().item()
    
    def get_score(self):
        return self.true_ids / self.total_ids
    
    def reset(self):
        self.true_ids = 0.0
        self.total_ids = 0.0
    
    def __repr__(self):
        return f"MaskedAccuracyScore(true_ids={self.true_ids}, total_ids={self.total_ids})"


class BLEUScore(Metric):
    def __init__(self,
                 tokenizer: BilingualTokenizer,
                 max_n: int = 4,
                 weights: list[float] = [0.25, 0.25, 0.25, 0.25]):
        assert len(weights) == max_n
        assert sum(weights) == 1.0
        
        self.tokenizer = tokenizer
        self.max_n = max_n
        self.clipped_counts = torch.zeros(max_n)
        self.total_counts = torch.zeros(max_n)
        self.weights = torch.tensor(weights)
        
        self.candidate_len = 0.0
        self.refs_len = 0.0
    
    def transform(self,
                  logits: torch.FloatTensor,
                  y_trues: torch.LongTensor,
                  mask: torch.LongTensor):
        
        pad_id = self.tokenizer.pad_id()
        eos_id = self.tokenizer.eos_id()
        pred_ids = torch.argmax(logits, dim=-1)
        eos_mask = (pred_ids == eos_id)
        idx = torch.cumsum(eos_mask, -1).bool() | (~mask.bool())
        pred_ids = torch.where(idx, pad_id, pred_ids)
        
        list_seqs = self.tokenizer.decode_ids(pred_ids.tolist())
        list_references = self.tokenizer.decode_ids(y_trues.tolist())
        
        for i, (seq, reference) in enumerate(zip(list_seqs, list_references)):
            list_seqs[i] = seq.strip().split()
            list_references[i] = [reference.strip().split()]
        return list_seqs, list_references
    
    @torch.no_grad()
    def cumulate(self,
                 logits: torch.FloatTensor,
                 y_trues: torch.LongTensor,
                 attention_mask: torch.LongTensor):
        candidate_corpus, references_corpus = self.transform(logits, y_trues, attention_mask)
        
        for (candidate, refs) in zip(candidate_corpus, references_corpus):
            current_candidate_len = len(candidate)
            self.candidate_len += current_candidate_len
            
            # Get the length of the reference that's closest to the candidate
            refs_len_list = [float(len(ref)) for ref in refs]
            self.refs_len += min(refs_len_list, key=lambda x: abs(current_candidate_len - x))
            
            reference_counters = _compute_ngram_counter(refs[0], self.max_n)
            for ref in refs[1:]:
                reference_counters = reference_counters | _compute_ngram_counter(ref, self.max_n)
            
            candidate_counter = _compute_ngram_counter(candidate, self.max_n)
            
            clipped_counter = candidate_counter & reference_counters
            
            for ngram, count in clipped_counter.items():
                self.clipped_counts[len(ngram) - 1] += count
            
            for i in range(self.max_n):
                # The number of N-grams in a `candidate` of T tokens is `T - (N - 1)`
                self.total_counts[i] += max(current_candidate_len - i, 0)
    
    def get_score(self):
        if min(self.clipped_counts) == 0:
            return 0.0
        else:
            pn = self.clipped_counts / self.total_counts
            log_pn = self.weights * torch.log(pn)
            score = torch.exp(sum(log_pn))
            
            bp = math.exp(min(1 - self.refs_len / self.candidate_len, 0))
            
            return bp * score.item()
    
    def __repr__(self):
        return f'''BLEUScore(
    max_n={self.max_n},
    candidate_len={self.candidate_len},
    refs_len={self.refs_len},
    clipped_counts={self.clipped_counts.tolist()},
    total_counts={self.total_counts.tolist()},
    weights={self.weights.tolist()}
)'''
    
    def reset(self):
        self.clipped_counts = torch.zeros(self.max_n)
        self.total_counts = torch.zeros(self.max_n)
        self.candidate_len = 0.0
        self.refs_len = 0.0
    
    def __str__(self):
        return self.__repr__()
