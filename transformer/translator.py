import torch
from .model import TransformerConfig, Transformer, GenerationConfig, BilingualTokenizer
from .dataclasses_utils import PairEncoding, Language, BatchEncoding, SeparatedInput
from .utils import symbol2lang
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SearchObject:
    ids: torch.Tensor | list[int]
    score: float
    weight: torch.Tensor
    
    def __getitem__(self, index) -> torch.Tensor | list[int] | list[bool] | int:
        if index == 0:
            return self.ids
        elif index == 1:
            return self.score
        elif index == 2:
            return self.weight
        else:
            raise IndexError("Index out of range")


@dataclass
class SearchOutput:
    ids: torch.Tensor | list[int]
    eos_mask: torch.Tensor | list[int] | list[bool] | int
    score: float
    weight: torch.Tensor
    
    def __getitem__(self, index) -> torch.Tensor | list[int] | list[bool] | int:
        if index == 0:
            return self.ids
        elif index == 1:
            return self.eos_mask
        elif index == 2:
            return self.score
        else:
            raise IndexError("Index out of range")


class Translator:
    def __init__(self, config: TransformerConfig, use_gpu: bool = False):
        self.config = config
        self.tokenizer = BilingualTokenizer(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = Transformer(config).to(self.device)
        
        if config.pretrained_path is not None:
            self.model.load_state_dict(torch.load(config.pretrained_path))
        else:
            raise ValueError("Model weights are not provided!")
        
        self.model.eval()
    
    def preprocess_input(self,
                         input_text: str,
                         input_language: str,
                         target_language: str
                         ) -> PairEncoding:
        
        pair_encoding = self.tokenizer(input_texts=input_text,
                                       target_texts=None,
                                       input_language=symbol2lang[input_language],
                                       target_language=symbol2lang[target_language],
                                       padding=True,
                                       return_attention_mask=True,
                                       return_tensors=True)
        return pair_encoding
    
    def __beam_search_decoder(self,
                              target: BatchEncoding,
                              input_context: SeparatedInput,
                              max_length: int = None,
                              beam_width: int = 5,
                              n_best: int = 1):
        batch_size = 1
        assert batch_size == target.ids.shape[0]
        
        iter_list = [SearchObject(ids=target.ids,
                                  score=0.0,
                                  weight=torch.zeros(
                                      (self.config.num_attn_heads, 1, input_context.features.shape[1]),
                                      dtype=torch.float32))]
        
        results: list[SearchOutput] = [] * batch_size
        
        if max_length is None:
            max_length = self.config.max_position_embeddings
        
        for iter_len in tqdm(range(max_length)):
            new_iter_list = []
            for _, (ids, score, weight) in enumerate(iter_list):
                target_in = BatchEncoding(ids, target.mask)
                probs_outputs = self.model.next_token(target_in, input_context)
                logits = probs_outputs.logits  # [1, 1, vocab_size]
                attention_weight = probs_outputs.decoder_attention_weights  # (heads, 1, Tk)
                probs = torch.softmax(logits, dim=-1)  # [num_heads, 1, vocab_size]
                beam_probs, beam_ids = probs.topk(beam_width, dim=-1)
                
                for beam_index in range(beam_width):
                    taken_ids = beam_ids[:, :, beam_index]  # [1, 1]
                    eos_mask: torch.Tensor = (taken_ids == self.model.config.eos_token_id)
                    taken_probs = beam_probs[:, :, beam_index]  # [1, 1]
                    new_ids = torch.cat((ids, taken_ids), dim=1)
                    new_score = score - torch.log(taken_probs).item()
                    new_weight = torch.cat((weight, attention_weight), dim=-2)
                    
                    if eos_mask:
                        result_ids = new_ids[0].cpu().tolist()
                        results.append(SearchOutput(ids=result_ids,
                                                    eos_mask=eos_mask.item(),
                                                    score=new_score / (iter_len + 1),
                                                    weight=new_weight.tolist()))
                    
                    else:
                        new_candidate = SearchObject(ids=new_ids,
                                                     score=new_score,
                                                     weight=new_weight)
                        
                        new_iter_list.append(new_candidate)
            
            if len(new_iter_list) == 0:
                break
            
            new_iter_list.sort(key=lambda x: x.score / (iter_len + 1))
            iter_list = new_iter_list[:beam_width]
            
            if len(results) >= n_best * 2:
                break
        
        results.sort(key=lambda x: x.score)
        
        if len(results) < n_best:
            for i in range(n_best - len(results)):
                temp = iter_list[i]
                results.append(SearchOutput(ids=temp.ids[0].cpu().tolist(),
                                            eos_mask=False,
                                            score=temp.score / (len(temp.ids[0].cpu().tolist()) - 1),
                                            weight=temp.weight.tolist()))
        else:
            results = results[:n_best]
        
        return results
    
    def generate(self,
                 input: BatchEncoding,
                 target: BatchEncoding,
                 generation_config: GenerationConfig = GenerationConfig()):
        input_context = self.model.extract_features(input)
        
        if generation_config.max_length is None:
            max_length = self.config.max_position_embeddings
        else:
            max_length = generation_config.max_length
        
        results = self.__beam_search_decoder(target,
                                             input_context,
                                             max_length=max_length,
                                             beam_width=generation_config.beam_width,
                                             n_best=generation_config.n_best)
        
        output_results = []
        for result in results:
            tokens = self.tokenizer.convert_ids_to_tokens(result.ids)
            translation = self.tokenizer.decode_ids(result.ids)
            
            output_results.append({
                "token": tokens,
                "translation": translation,
                "score": result.score,
                "weight": result.weight
            })
        input_tokens = self.tokenizer.convert_ids_to_tokens(input.ids[0].tolist())
        return input_tokens, output_results
    
    
def load_translator(config_file):
    config = TransformerConfig.load(config_file)
    return Translator(config)
