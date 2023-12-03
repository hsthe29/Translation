import torch
from .model import TransformerConfig, Transformer, GenerationConfig, BilingualTokenizer
from .dataclasses_utils import PairEncoding, Language
from .utils import symbol2lang
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class DecodeCandidate:
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
class DecodeOutput:
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
    def __init__(self, pretrained_model_path: str, config: TransformerConfig, use_gpu: bool = False):
        self.config = config
        self.tokenizer = BilingualTokenizer(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = Transformer(config).to(self.device)
        self.model.load_state_dict(torch.load(pretrained_model_path))
        
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
                              target_start_ids: torch.Tensor,
                              input_features: torch.Tensor,
                              input_mask: torch.Tensor = None,
                              max_length: int = None,
                              beam_width: int = 5,
                              n_best: int = 1):
        batch_size = 1
        assert batch_size == target_start_ids.shape[0]
        
        iter_list = [DecodeCandidate(ids=target_start_ids,
                                     score=0.0,
                                     weight=torch.zeros(
                                         (self.config.num_attn_heads, 1, input_features.shape[1]),
                                         dtype=torch.float32))]
        
        results: list[DecodeOutput] = [] * batch_size
        
        if max_length is None:
            max_length = self.config.max_position_embeddings
        # walk over each step in sequence
        for iter_len in tqdm(range(max_length)):
            new_iter_list = []
            for _, (ids, score, weight) in enumerate(iter_list):
                target_mask = ids.ne(self.config.pad_token_id).to(dtype=torch.int64)
                probs_outputs = self.model.predict_ids_probabilities(ids,
                                                                     input_features,
                                                                     target_in_mask=target_mask,
                                                                     input_mask=input_mask)
                logits = probs_outputs.logits[:, -1, :]  # [1, 1, vocab_size]
                attention_weight = probs_outputs.decoder_cross_attention_weight[0, :, -1:, :]  # (heads, 1, Tk)
                probs = torch.softmax(logits, dim=-1)  # [1, ]
                beam_probs, beam_ids = probs.topk(beam_width, dim=-1)
                
                for beam_index in range(beam_width):
                    taken_ids = beam_ids[:, beam_index]  # (1,)
                    eos_mask: torch.Tensor = (taken_ids == self.model.config.eos_token_id)
                    # taken_ids = torch.where(eos_mask, self.config.pad_token_id, taken_ids)
                    taken_probs = beam_probs[:, beam_index]  # (1,)
                    new_ids = torch.cat((ids, taken_ids[:, None]), dim=1)
                    new_score = score - torch.log(taken_probs).squeeze(0).item()
                    new_weight = torch.cat((weight, attention_weight), dim=-2)
                    
                    if eos_mask:
                        result_ids = new_ids[0].cpu().tolist()
                        results.append(DecodeOutput(ids=result_ids,
                                                    eos_mask=eos_mask.squeeze(0).cpu().tolist(),
                                                    score=new_score / (iter_len + 1),
                                                    weight=new_weight.tolist()))
                    
                    else:
                        new_candidate = DecodeCandidate(ids=new_ids,
                                                        score=new_score,
                                                        weight=new_weight)
                        
                        new_iter_list.append(new_candidate)
            
            if len(new_iter_list) == 0:
                break
            
            del iter_list
            new_iter_list.sort(key=lambda x: x.score / (iter_len + 1))
            iter_list = new_iter_list[:beam_width]
            
            if len(results) >= n_best * 2:
                break
        
        results.sort(key=lambda x: x.score)
        
        if len(results) < n_best:
            for i in range(n_best - len(results)):
                temp = iter_list[i]
                results.append(DecodeOutput(ids=temp.ids[0].cpu().tolist(),
                                            eos_mask=False,
                                            score=temp.score / (len(temp.ids[0].cpu().tolist()) - 1),
                                            weight=temp.weight.tolist()))
        else:
            results = results[:n_best]
        
        return results
    
    def generate(self,
                 input_ids: torch.Tensor,
                 target_in_ids: torch.Tensor,
                 input_mask: torch.Tensor | None = None,
                 config: GenerationConfig = GenerationConfig()):
        input_ids = input_ids.to(self.device)
        target_in_ids = target_in_ids.to(self.device)
        if input_mask is not None:
            input_mask = input_mask.to(self.device)
        
        input_context = self.model.extract_input_features(input_ids, input_mask)
        
        if config.max_length is None:
            max_length = self.config.max_position_embeddings
        else:
            max_length = config.max_length
        
        results = self.__beam_search_decoder(target_in_ids,
                                             input_context.last_hidden_states,
                                             input_mask,
                                             max_length=max_length,
                                             beam_width=config.beam_width,
                                             n_best=config.n_best)
        
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
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        return input_tokens, output_results
