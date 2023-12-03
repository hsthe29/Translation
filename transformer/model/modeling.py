import torch
from torch import nn
from .config import TransformerConfig
from .layers import EncoderLayer, DecoderLayer, TransformerEmbeddings, EncoderOutput, DecoderOutput
from ..dataclasses_utils import TransformerOutput


class TransformerEncoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerEncoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(config.encoder_hidden_layers):
            self.layers.append(EncoderLayer(config))
    
    def forward(self, input_features, pos_embed, attention_mask=None):
        x = input_features
        layer_outputs = None
        for _, layer in enumerate(self.layers):
            layer_outputs = layer(x, pos_embed, attention_mask)
            x = layer_outputs.last_hidden_states
        
        return layer_outputs


class TransformerDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerDecoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(config.encoder_hidden_layers):
            self.layers.append(DecoderLayer(config))
    
    def forward(self,
                target_in_features: torch.FloatTensor,
                input_features: torch.FloatTensor,
                pos_embeddings: torch.FloatTensor,
                target_in_mask: torch.FloatTensor | None,
                input_mask: torch.Tensor | None = None):
        layer_outputs = None
        x = target_in_features
        for _, layer in enumerate(self.layers):
            layer_outputs = layer(x,
                                  input_features,
                                  pos_embeddings,
                                  target_in_mask,
                                  input_mask)
            x = layer_outputs.last_hidden_states
        
        return layer_outputs


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        
        self.config = config
        
        self.shared_embeddings = TransformerEmbeddings(config)
        
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        self.final_fc = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self,
                input_ids: torch.Tensor,
                target_in_ids: torch.Tensor,
                input_mask: torch.Tensor | None = None,
                target_in_mask: torch.Tensor | None = None):
        """
            Params with shapes:
                input_ids: [batch_size, Ti]
                target_in_ids: [batch_size, Tt]
                input_attention_mask: [batch_size, Ti]
                target_in_attention_mask: [batch_size, Tt]
            
        """
        if target_in_mask is not None:
            target_in_mask = apply_causal_mask(target_in_mask)
            target_in_mask = (1.0 - target_in_mask) * torch.finfo(torch.float32).min
        
        if input_mask is not None:
            input_mask = (1.0 - input_mask[:, None, None, :].to(dtype=torch.float32)) * torch.finfo(
                torch.float32).min
        
        input_features = self.shared_embeddings(input_ids)
        target_features = self.shared_embeddings(target_in_ids)
        
        encoder_outputs: EncoderOutput = self.encoder(input_features[0], input_features[1], input_mask)
        decoder_outputs: DecoderOutput = self.decoder(target_features[0],
                                                      encoder_outputs.last_hidden_states,
                                                      target_features[1],
                                                      target_in_mask,
                                                      input_mask)
        
        logits = self.final_fc(decoder_outputs.last_hidden_states)
        
        return TransformerOutput(
            logits=logits,
            encoder_attention_weight=encoder_outputs.attention_weight.detach(),
            decoder_self_attention_weight=decoder_outputs.self_attention_weight.detach(),
            decoder_cross_attention_weight=decoder_outputs.cross_attention_weight.detach()
        )
    
    @torch.no_grad()
    def extract_input_features(self,
                               ids: torch.LongTensor,
                               mask: torch.LongTensor = None) -> EncoderOutput:
        """Use for inference"""
        
        features = self.shared_embeddings(ids)
        if mask is not None:
            mask = (1.0 - mask[:, None, None, :].to(dtype=torch.float32)) * torch.finfo(
                torch.float32).min
        return self.encoder(features[0], features[1], mask)
    
    @torch.no_grad()
    def predict_ids_probabilities(self,
                                  target_in_ids,
                                  input_features,
                                  target_in_mask=None,
                                  input_mask=None) -> TransformerOutput:
        """Use for inference"""
        
        self.eval()
        features = self.shared_embeddings(target_in_ids)
        if target_in_mask is not None:
            target_in_mask = apply_causal_mask(target_in_mask)
            target_in_mask = (1.0 - target_in_mask) * torch.finfo(torch.float32).min
        
        decoder_outputs = self.decoder(features[0],
                                       input_features,
                                       features[1],
                                       target_in_mask,
                                       input_mask)
        
        logits = self.final_fc(decoder_outputs.last_hidden_states)
        
        return TransformerOutput(
            logits=logits,
            decoder_self_attention_weight=decoder_outputs.self_attention_weight.detach(),
            decoder_cross_attention_weight=decoder_outputs.cross_attention_weight.detach()
        )


def apply_causal_mask(mask):
    n = mask.shape[1]
    causal_mask = torch.tril(torch.ones(n, n)).to(mask.device, dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)[:, None, None, :]
    final_mask = torch.minimum(mask, causal_mask)
    return final_mask
