import torch
from torch import nn
from .config import TransformerConfig
from .layers import EncoderLayer, DecoderLayer, Embeddings, EncoderOutput, DecoderOutput
from ..dataclasses_utils import TransformerOutput, SeparatedInput, BatchEncoding


class Encoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(config.encoder_hidden_layers):
            self.layers.append(EncoderLayer(config))
    
    def forward(self, input: SeparatedInput):
        output = None
        for _, layer in enumerate(self.layers):
            output = layer(input)
            input = SeparatedInput(output.last_hidden_states, input.pos_embedding, input.attention_mask)
        
        return output


class Decoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for i in range(config.encoder_hidden_layers):
            self.layers.append(DecoderLayer(config))
    
    def forward(self, target: SeparatedInput, input_context: SeparatedInput):
        output = None
        for _, layer in enumerate(self.layers):
            output = layer(target, input_context)
            target = SeparatedInput(output.last_hidden_states, target.pos_embedding, target.attention_mask)
        
        return output


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Transformer, self).__init__()
        
        self.config = config
        
        self.embeddings = Embeddings(config)
        
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        self.final_fc = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input: BatchEncoding, target: BatchEncoding):
        """
            Params with shapes:
                input_ids: [batch_size, Ti]
                target_in_ids: [batch_size, Tt]
                input_attention_mask: [batch_size, Ti]
                target_in_attention_mask: [batch_size, Tt]
            
        """
        input_mask = input.mask
        target_mask = target.mask
        
        if target_mask is not None:
            target_mask = apply_causal_mask(target_mask)
            target_mask = (1.0 - target_mask) * torch.finfo(torch.float32).min
        
        if input_mask is not None:
            input_mask = (1.0 - input_mask[:, None, None, :].to(dtype=torch.float32)) * torch.finfo(
                torch.float32).min
        
        input_embeddings = self.embeddings(input.ids)
        target_embeddings = self.embeddings(target.ids)
        
        input_embeddings.attention_mask = input_mask
        target_embeddings.attention_mask = target_mask
        
        encoder_outputs: EncoderOutput = self.encoder(input_embeddings)
        
        input_context = SeparatedInput(encoder_outputs.last_hidden_states,
                                       input_embeddings.pos_embedding,
                                       input_embeddings.attention_mask)
        
        decoder_outputs: DecoderOutput = self.decoder(target_embeddings, input_context)
        
        logits = self.final_fc(decoder_outputs.last_hidden_states)
        
        return TransformerOutput(
            logits=logits,
            encoder_attention_weights=encoder_outputs.attention_weights,
            decoder_attention_weights=decoder_outputs.attention_weights
        )
    
    @torch.no_grad()
    def extract_features(self, input: BatchEncoding) -> SeparatedInput:
        """Use for inference"""
        
        input_embeddings = self.embeddings(input.ids)
        attention_mask = input.mask
        
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask[:, None, None, :].to(dtype=torch.float32)) * torch.finfo(
                torch.float32).min
        
        input_embeddings.attention_mask = attention_mask
        
        encoder_output: EncoderOutput = self.encoder(input_embeddings)
        
        input_context = SeparatedInput(encoder_output.last_hidden_states,
                                       input_embeddings.pos_embedding,
                                       input_embeddings.attention_mask)
        
        return input_context
    
    @torch.no_grad()
    def next_token(self,
                   target: BatchEncoding,
                   input_context: SeparatedInput) -> TransformerOutput:
        """Use for inference"""
        
        target_embeddings = self.embeddings(target.ids)
        target_mask = target.mask
        
        if target_mask is not None:
            target_mask = apply_causal_mask(target_mask)
            target_mask = (1.0 - target_mask) * torch.finfo(torch.float32).min
        
        target_embeddings.attention_mask = target_mask
        
        decoder_outputs = self.decoder(target_embeddings, input_context)
        
        logits = self.final_fc(decoder_outputs.last_hidden_states)
        
        return TransformerOutput(
            logits=logits[:, -1:, :],
            decoder_attention_weights=decoder_outputs.attention_weights[0, :, -1:, :]
        )


def apply_causal_mask(mask):
    n = mask.shape[1]
    causal_mask = torch.tril(torch.ones(n, n)).to(mask.device, dtype=torch.float32)
    mask = mask.to(dtype=torch.float32)[:, None, None, :]
    final_mask = torch.minimum(mask, causal_mask)
    return final_mask
