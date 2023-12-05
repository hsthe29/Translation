import torch
from torch import nn
from torch.nn import functional as tf
from ..dataclasses_utils import EncoderOutput, DecoderOutput, SeparatedInput
from .config import TransformerConfig
from ..utils import ACT2FN

import math


class Embeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(Embeddings, self).__init__()
        self.ids_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        self.pos_embedding = nn.Embedding(config.max_position_embeddings,
                                          config.hidden_size)
    
    def forward(self, ids) -> SeparatedInput:
        ids_embedding = self.ids_embedding(ids)
        pos_embedding = self.pos_embedding(torch.arange(ids.shape[-1]).unsqueeze(0).to(ids.device))
        
        return SeparatedInput(ids_embedding, pos_embedding)


class AddNormLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(AddNormLayer, self).__init__()
        
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, x, x_residual):
        return self.layer_norm(x + x_residual)


class FeedForwardLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(FeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, features):
        x = self.act_fn(self.fc1(features))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(config.attn_dropout_prob)
        self.scaling = math.sqrt(config.hidden_size // config.num_attn_heads)
        
    def forward(self,
                query: torch.FloatTensor,
                key: torch.FloatTensor,
                value: torch.FloatTensor,
                mask: torch.FloatTensor | None = None):
        scores = torch.matmul(query, key.transpose(-1, -2))
        
        scaled_scores = scores / self.scaling
        
        if mask is not None:
            scaled_scores = scaled_scores + mask
            
        weights = tf.softmax(scaled_scores, dim=-1)
        weights = self.dropout(weights)
        
        outputs = torch.matmul(weights, value)
        return outputs, weights
        

class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(SelfAttention, self).__init__()
        
        self.num_attn_heads = config.num_attn_heads
        
        assert config.hidden_size % self.num_attn_heads == 0
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.base_attention = ScaledDotProductAttention(config)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
    def split_heads(self, x):
        # [N, T, D] -> [N, T, h, S] -> [N, h, T, S]
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def merge_heads(self, x):
        # [N, h, T, S] -> [N, T, h, S] -> [N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (-1,)
        return x.view(new_x_shape)
    
    def forward(self, input: SeparatedInput):
        x = input.features + input.pos_embedding
        query = self.split_heads(self.q_proj(x))
        key = self.split_heads(self.k_proj(x))
        value = self.split_heads(self.v_proj(x))
        
        outputs, weights = self.base_attention(query, key, value, input.attention_mask)
        
        outputs = self.merge_heads(outputs)
        outputs = self.out_proj(outputs)
        
        return outputs, weights


class CrossAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(CrossAttention, self).__init__()
        
        self.num_attn_heads = config.num_attn_heads
        
        assert config.hidden_size % self.num_attn_heads == 0
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.base_attention = ScaledDotProductAttention(config)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
    
    def split_heads(self, x):
        # [N, T, D] -> [N, T, h, S] -> [N, h, T, S]
        new_x_shape = x.size()[:-1] + (self.num_attn_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def merge_heads(self, x):
        # [N, h, T, S] -> [N, T, h, S] -> [N, T, D]
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (-1,)
        return x.view(new_x_shape)
    
    def forward(self,
                q_input: SeparatedInput,
                k_input: SeparatedInput,
                v_input: SeparatedInput):
        q = q_input.features + q_input.pos_embedding
        k = k_input.features + k_input.pos_embedding
        v = v_input.features + v_input.pos_embedding
        
        query = self.split_heads(self.q_proj(q))
        key = self.split_heads(self.k_proj(k))
        value = self.split_heads(self.v_proj(v))
        
        outputs, weights = self.base_attention(query, key, value, k_input.attention_mask)
        
        outputs = self.merge_heads(outputs)
        outputs = self.out_proj(outputs)
        
        return outputs, weights


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(EncoderLayer, self).__init__()
        self.self = SelfAttention(config)
        self.ff = FeedForwardLayer(config)
        self.attn_add_norm = AddNormLayer(config)
        self.ff_add_norm = AddNormLayer(config)
    
    def forward(self, input: SeparatedInput):
        x = input.features
        attn_outputs, attn_weights = self.self(input)
        
        x = self.attn_add_norm(attn_outputs, x)
        
        ff_outputs = self.ff(x)
        x = self.ff_add_norm(ff_outputs, x)
        
        return EncoderOutput(last_hidden_states=x,
                             attention_weights=attn_weights.detach())


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(DecoderLayer, self).__init__()
        self.self = SelfAttention(config)
        self.cross = CrossAttention(config)
        
        self.ffl = FeedForwardLayer(config)
        
        self.self_add_norm = AddNormLayer(config)
        self.cross_add_norm = AddNormLayer(config)
        self.ffl_add_norm = AddNormLayer(config)
    
    def forward(self,
                target: SeparatedInput,
                input_context: SeparatedInput):
        x = target.features
        attn_outputs, self_weights = self.self(target)
        x = self.self_add_norm(attn_outputs, x)
        
        attn_outputs, cross_weights = self.cross(
            q_input=SeparatedInput(x, target.pos_embedding),
            k_input=input_context,
            v_input=input_context
        )
        x = self.cross_add_norm(attn_outputs, x)
        
        ffl_outputs = self.ffl(x)
        x = self.ffl_add_norm(ffl_outputs, x)
        return DecoderOutput(last_hidden_states=x,
                             attention_weights=cross_weights.detach())
