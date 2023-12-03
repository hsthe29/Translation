import torch
from torch import nn
from torch.nn import functional as tf
from ..dataclasses_utils import EncoderOutput, DecoderOutput
from .config import TransformerConfig
from ..utils import ACT2FN

import math


def positional_encoding(length, depth):
    pe = torch.zeros((length, depth))
    
    depth = depth // 2
    
    positions = torch.arange(length, dtype=torch.float32)[:, None]  # (seq, 1)
    depths = torch.arange(depth, dtype=torch.float32)[None, :] / depth  # (1, mapping_depth)
    
    angle_rates = 1 / (10000 ** depths)  # (1, mapping_depth) | n^(2i/d)
    angle_rads = positions * angle_rates  # (pos, mapping_depth)
    
    sin_values = torch.sin(angle_rads)
    cos_values = torch.cos(angle_rads)
    
    pe[:, ::2] = sin_values
    pe[:, 1::2] = cos_values
    
    return pe[None, :, :]


class TransformerEmbeddings(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerEmbeddings, self).__init__()
        self.ids_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        self.pos_embedding = nn.Embedding(config.max_position_embeddings,
                                          config.hidden_size)
        
        self.dropout = nn.Dropout(config.embed_dropout_prob)
    
    def forward(self, ids):
        ids_embed = self.ids_embedding(ids)
        pos_embed = self.pos_embedding(torch.arange(ids.shape[-1]).unsqueeze(0).to(ids.device))
        
        ids_embed = self.dropout(ids_embed)
        
        return ids_embed, pos_embed


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


class TransformerSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerSelfAttention, self).__init__()
        
        self.num_attn_heads = config.num_attn_heads
        
        assert config.hidden_size % self.num_attn_heads == 0
        
        self.scaling = math.sqrt(config.hidden_size // self.num_attn_heads)
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.pos_embed_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attn_dropout_prob)
    
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
                q: torch.FloatTensor,
                k: torch.FloatTensor,
                v: torch.FloatTensor,
                pos_embed: torch.FloatTensor,
                mask: torch.FloatTensor = None):
        key = self.split_heads(self.k_proj(k))
        value = self.split_heads(self.v_proj(v))
        query = self.split_heads(self.q_proj(q))
        
        pos = self.split_heads(self.pos_embed_proj(pos_embed))
        
        c2c_scores = torch.matmul(query, key.transpose(-1, -2))
        c2p_scores = torch.matmul(query, pos.transpose(-1, -2))
        p2c_scores = torch.matmul(pos, key.transpose(-1, -2))

        scores = (c2c_scores + c2p_scores + p2c_scores) / self.scaling
        
        if mask is not None:
            scores = scores + mask
        
        # Normalize the attention scores to probabilities.
        weights = tf.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        outputs = torch.matmul(weights, value)
        outputs = self.merge_heads(outputs)
        outputs = self.out_proj(outputs)
        
        return outputs, weights


class TransformerCrossAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerCrossAttention, self).__init__()
        
        self.num_attn_heads = config.num_attn_heads
        
        assert config.hidden_size % self.num_attn_heads == 0
        
        self.scaling = math.sqrt(config.hidden_size // self.num_attn_heads)
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attn_dropout_prob)
    
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
                q: torch.FloatTensor,
                k: torch.FloatTensor,
                v: torch.FloatTensor,
                mask: torch.FloatTensor = None):
        key = self.split_heads(self.k_proj(k))
        value = self.split_heads(self.v_proj(v))
        query = self.split_heads(self.q_proj(q))
        
        c2c_scores = torch.matmul(query, key.transpose(-1, -2))
        
        scores = c2c_scores / self.scaling
        
        if mask is not None:
            scores = scores + mask
        
        # Normalize the attention scores to probabilities.
        weights = tf.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        outputs = torch.matmul(weights, value)
        outputs = self.merge_heads(outputs)
        outputs = self.out_proj(outputs)
        
        return outputs, weights


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(EncoderLayer, self).__init__()
        self.self = TransformerSelfAttention(config)
        self.ff = FeedForwardLayer(config)
        self.attn_add_norm = AddNormLayer(config)
        self.ff_add_norm = AddNormLayer(config)
    
    def forward(self,
                input_features: torch.FloatTensor,
                pos_embeddings: torch.FloatTensor,
                attention_mask: torch.FloatTensor = None):
        attn_outputs, attn_weights = self.self(
            q=input_features,
            k=input_features,
            v=input_features,
            pos_embed=pos_embeddings,
            mask=attention_mask
        )
        x = self.attn_add_norm(attn_outputs, input_features)
        
        ff_outputs = self.ff(x)
        x = self.ff_add_norm(ff_outputs, x)
        
        return EncoderOutput(last_hidden_states=x,
                             attention_weight=attn_weights)


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(DecoderLayer, self).__init__()
        self.self = TransformerSelfAttention(config)
        self.cross = TransformerCrossAttention(config)
        
        self.ff = FeedForwardLayer(config)
        
        self.self_attn_add_norm = AddNormLayer(config)
        self.cross_attn_add_norm = AddNormLayer(config)
        self.ff_add_norm = AddNormLayer(config)
    
    def forward(self,
                target_in_features: torch.FloatTensor,
                input_features: torch.FloatTensor,
                pos_embeddings: torch.FloatTensor,
                target_in_mask: torch.Tensor,
                input_mask: torch.Tensor):
        attn_outputs, self_weights = self.self(
            q=target_in_features,
            k=target_in_features,
            v=target_in_features,
            pos_embed=pos_embeddings,
            mask=target_in_mask
        )
        x = self.self_attn_add_norm(attn_outputs, target_in_features)
        
        attn_outputs, cross_weights = self.cross(
            q=x,
            k=input_features,
            v=input_features,
            mask=input_mask
        )
        x = self.cross_attn_add_norm(attn_outputs, x)
        
        ff_outputs = self.ff(x)
        x = self.ff_add_norm(ff_outputs, x)
        return DecoderOutput(last_hidden_states=x,
                             self_attention_weight=self_weights,
                             cross_attention_weight=cross_weights)
