import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

from diffusers.models.attention_processor import Attention
from attn.rope import apply_rotary_emb, compute_axial_cis

def register_attention_control(unet, attention_store):
    attn_procs = {}
    cross_att_count = 0
    
    for name in unet.attn_processors.keys():
        if name.startswith("mid_block"):
            place_in_unet = "mid"
        elif name.startswith("up_blocks"):
            place_in_unet = "up"
        elif name.startswith("down_blocks"):
            place_in_unet = "down"
        else:
            continue

        cross_att_count += 1
        attn_procs[name] = PositionalEmbeddedAttnProcessor(attnstore=attention_store, place_in_unet=place_in_unet)

    unet.set_attn_processor(attn_procs)
    attention_store.num_att_layers = cross_att_count

# TODO: 感觉需要一个能加超分辨率的版本
class AttentionStore:    
    
    def __init__(self, attn_res, enable_attention_tracking=True):
        """
        Initialize an empty AttentionStore :param step_index: used to visualize only a specific step in the diffusion
        process
        """
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.curr_step_index = 0
        self.attn_res = attn_res
        self.enable_attention_tracking = enable_attention_tracking
        
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if self.cur_att_layer >= 0:
            if attn.shape[1] == self.attn_res ** 2:
                self.step_store[key].append(attn)

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.between_steps()

    def between_steps(self):
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def aggregate_attention(self, from_where: List[str], is_cross: bool = True) -> torch.Tensor:
        """Aggregates the attention across the different layers and heads at the specified resolution."""
        out = []
        attention_maps = self.get_average_attention()
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                cross_maps = item.reshape(-1, self.attn_res, self.attn_res, item.shape[-1])
                out.append(cross_maps)

        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    def reset(self):
        self.cur_att_layer = 0
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class AttendExciteAttnProcessor:
    def __init__(self, attnstore, place_in_unet):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.attnstore.enable_attention_tracking:    
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class PositionalEmbeddedAttnProcessor:
    def __init__(self, attnstore, place_in_unet, theta=100.0):
        super().__init__()
        self.attnstore = attnstore
        self.place_in_unet = place_in_unet
        self.theta = theta
        self.freqs_cis = None

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        
        # 计算并应用2D RoPE
        resolution = int(math.sqrt(sequence_length))
        if self.freqs_cis is None or self.freqs_cis.shape[0] != sequence_length:
            self.freqs_cis = compute_axial_cis(dim=hidden_states.shape[-1], end_x=resolution, end_y=resolution, theta=self.theta)
            self.freqs_cis = self.freqs_cis.to(hidden_states.device)
        
        hidden_states = apply_rotary_emb(hidden_states, self.freqs_cis)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if self.attnstore.enable_attention_tracking:    
            self.attnstore(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states