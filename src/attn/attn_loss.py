import torch
from typing import List
from attn.initno_utils import fn_get_otsu_mask, fn_clean_mask, fn_get_topk, fn_smoothing_func

def fn_augmented_compute_losss(
    self, 
    indices: List[int], 
    smooth_attentions: bool = True,
    K: int = 1,
    attention_res: int = 16,) -> torch.Tensor:
    # -----------------------------
    # cross-attention response loss
    # -----------------------------
    aggregate_cross_attention_maps = self.attention_store.aggregate_attention(
        from_where=("up", "down", "mid"), is_cross=True)
    
    # cross attention map preprocessing
    cross_attention_maps = aggregate_cross_attention_maps[:, :, 1:-1]
    cross_attention_maps = cross_attention_maps * 100
    cross_attention_maps = torch.nn.functional.softmax(cross_attention_maps, dim=-1)
    # Shift indices since we removed the first token
    indices = [index - 1 for index in indices]
    # clean_cross_attention_loss
    clean_cross_attn_loss = 0.
    # Extract the maximum values
    topk_value_list, topk_coord_list_list = [], []
    for i in indices:
        cross_attention_map_cur_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
        topk_coord_list, _ = fn_get_topk(cross_attention_map_cur_token, K=K)
        topk_value = 0
        for coord_x, coord_y in topk_coord_list: topk_value = topk_value + cross_attention_map_cur_token[coord_x, coord_y]
        topk_value = topk_value / K
        topk_value_list.append(topk_value)
        topk_coord_list_list.append(topk_coord_list)
        # -----------------------------------
        # clean cross_attention_map_cur_token
        # -----------------------------------
        clean_cross_attention_map_cur_token                     = cross_attention_map_cur_token
        clean_cross_attention_map_cur_token_mask                = fn_get_otsu_mask(clean_cross_attention_map_cur_token)
        clean_cross_attention_map_cur_token_mask                = fn_clean_mask(clean_cross_attention_map_cur_token_mask, topk_coord_list[0][0], topk_coord_list[0][1])
        clean_cross_attention_map_cur_token_foreground          = clean_cross_attention_map_cur_token * clean_cross_attention_map_cur_token_mask + (1 - clean_cross_attention_map_cur_token_mask)
        clean_cross_attention_map_cur_token_background          = clean_cross_attention_map_cur_token * (1 - clean_cross_attention_map_cur_token_mask)
        if clean_cross_attention_map_cur_token_background.max() > clean_cross_attention_map_cur_token_foreground.min():
            clean_cross_attn_loss = clean_cross_attn_loss + clean_cross_attention_map_cur_token_background.max()
        else: clean_cross_attn_loss = clean_cross_attn_loss + clean_cross_attention_map_cur_token_background.max() * 0
    cross_attn_loss_list = [max(0 * curr_max, 1.0 - curr_max) for curr_max in topk_value_list]
    cross_attn_loss = max(cross_attn_loss_list)
    # ------------------------------
    # cross attention alignment loss
    # ------------------------------
    alpha = 0.9
    if self.cross_attention_maps_cache is None: self.cross_attention_maps_cache = cross_attention_maps.detach().clone()
    else: self.cross_attention_maps_cache = self.cross_attention_maps_cache * alpha + cross_attention_maps.detach().clone() * (1 - alpha)
    cross_attn_alignment_loss = 0
    for i in indices:
        cross_attention_map_cur_token = cross_attention_maps[:, :, i]
        if smooth_attentions: cross_attention_map_cur_token = fn_smoothing_func(cross_attention_map_cur_token)
        cross_attention_map_cur_token_cache = self.cross_attention_maps_cache[:, :, i]
        if smooth_attentions: cross_attention_map_cur_token_cache = fn_smoothing_func(cross_attention_map_cur_token_cache)
        cross_attn_alignment_loss = cross_attn_alignment_loss + torch.nn.L1Loss()(cross_attention_map_cur_token, cross_attention_map_cur_token_cache)          
    # ----------------------------
    # self-attention conflict loss
    # ----------------------------
    self_attention_maps = self.attention_store.aggregate_attention(
        from_where=("up", "down", "mid"), is_cross=False)
    self_attention_map_list = []
    for topk_coord_list in topk_coord_list_list:
        self_attention_map_cur_token_list = []
        for coord_x, coord_y in topk_coord_list:
            self_attention_map_cur_token = self_attention_maps[coord_x, coord_y]
            self_attention_map_cur_token = self_attention_map_cur_token.view(attention_res, attention_res).contiguous()
            self_attention_map_cur_token_list.append(self_attention_map_cur_token)
        if len(self_attention_map_cur_token_list) > 0:
            self_attention_map_cur_token = sum(self_attention_map_cur_token_list) / len(self_attention_map_cur_token_list)
            if smooth_attentions: self_attention_map_cur_token = fn_smoothing_func(self_attention_map_cur_token)
        else:
            self_attention_map_per_token = torch.zeros_like(self_attention_maps[0, 0])
            self_attention_map_per_token = self_attention_map_per_token.view(attention_res, attention_res).contiguous()
        self_attention_map_list.append(self_attention_map_cur_token)
    self_attn_loss, number_self_attn_loss_pair = 0, 0
    number_token = len(self_attention_map_list)
    for i in range(number_token):
        for j in range(i + 1, number_token): 
            number_self_attn_loss_pair = number_self_attn_loss_pair + 1
            self_attention_map_1 = self_attention_map_list[i]
            self_attention_map_2 = self_attention_map_list[j]
            self_attention_map_min = torch.min(self_attention_map_1, self_attention_map_2) 
            self_attention_map_sum = (self_attention_map_1 + self_attention_map_2)
            cur_self_attn_loss = (self_attention_map_min.sum() / (self_attention_map_sum.sum() + 1e-6))
            self_attn_loss = self_attn_loss + cur_self_attn_loss
    if number_self_attn_loss_pair > 0: self_attn_loss = self_attn_loss / number_self_attn_loss_pair
    joint_loss = cross_attn_loss * 1. + clean_cross_attn_loss * 0.1 + cross_attn_alignment_loss * 0.1 + self_attn_loss * 1.
    return joint_loss, cross_attn_loss, clean_cross_attn_loss, self_attn_loss