import torch
import torch.nn as nn
import math

### Copied From https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py

def init_t_xy(end_x: int, end_y: int, base_res: int = 64):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    scale = base_res / end_x
    t_x = (t % end_x).float() * scale
    t_y = torch.div(t, end_x, rounding_mode='floor').float() * scale
    return t_x, t_y

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0, base_res: int = 64):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y, base_res)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x_)
    x_out = torch.view_as_real(x_ * freqs_cis).flatten(2)
    return x_out.type_as(x)

if __name__ == "__main__":
    # 示例用法
    dim = 64
    end_x, end_y = 16, 16
    
    # 计算轴向的复数旋转
    axial_cis = compute_axial_cis(dim, end_x, end_y)
    print("轴向复数旋转形状:", axial_cis.shape)
    
    # 应用旋转位置编码
    xq = torch.randn(1, end_x * end_y, dim)
    xk = torch.randn(1, end_x * end_y, dim)
    xq_rot, xk_rot = apply_rotary_emb(xq, xk, axial_cis)
    print("旋转后的查询和键形状:", xq_rot.shape, xk_rot.shape)