import numpy as np
from scipy.interpolate import interp2d


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    生成2D正弦余弦位置嵌入。

    参数:
    embed_dim: int, 嵌入维度
    grid_size: int, 网格的高度和宽度（假设是正方形）
    cls_token: bool, 是否包含分类令牌
    extra_tokens: int, 额外令牌的数量

    返回:
    pos_embed: numpy数组
               如果cls_token=False: 形状为 [grid_size*grid_size, embed_dim]
               如果cls_token=True 且 extra_tokens>0: 形状为 [extra_tokens+grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 这里w先行
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    从给定的2D网格生成正弦余弦位置嵌入。

    参数:
    embed_dim: int, 嵌入维度
    grid: numpy数组, 形状为 [2, 1, grid_size, grid_size]

    返回:
    emb: numpy数组, 形状为 [grid_size*grid_size, embed_dim]
    """
    assert embed_dim % 2 == 0

    # 使用一半的维度来编码grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    为1D位置生成正弦余弦嵌入。

    参数:
    embed_dim: int, 每个位置的输出维度
    pos: numpy数组, 要编码的位置列表，大小为 (M,)

    返回:
    emb: numpy数组, 形状为 (M, embed_dim)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), 外积

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_multi_resolution_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0, max_grid_size = 0):
    """
    生成2D正弦余弦位置嵌入，并根据需要进行下采样。

    参数:
    embed_dim: int, 嵌入维度
    grid_size: int, 当前网格的高度和宽度（假设是正方形）
    cls_token: bool, 是否包含分类令牌
    extra_tokens: int, 额外令牌的数量
    max_grid_size: int, 最大网格大小，用于生成基础位置编码

    返回:
    pos_embed: numpy数组
    """
    # 为最大网格大小生成位置编码
    grid_h = np.arange(max_grid_size, dtype=np.float32)
    grid_w = np.arange(max_grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # 这里w先行
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, max_grid_size, max_grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = pos_embed.reshape(max_grid_size, max_grid_size, embed_dim)

    # 如果当前网格大小小于最大网格大小，进行下采样
    if grid_size < max_grid_size:
        x = np.linspace(0, max_grid_size-1, grid_size)
        y = np.linspace(0, max_grid_size-1, grid_size)
        
        # 对每个嵌入维度进行插值
        pos_embed_interp = np.zeros((grid_size, grid_size, embed_dim))
        for i in range(embed_dim):
            interp_func = interp2d(grid_h, grid_w, pos_embed[:,:,i], kind='linear')
            pos_embed_interp[:,:,i] = interp_func(x, y)
        
        pos_embed = pos_embed_interp.reshape(grid_size*grid_size, embed_dim)
    else:
        pos_embed = pos_embed.reshape(max_grid_size*max_grid_size, embed_dim)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    
    return pos_embed