'''
Fichier inspiré de la section "Basic blocks in 3D" du notebook adapté à la 2D. 
Regroupe tous les outils utile pour la construction du UNet complet.

Pas besoin de découpager par chunk dans cette solution car 2D.

Contient :
  - timestep_embedding        : encodage du pas de temps
  - ResidualBlock2D           : bloc résiduel conditionné sur le timestep
  - MultiHeadAttention2D      : attention multi-têtes (self ou cross)
  - AttentionBlock2D          : applique la MHA pour feature maps 2D
  - Downsample2D / Upsample2D : convolutions -> changement de résolution
  - DownBlock2D / UpBlock2D   : blocs encodeur/décodeur du UNet
'''


 
import math
from typing import Optional, Sequence
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    timesteps   : entiers (B, )
    dim         : dimension de sortie
    max_period  : période maximale des sinusoïdes
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb  # (B, dim)
 
 

 
class ResidualBlock2D(nn.Module):
 
    def __init__(self, in_ch, out_ch, time_emb_dim: Optional[int] = None, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
 
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None
 
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
            if time_emb_dim is not None else None
        )
 
    def forward(self, x, t_emb= None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
 
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1) #2 unsqueeze pour 2D, et 3 si 3D : h = h + t
 
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
 
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
 
        return x + h
 

 
class MultiHeadAttention2D(nn.Module):
    """
    Attention multi-têtes pour tokens 2D.
    Pas de Chunking car données plus petites qu'en 3D.

    Si context (forward) est None : self-attention.
    Si context (forward) est fourni : cross-attention (Q depuis x, K/V depuis context).

    De plus, on utilise uniquement F.scaled_dot_product_attention (pas le choix comme version 3D)
    """ 
 
    def __init__(self, dim, num_heads, head_dim, cross_dim: Optional[int] = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.inner_dim = num_heads * head_dim
 
        kv_dim = cross_dim if cross_dim is not None else dim
        self.to_q   = nn.Linear(dim,     self.inner_dim, bias=False)
        self.to_k   = nn.Linear(kv_dim,  self.inner_dim, bias=False)
        self.to_v   = nn.Linear(kv_dim,  self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
 
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x       : (B, N, dim)
        context : (B, M, cross_dim) ou None
        returns : (B, N, dim)
        """
        b, n, _ = x.shape
        ctx = x if context is None else context
 
        q = self.to_q(x).view(b, n,          self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(ctx).view(b, ctx.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(ctx).view(b, ctx.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
 
        # Flash attention si disponible, sinon fallback standard
        bh = b * self.num_heads
        out = F.scaled_dot_product_attention(
            q.contiguous().view(bh, n,          self.head_dim),
            k.contiguous().view(bh, ctx.shape[1], self.head_dim),
            v.contiguous().view(bh, ctx.shape[1], self.head_dim),
            dropout_p=0.0,
            is_causal=False,
        ).view(b, self.num_heads, n, self.head_dim)
 
        out = out.transpose(1, 2).contiguous().view(b, n, self.inner_dim)
        return self.to_out(out)
 

 
class AttentionBlock2D(nn.Module):
    """
    Applique MultiHeadAttention2D sur une feature map (B, C, H, W).
 
    pool_before_attn : average pool avant l'attention puis interpolation de retour.
                       Réduit N = H*W, économise de la mémoire sur les grandes résolutions.
    """
 
    def __init__(
        self,
        channels: int,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: Optional[int] = None,
        pool_before_attn: bool = False,
        pool_kernel: int = 2,
    ):
        super().__init__()
        self.norm     = nn.GroupNorm(8, channels)
        self.proj_in  = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.mha = MultiHeadAttention2D(
            dim=channels,
            num_heads=num_heads,
            head_dim=head_dim,
            cross_dim=cross_attention_dim,
        )
        self.pool_before_attn = pool_before_attn
        self.pool_kernel      = pool_kernel
 
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x       : (B, C, H, W)
        context : (B, M, cross_dim) ou None
 
        Returns : (B, C, H, W)
        """
        b, c, h, w = x.shape
        h_in = self.norm(x)           # (B, C, H, W)
        h_in = F.silu(h_in)           # (B, C, H, W)
        h_in = self.proj_in(h_in)     # (B, C, H, W)

 
        if self.pool_before_attn and (h > 1 or w > 1):
            h_pooled = F.avg_pool2d(h_in, kernel_size=self.pool_kernel, stride=self.pool_kernel,padding=0)
            ph, pw   = h_pooled.shape[2:]
            h_flat   = h_pooled.view(b, c, ph * pw).permute(0, 2, 1)    # (B, pH*pW, C)
            attn_out = self.mha(h_flat, context)                        # (B, pH*pW, C)
            attn_out = attn_out.permute(0, 2, 1).view(b, c, ph, pw)     # (B, C, pH, pW)
            # upsample back to original spatial dims
            attn_out = F.interpolate(attn_out, size=(h, w), mode="bilinear", align_corners=False) # (B, C, pH, pW)
        else:
            h_flat   = h_in.view(b, c, h * w).permute(0, 2, 1)          # (B, H*W, C)
            attn_out = self.mha(h_flat, context)
            attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)       # (B, C, H, W)
 
        return x + self.proj_out(attn_out)                              # (B, C, H, W)
 

 
 
class Downsample2D(nn.Module):
    """Réduction de résolution *2 par convolution stride=2."""
 
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)
 
 
class Upsample2D(nn.Module):
    """Augmentation de résolution *2 par convolution transposée."""
 
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)
 


class DownBlock2D(nn.Module):
    def __init__(
        self, in_ch, out_ch, time_emb_dim = None, use_attn = False, num_heads = 1, head_dim = 32,
        cross_attention_dim = None, attn_pool = False, pool_kernel = 2
    ):
        super().__init__()
        self.res1 = ResidualBlock2D(in_ch,  out_ch, time_emb_dim)
        self.attn = (
            AttentionBlock2D(out_ch, num_heads, head_dim, cross_attention_dim=cross_attention_dim,
                            pool_before_attn=attn_pool, pool_kernel=pool_kernel,
            )
            if use_attn else None
        )
        self.res2 = ResidualBlock2D(out_ch, out_ch, time_emb_dim)
 
    def forward(self, x, t_emb = None, context = None):
        x = self.res1(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        x = self.res2(x, t_emb)
        return x
 
 
class UpBlock2D(nn.Module):
    def __init__(
        self, in_ch, out_ch, time_emb_dim = None, use_attn = False, num_heads = 1, head_dim = 32,
        cross_attention_dim = None, attn_pool = False, pool_kernel = 2,
    ):
        super().__init__()
        self.res1 = ResidualBlock2D(in_ch,  out_ch, time_emb_dim)
        self.attn = AttentionBlock2D(out_ch, num_heads, head_dim, cross_attention_dim=cross_attention_dim,
                pool_before_attn=attn_pool, pool_kernel=pool_kernel) if use_attn else None
        self.res2 = ResidualBlock2D(out_ch, out_ch, time_emb_dim)
 
    def forward(self, x, skip, t_emb = None, context = None):
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        if self.attn is not None:
            x = self.attn(x, context)
        x = self.res2(x, t_emb)
        return x
