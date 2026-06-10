"""
Fichier définissant le modèle de diffusion (UNet prédisant le bruit).
UNet guidé
    input   : Image bruitée + anatomie
    output  : Prédiction du bruit

Guidage par embeddings de temps et du site
"""

import math
from typing import List, Optional, Sequence, Tuple
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
 
from .blocks import (
    AttentionBlock2D,
    DownBlock2D,
    Downsample2D,
    ResidualBlock2D,
    UpBlock2D,
    Upsample2D,
    timestep_embedding,
)
 

class UNet2DConditionModel_Optimized(nn.Module):
    """
    UNet 2D 
    """
    def __init__(
            self, 
            in_channels : int=2, 
            out_channels: int = 1,
            down_block_types: Sequence[str] = ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types: Sequence[str] = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            block_out_channels: Sequence[int] = (32, 64, 256, 512),
            cross_attention_dim: int = 512,
            attention_head_dim: int = 64,
            time_embedding_dim: int = 512,
            # optimization-specific args:
            gradient_checkpointing: bool = True,
            attention_pool: bool = True,  # apply pooling before attention when configured per block
            attn_pool_kernel: int = 2,
            attn_on_resolutions: Optional[List[int]] = None,  # indices of blocks where attention is allowed (down indices)
        ):
        """
        Justification in_channels / out_channels: 
            noisy_latents  : (B, 1, ...) - volume bruité
            volume_anat_map: (B, 1, ...) - carte anatomique
            Donc -> 2 canaux en entrée, 1 en sortie (prédiction du bruit seulement)
        """
        super().__init__()

        assert len(down_block_types) == len(up_block_types) == len(block_out_channels), \
            "down_block_types, up_block_types and block_out_channels must have same length"
        
        self.block_out_channels = list(block_out_channels)
        self.cross_attention_dim = cross_attention_dim
        self.attention_head_dim = attention_head_dim
        self.time_embedding_dim = time_embedding_dim

        self.gradient_checkpointing = gradient_checkpointing
        self.attention_pool = attention_pool
        self.attn_pool_kernel = attn_pool_kernel
        self.attn_on_resolutions = attn_on_resolutions if attn_on_resolutions is not None else list(range(len(block_out_channels)))

        # initial conv
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # time embedding MLP 
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )
        
        # build down blocks
        self.down_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        prev_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            use_attn  = ("CrossAttn" in down_block_types[i] or "Attn" in down_block_types[i]) \
                        and (i in self.attn_on_resolutions)
            num_heads = max(1, out_ch // attention_head_dim)
 
            self.down_blocks.append(
                DownBlock2D(
                    in_ch               = prev_ch,
                    out_ch              = out_ch,
                    time_emb_dim        = time_embedding_dim,
                    use_attn            = use_attn,
                    num_heads           = num_heads,
                    head_dim            = attention_head_dim,
                    cross_attention_dim = cross_attention_dim if use_attn else None,
                    attn_pool           = attention_pool and use_attn,
                    pool_kernel         = attn_pool_kernel,
                )
            )
            # Pas de downsampler après le dernier niveau
            if i != len(block_out_channels) - 1:
                self.downsamplers.append(Downsample2D(out_ch))
 
            prev_ch = out_ch

        # middle (bottleneck)
        mid_ch = block_out_channels[-1]
        num_heads_mid = max(1, mid_ch // attention_head_dim)
        self.mid_block1 = ResidualBlock2D(mid_ch, mid_ch, time_emb_dim=time_embedding_dim)
        self.mid_attn   = AttentionBlock2D(
            channels            = mid_ch,
            num_heads           = num_heads_mid,
            head_dim            = attention_head_dim,
            cross_attention_dim = cross_attention_dim,
            pool_before_attn    = attention_pool,
            pool_kernel         = attn_pool_kernel,
        )
        self.mid_block2 = ResidualBlock2D(mid_ch, mid_ch, time_emb_dim=time_embedding_dim)


        # build up blocks
        self.upsamplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        rev_channels = list(reversed(block_out_channels))
        prev_ch = rev_channels[0]
        for i, out_ch in enumerate(rev_channels):
            # L'indice dans le chemin descendant correspondant
            down_idx  = len(block_out_channels) - 1 - i
            use_attn  = ("CrossAttn" in up_block_types[i] or "Attn" in up_block_types[i]) \
                        and (down_idx in self.attn_on_resolutions)
            num_heads = max(1, out_ch // attention_head_dim)
            self.up_blocks.append(
                UpBlock2D(
                    in_ch               = prev_ch + out_ch,   # concaténation skip
                    out_ch              = out_ch,
                    time_emb_dim        = time_embedding_dim,
                    use_attn            = use_attn,
                    num_heads           = num_heads,
                    head_dim            = attention_head_dim,
                    cross_attention_dim = cross_attention_dim if use_attn else None,
                    attn_pool           = attention_pool and use_attn,
                    pool_kernel         = attn_pool_kernel,
                )
            )
            # Pas d'upsampler après le dernier niveau
            if i != len(rev_channels) - 1:
                self.upsamplers.append(Upsample2D(out_ch))
 
            prev_ch = out_ch
        
        # final normalization and conv
        self.norm_out = nn.GroupNorm(8, block_out_channels[0])
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _maybe_checkpoint(self, module, *args):
        """
        Gradient checkpointing si activé ET si le module est en mode train.
        On vérifie module.training et non self.training :
        si un sous-module est gelé/mis en eval individuellement, le checkpointing
        ne lui est pas appliqué inutilement.
        """
        if self.gradient_checkpointing and module.training:
            return checkpoint(module, *args, use_reentrant=False)
        return module(*args)
        
    # forward
    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: Optional[torch.Tensor] = None):
        """
        sample: (B, in_channels, H, W) - noisy image + anatomy map
        timestep: (B,) or scalar 
        encoder_hidden_states: (B, 1, cross_attention_dim) if using cross-attention (embedding of the label from nn.Embedding)
        """
        if timestep.dim() == 0:
            timestep = timestep.view(1).expand(sample.shape[0])
        # Embedding temporel
        t_emb = timestep_embedding(timestep, self.time_embedding_dim)
        t_emb = self.time_mlp(t_emb)          # (B, time_embedding_dim=512)

        # input conv
        x = self.conv_in(sample)

        # down path
        skips = []
        for i, db in enumerate(self.down_blocks):
            for i, db in enumerate(self.down_blocks):
                x = self._maybe_checkpoint(db, x, t_emb, encoder_hidden_states)
                skips.append(x)
                if i < len(self.downsamplers):
                    x = self.downsamplers[i](x)

        # bottleneck
        x = self._maybe_checkpoint(self.mid_block1, x, t_emb)
        x = self._maybe_checkpoint(self.mid_attn, x, encoder_hidden_states)
        x = self._maybe_checkpoint(self.mid_block2, x, t_emb)

        # up path
        for i, ub in enumerate(self.up_blocks):
            skip = skips.pop()
            x = self._maybe_checkpoint(ub, x, skip, t_emb, encoder_hidden_states)
            if i < len(self.upsamplers):
                x = self.upsamplers[i](x)
                
        # output
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

