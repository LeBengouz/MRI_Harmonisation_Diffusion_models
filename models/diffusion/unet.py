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
 

