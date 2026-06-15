'''
SliceDataset (adapté de VolumeDataset) + label 

Adapté depuis VolumeDataset (3D, fichiers .pt + torchio) remplacé par :
    - Slices 2D (.npy), shape (H, W)
    - Augmentations 2D à la place de torchio
    - Mapping label : {ds_name}_gamma_{val}  ->  int 
    - Support des deux modes d'anatomie :
        - "naive"   : make_structural_anatomy_map_2d (calculée ici)
        - "encoded" : load_beta_encoded_anatomy (pré-calculée par le beta_encoder (cf. precompute), chargé via un CSV à l'aide du nom)
'''

import os
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
 
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# anatomy_map.py se trouve dans le dossier models/ à la racine du projet
try:
    from models.anatomy_map import (
        load_beta_encoded_anatomy,
        make_structural_anatomy_map_2d,
    )
except ImportError as e:
    raise ImportError(
        "Impossible d'importer models.anatomy_map "
    ) from e


# === Adaptations des transformations de torchio à la 2D

def _random_hflip(img, p = 0.5):
    """Flip horizontal aléatoire. img: (1, H, W) - tensor"""
    if random.random() < p:
        return img.flip(-1)
    return img
 
 
def _random_vflip(img, p = 0.5):
    """Flip vertical aléatoire. img: (1, H, W) - tensor"""
    if random.random() < p:
        return img.flip(-2)
    return img
 
 
def _random_rotate90(img, p = 0.5):
    """Rotation 90 degrés aléatoire. img: (1, H, W) - tensor"""
    if random.random() < p:
        k = random.choice([1, 2, 3])
        return torch.rot90(img, k=k, dims=[-2, -1])
    return img
 
def _random_gamma(img, log_gamma_range = (-0.4, 0.4)) :
    """
    Augmentation gamma : x -> sign(x) * |x|^exp(log_gamma).
    (valeurs négatives possibles).
    Retourne (img_augmentée, log_gamma_val).
    """
    log_gamma_val = random.uniform(*log_gamma_range)
    gamma = float(np.exp(log_gamma_val))

    sign = img.sign()
    img_gamma = sign * (img.abs() + 1e-8).pow(gamma)
    return img_gamma, log_gamma_val
 
 
def _zscore_normalize(img: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalisation z-score par image. img: (1, H, W)"""
    return (img - img.mean()) / (img.std(unbiased=False) + eps)


# class dataset

class SliceDataset(Dataset):
    """
    Dataset de slices 2D extrait d'un dossier au format .npy

    Chaque fichier .npy doit :
        - avoir un nom unique (utilisé comme `ds_name` dans le label) : sous forme [site acquisition]_[numero].npy -> ds_name = [site acquisition]
        - contenir un tableau 2D numpy float de shape (H, W)
    """
    