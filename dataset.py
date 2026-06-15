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
    GAMMA_VALUES_DEFAULT = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]

    def __init__(self, slice_dir, train = True, anatomy_mode: Literal["naive", "encoded"] = "naive", anatomy_csv_path = None, gamma_values = None,
        augment_spatial = True):

        self.slice_dir = Path(slice_dir)
        self.train = train
        self.anatomy_mode = anatomy_mode
        self.anatomy_csv_path = anatomy_csv_path
        self.augment_spatial = augment_spatial

        assert self.slice_dir.exists(), f"Dossier des slices introuvable : {self.slice_dir}"
 
        self.npy_files: List[Path] = sorted(self.slice_dir.glob("*.npy"))
        assert len(self.npy_files) > 0, f"Aucun fichier .npy trouvé dans : {self.slice_dir}"
 
        # Valeurs possibles pour l'augmentation gamma
        self.gamma_values = gamma_values if gamma_values is not None else self.GAMMA_VALUES_DEFAULT

 
        if anatomy_mode == "encoded":
            assert anatomy_csv_path is not None, \
                "anatomy_csv_path requis quand anatomy_mode='encoded'"
            assert Path(anatomy_csv_path).exists(), \
                f"CSV introuvable : {anatomy_csv_path}"

        def __len__(self):
            return len(self.npy_files)

        def __getitem__(self, idx):
            """
            L'anatomy map est chargée et retournée ici pour charger avec le dataloader
            """
            npy_path = self.npy_files[idx]
            slice_name = npy_path.stem
            ds_name = npy_path.stem.rsplit("_", 1)[0]  

            # conversion tensor (1, H, W)
            arr = np.load(npy_path).astype(np.float32)
            assert arr.ndim == 2, f"Attendu (H,W), reçu {arr.shape} pour {npy_path}"
            img = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)

            # augmentations si train (remplace torchio)
            if self.train and self.augment_spatial:
                img = _random_hflip(img, p=0.5)
                img = _random_vflip(img, p=0.3)
                img = _random_rotate90(img, p=0.3)


            # augmentation gamma + label ?
            if self.train:
                log_gamma_val = random.choice(self.gamma_values)
                img_augmented, _ = _random_gamma(img, log_gamma_range=(log_gamma_val, log_gamma_val))
                label = f"{ds_name}_gamma_{log_gamma_val}"
            else:
                img_augmented = img
                log_gamma_val = 0.0
                label = f"{ds_name}_gamma_{log_gamma_val}"

            # normaliser
            img_augmented = _zscore_normalize(img_augmented)


            # anatomy map
            if self.anatomy_mode == "naive":
                anat_map = make_structural_anatomy_map_2d(
                    img.unsqueeze(0),  # (1, 1, H, W)
                ).squeeze(0)  # -> (1, H, W)
    
            else:  # encoded
                # load_beta_encoded_anatomy attend une liste de noms
                anat_map = load_beta_encoded_anatomy(
                    csv_path=self.anatomy_csv_path,
                    slice_names=[slice_name],
                ).squeeze(0)  # (B=1, 1, H, W) -> (1, H, W)
    
            return img_augmented, anat_map, label
 

