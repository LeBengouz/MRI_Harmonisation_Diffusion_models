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

import albumentations as A

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
 
# Non utilisé car produit cerveau ne pouvant pas apparaitre réellement -> data augmentation inutile
def _random_vflip(img, p = 0.5):
    """Flip vertical aléatoire. img: (1, H, W) - tensor"""
    if random.random() < p:
        return img.flip(-2)
    return img
 
# non utilisé -> à delete
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
        augment_spatial = True, augment_elastic = True, elastic_alpha = 100.0, elastic_sigma = 6.0):

        self.slice_dir = Path(slice_dir)
        self.train = train
        self.anatomy_mode = anatomy_mode
        self.anatomy_csv_path = anatomy_csv_path
        self.augment_spatial = augment_spatial

        self.augment_elastic = augment_elastic

        # Transform définie ici puis utilisé à chaque __getitem__
        # on gère la proba dans __getitem__
        # padding constant à 0 (= fond noir) ce qui correspond aux données normalement 
        self.elastic_transform = A.Compose(
            [A.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, p=1.0, border_mode=0)], 
            additional_targets={"anat_map": "image"}
        ) if augment_elastic else None

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

        # augmentations si train (remplace torchio) -> vieux fonctionnement garantissant pas coherence anat map
        # if self.train and self.augment_spatial:
        #    img = _random_hflip(img, p=0.5)
            # img = _random_vflip(img, p=0.3) 
            # img = _random_rotate90(img, p=0.3) # supp car rotation 90 degrés pose problème de dimensions
        if self.train and self.augment_spatial:
            # On tire le dé UNE seule fois et on applique la même décision aux deux tenseurs plus tard
            do_hflip = random.random() < 0.5
        else:
            do_hflip = False


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
            ).squeeze(0)  # (B=1, 1, H, W) -> (1, H, W)Building Trustworthy AI Agents
        
        if do_hflip:
            img_augmented = img_augmented.flip(-1)
            anat_map = anat_map.flip(-1)

        # Elastic deformation = la même sur la raw image et son anatomy map
        if self.train and self.elastic_transform is not None and random.random() < 0.5:
            # conversion numpy
            img_np = img_augmented.squeeze(0).numpy()   # (H, W))
            anat_np = anat_map.squeeze(0).numpy()           

            result = self.elastic_transform(image=img_np, anat_map=anat_np) # la même déformation élastique est appliquée à image et à anat_map

            img_augmented = torch.from_numpy(result["image"]).unsqueeze(0)   # (1, H, W)
            anat_map = torch.from_numpy(result["anat_map"]).unsqueeze(0)

        return img_augmented, anat_map, label
 
def build_label_mapping(slice_dir, gamma_values: Optional[List[float]] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Scanne `slice_dir` pour récupérer tous les `ds_name` (= stems des .npy),
    puis construit le produit cartésien avec les valeurs gamma.

    La fonction renvoie deux dictionnaires pour croiser l'indice avec le nom et inversement.
    L'indice 0 est réservé à l'embedding «non-conditionné» (classifier-free guidance).
 
    Args:
        slice_dir:    Répertoire contenant les .npy.
        gamma_values: Valeurs de log_gamma. Si None, on utilise les valeurs par défaut.
    """
    gamma_values = gamma_values or SliceDataset.GAMMA_VALUES_DEFAULT
 
    slice_dir = Path(slice_dir)
    ds_names = sorted({p.stem.rsplit("_", 1)[0] for p in slice_dir.glob("*.npy")})
    assert len(ds_names) > 0, f"Pas de fichier .npy dans {slice_dir}"
 
    all_labels = set()
    for name in ds_names:
        for g in gamma_values:
            label = f"{name}_gamma_{g}"
            all_labels.add(label)

    all_labels = sorted(all_labels)
 
    ds2id: Dict[str, int] = {label: idx + 1 for idx, label in enumerate(all_labels)}
    id2ds: Dict[int, str] = {idx: label for label, idx in ds2id.items()}

    n_classes = len(all_labels)
    print(f"Nombre de classes: {n_classes}")
 
    return ds2id, id2ds


def build_label_mapping_from_csv(anatomy_csv_path, gamma_values: Optional[List[float]] = None,) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Alternative plus rapide de build_label_mapping utilisant le(s) CSV de l'anatomy encoder.
    (colonnes : name, raw_path, encoded_path)
 
    Utile quand anatomy_mode="encoded" . Le mapping recouvrira donc que les sites concernés et ira plus vite en lecture.
    
    anatomy_csv_path: chemin vers un CSV (str), OU liste de chemins
    """
    gamma_values = gamma_values or SliceDataset.GAMMA_VALUES_DEFAULT

    # transforme en liste pour traiter str unique et liste de chemins de la me manière
    csv_paths = [anatomy_csv_path] if isinstance(anatomy_csv_path, str) else list(anatomy_csv_path)
    assert len(csv_paths) > 0, "Au moins un chemin de CSV doit être fourni"
 
    ds_names_all = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        assert "name" in df.columns, f"Le CSV doit contenir une colonne 'name' ({csv_path})"
        ds_names = df["name"].astype(str).str.rsplit("_", n=1).str[0].unique().tolist()
        ds_names_all.update(ds_names)

    ds_names_all = sorted(ds_names_all)
    assert len(ds_names_all) > 0, f"Aucun nom (name) valide trouvé dans {csv_paths}"

    all_labels = set()

    for name in ds_names_all:
        for g in gamma_values:
            label = f"{name}_gamma_{g}"
            all_labels.add(label)

    all_labels = sorted(all_labels)

 
    ds2id: Dict[str, int] = {label: idx + 1 for idx, label in enumerate(all_labels)}
    id2ds: Dict[int, str] = {idx: label for label, idx in ds2id.items()}

    n_classes = len(all_labels)
    print(f"Nombre de classes: {n_classes}")
 
    return ds2id, id2ds


def collate_fn_with_label_ids(
    batch: List[Tuple[torch.Tensor, torch.Tensor, str]],
    ds2id: Dict[str, int],
    p_uncond: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function pour assembler une liste de données en batch. Besoin car on ajoute des opérations particulières :
    - stack : assemble les tenseurs individuels en batch
    - Conversion string : traduit chaque label string en index entier via ds2id.
    -  Masquage CFG = remplace certains label_ids par 0 avec probabilité p_uncond. (Classifier-Free-Guidance)
    
    Args:
        batch:    Liste de (slice_z, anat_map, label_str) venant du Dataset.
        ds2id:    Mapping label_str -> int issu de build_label_mapping.
        p_uncond: Probabilité de masquage pour le classifier-free guidance.
 
    Returns:
        slices    : (B, 1, H, W)
        anat_maps : (B, 1, H, W)
        label_ids : (B,) LongTensor - 0 = non-conditionné
    """
    slices, anat_maps, labels = zip(*batch)
 
    slices = torch.stack(slices, dim=0)       # (B, 1, H, W)
    anat_maps = torch.stack(anat_maps, dim=0) # (B, 1, H, W)
 
    ids = torch.tensor([ds2id[l] for l in labels], dtype=torch.long)
 
    # Masquage classifier-free guidance
    mask = torch.rand(ids.shape) < p_uncond
    ids[mask] = 0
 
    return slices, anat_maps, ids



if __name__ == "__main__":
    from pathlib import Path
    from torch.utils.data import DataLoader

    SLICE_DIR = "/NAS/coolio/benolive/Diffusion_beta_encoder/data/brain_slices/train/raw"
    ANATOMY_CSV_PATH = "/NAS/coolio/benolive/Diffusion_beta_encoder/data/csv_files/diffusion_data/data_by_name_train.csv"
    BATCH_SIZE = 10

    slice_dir = Path(SLICE_DIR)
    anatomy_csv_path = Path(ANATOMY_CSV_PATH)

    assert slice_dir.exists()
    assert anatomy_csv_path.exists()

    npy_files = sorted(slice_dir.glob("*.npy"))
    assert len(npy_files) > 0

    print("=== Tests rapides SliceDataset / mappings / DataLoader ===")


    # Test 1 - mapping depuis le dossier
    ds2id_dir, id2ds_dir = build_label_mapping(slice_dir)

    sites_dir = {
        p.stem.rsplit("_", 1)[0]
        for p in npy_files
    }

    assert len(ds2id_dir) == len(sites_dir) * len(SliceDataset.GAMMA_VALUES_DEFAULT)
    assert len(id2ds_dir) == len(ds2id_dir)

    print("1] build_label_mapping OK")



    # Test 2 - mapping depuis le CSV
    ds2id_csv, id2ds_csv = build_label_mapping_from_csv(anatomy_csv_path)

    df = pd.read_csv(anatomy_csv_path)

    sites_csv = set(
        df["name"]
        .dropna()
        .astype(str)
        .str.rsplit("_", n=1)
        .str[0]
        .tolist()
    )

    assert len(ds2id_csv) == len(sites_csv) * len(SliceDataset.GAMMA_VALUES_DEFAULT)
    assert len(id2ds_csv) == len(ds2id_csv)

    print("2] build_label_mapping_from_csv OK")



    # Test 3 - Dataset param naive
    naive_ds = SliceDataset(slice_dir=slice_dir, train=True, anatomy_mode="naive", augment_spatial=False)

    img, anat_map, label = naive_ds[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(anat_map, torch.Tensor)
    assert label in ds2id_dir

    print("3] SliceDataset naif OK")


    # Test 4 - DataLoader naif
    naive_loader = DataLoader(
        naive_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    imgs, anat_maps, labels = next(iter(naive_loader))

    assert imgs.ndim == 4
    assert anat_maps.ndim == 4
    assert labels[0] in ds2id_dir

    print("4] DataLoader naif OK")


    # Test 5 - Dataset encoded
    encoded_ds = SliceDataset(slice_dir=slice_dir, train=True, anatomy_mode="encoded", anatomy_csv_path=anatomy_csv_path, augment_spatial=False)

    img, anat_map, label = encoded_ds[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(anat_map, torch.Tensor)
    assert label in ds2id_csv

    print("5] SliceDataset encoded OK")


    # Test 6 - DataLoader encoded
    encoded_loader = DataLoader(
        encoded_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    imgs, anat_maps, labels = next(iter(encoded_loader))

    assert imgs.ndim == 4
    assert anat_maps.ndim == 4
    assert labels[0] in ds2id_csv

    print("6] DataLoader encoded OK")


    print()
    print("Tous les tests sont passés.")