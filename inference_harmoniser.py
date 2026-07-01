"""
inference_harmoniser.py

Harmonisation de site : reconstruit toutes les slices d'un dossier comme si elles provenaient d'un site cible (sur lequel le modèle a été entraîné).

Usage (via main_diffusion_2d.py, MODE = "inference") :
    - Choisir le checkpoint (contient model + embedder + ds2id)
    - Choisir le site cible dans la config
    - Les slices harmonisées sont sauvegardées en .npy dans output_dir
"""

import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from accelerate import Accelerator
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import re
from collections import defaultdict


from dataset import SliceDataset, collate_fn_with_label_ids
from models.diffusion.unet import UNet2DConditionModel_Optimized
from outils.checkpoints import load_checkpoint_for_eval
from train_diffusion import build_model, ddim_inference
from outils.checkpoints import load_checkpoint_for_eval, read_checkpoint_metadata

# padding assurant que la taille des données corresponde
def pad_to_multiple(x, multiple=16):
    h, w  = x.shape[-2], x.shape[-1]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h)), h, w


def inference_harmoniser(cfg, checkpoint_path, output_dir):
    """
    Harmonise toutes les slices de cfg["test_dir"] vers le site cible
    défini par cfg["target_site"] et cfg["target_gamma"].

    Args:
        cfg:             dict chargé depuis configs/inference_exp1.json
        checkpoint_path: chemin vers le checkpoint (doit contenir ds2id)
        output_dir:      dossier de sortie pour les slices harmonisées (.npy)
    """
    print(f"[inference_harmoniser] démarrage depuis {checkpoint_path}")

    os.makedirs(output_dir, exist_ok=True)

    accelerator = Accelerator(mixed_precision="fp16")

    # Chargement du modèle avec la bonne architecture
    model_diffusion = build_model(cfg)

    # Embedder temporaire - sa taille sera corrigée après lecture du ds2id
    # On crée avec n=1 puis on remplace après load
    embedder_placeholder = nn.Embedding(1, cfg["cross_attention_dim"])

    model_diffusion, embedder_placeholder = accelerator.prepare(
        model_diffusion, embedder_placeholder
    )

    epoch_loaded, step_loaded, ds2id = read_checkpoint_metadata(checkpoint_path, accelerator)

    n_classes = len(ds2id)
    print(f"[inference_harmoniser] {n_classes} classes (epoch={epoch_loaded})")

    embedder = nn.Embedding(n_classes + 1, cfg["cross_attention_dim"])
    model_diffusion, embedder = accelerator.prepare(model_diffusion, embedder)

    # Un seul chargement des poids, embedder à la bonne taille
    load_checkpoint_for_eval(checkpoint_path, model_diffusion, embedder, accelerator)

    # Vérifier que le site cible existe dans ds2id
    target_label = cfg["target_label"]
    print(f"[inference_harmoniser] label cible : {target_label}")
    if target_label not in ds2id:
        labels_disponibles = sorted(ds2id.keys())
        raise ValueError(
            f"[inference_harmoniser] label cible '{target_label}' absent du checkpoint.\n"
            f"Labels disponibles : {labels_disponibles}"
        )
    target_id = ds2id[target_label]
    print(f"[inference_harmoniser] label cible '{target_label}' -> id {target_id}")

    # DataLoader test (slices à harmoniser)
    # On force target_label pour toutes les slices : le label original du test set (gamma_0.0) n'existe pas forcément dans ds2id mais on l'ajoute
    # Car on ignore le label original dans la boucle (on utilise cond_emb du target_label).
    # Le collate doit juste retourner un tensor valide -> on mappe tout sur target_label.
    ds2id_inference = {label: ds2id[target_label] for label in [target_label]}
    ds2id_inference["__default__"] = ds2id[target_label]

    collate = partial(collate_force_target, target_id=ds2id[target_label])



    test_ds = SliceDataset(
        slice_dir=cfg["test_dir"],
        train=False,
        anatomy_mode=cfg["anatomy_mode"],
        anatomy_csv_path=cfg["anatomy_csv_path_test"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
        prefetch_factor=2 if cfg["num_workers"] > 0 else None,
        collate_fn=collate,
    )
    test_loader = accelerator.prepare(test_loader)

    noise_scheduler = DDIMScheduler(num_train_timesteps=cfg["num_train_timesteps"])

    model_diffusion.eval()
    embedder.eval()

    # Embedding cible fixe : même vecteur pour tout le dataset
    target_id_tensor = torch.tensor([target_id], device=accelerator.device, dtype=torch.long)
    uncond_id_tensor = torch.tensor([0],         device=accelerator.device, dtype=torch.long)

    saved_count = 0

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="harmonisation", disable=not accelerator.is_main_process)):
            slices, anat_maps, _ = batch   # on ignore les label_ids originaux
            slices    = slices.to(accelerator.device, non_blocking=True).float()
            anat_maps = anat_maps.to(accelerator.device, non_blocking=True).float()

            batch_size = slices.shape[0]

            # Embedding cible étendu à la taille du batch
            cond_emb   = embedder(target_id_tensor.expand(batch_size)).unsqueeze(1)   # (B, 1, cross_dim)
            uncond_emb = embedder(uncond_id_tensor.expand(batch_size)).unsqueeze(1)   # (B, 1, cross_dim)

            # padding -> multiplke de 16 (besoin pour le modèle)
            slices_pad,    orig_h, orig_w = pad_to_multiple(slices)
            anat_maps_pad, _,      _      = pad_to_multiple(anat_maps)
            
            # Harmonisation via DDIM partiel
            print(f"slices shape: {slices.shape}")  # -> (B, 1, H, W)
            harmonised = ddim_inference(
                model=accelerator.unwrap_model(model_diffusion),
                noise_scheduler=noise_scheduler,
                eval_slices=slices_pad,       # sert uniquement à torch.randn_like pour le bruit initial
                anat_map=anat_maps_pad,
                cond_emb=cond_emb,
                uncond_emb=uncond_emb,
                accelerator=accelerator,
                num_inference_steps=cfg["num_inference_steps"],
                guidance_scale=cfg.get("guidance_scale", 1.0),
            )
            harmonised = harmonised[:, :, :orig_h, :orig_w] # On remet à la taille originale

            # Sauvegarde des slices harmonisées en .npy
            if accelerator.is_main_process:
                # npy_files est la liste des fichiers dans l'ordre du DataLoader
                # On récupère les noms originaux via test_ds.npy_files
                batch_paths = test_ds.npy_files[saved_count:saved_count + batch_size]
                for i in range(batch_size):
                    arr      = harmonised[i, 0].float().cpu().numpy()
                    orig_name = Path(batch_paths[i]).stem
                    out_path  = os.path.join(output_dir, f"{orig_name}.npy")
                    np.save(out_path, arr)
                    saved_count += 1

    if accelerator.is_main_process:
        _save_control_figure(
            output_dir=output_dir,
            target_label=target_label,
            epoch_loaded=epoch_loaded,
            test_ds=test_ds,
            n_subjects=cfg.get("n_subjects_control", 3),
        )


def collate_force_target(batch, target_id):
    slices   = torch.stack([b[0] for b in batch])
    anat_maps = torch.stack([b[1] for b in batch])
    ids      = torch.tensor([target_id] * len(batch), dtype=torch.long)
    return slices, anat_maps, ids


def _save_control_figure(output_dir, target_label, epoch_loaded, test_ds, n_subjects=3):
    """
    Produit n_subjects figures (un plot par sujet),
    Chaque plot montre les slices de tous les sites du sujet, harmonisés : 


    Attend des fichiers nommés : {stem_original}.npy dans output_dir,
    où stem contient "sub{N}" permettant d'extraire le numéro de sujet.

    Args:
        output_dir:   dossier contenant les .npy harmonisés
        target_label: label cible (pour le titre)
        epoch_loaded: epoch du checkpoint (pour le titre)
        test_ds :     Le dataset pour obtenir les raw et encoded slices
        n_subjects:   nombre de sujets distincts à afficher
    """
    harmonised_files = sorted(Path(output_dir).glob("*.npy"))
    if not harmonised_files:
        print("[_save_control_figure] aucun fichier .npy trouvé")
        return

    # Grouper les harmonisées par sujet
    subjects = defaultdict(list)
    for f in harmonised_files:
        match = re.search(r"sub(\d+)", f.stem)
        if match:
            subjects[match.group(1)].append(f)

    # Indexer les fichiers raw du dataset par stem
    raw_index = {Path(p).stem: p for p in test_ds.npy_files}

    selected_subjects = sorted(subjects.keys())[:n_subjects]

    for sub_id in selected_subjects:
        files    = sorted(subjects[sub_id])
        n_slices = len(files)

        # 3 colonnes : Original | Anat | Harmonisé
        fig, axes = plt.subplots(
            n_slices, 3,
            figsize=(12, 3.5 * n_slices),
            squeeze=False
        )

        col_titles = ["Original", "Anat map", "Harmonisé"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

        for row, f in enumerate(files):
            stem = f.stem

            # --- Harmonisée ---
            harm_arr = np.load(f)

            # --- Raw ---
            raw_path = raw_index.get(stem)
            if raw_path is not None:
                raw_arr = np.load(raw_path).astype(np.float32)
            else:
                raw_arr = np.zeros_like(harm_arr)
                print(f"[_save_control_figure] raw introuvable pour {stem}")

            # --- Anat map : recalculée depuis le dataset ---
            # On cherche l'index dans test_ds pour charger via __getitem__
            try:
                idx      = [Path(p).stem for p in test_ds.npy_files].index(stem)
                _, anat, _ = test_ds[idx]
                anat_arr = anat[0].numpy()   # (1, H, W) -> (H, W)
            except (ValueError, Exception) as e:
                anat_arr = np.zeros_like(harm_arr)
                print(f"[_save_control_figure] anat introuvable pour {stem} : {e}")

            for col, arr in enumerate([raw_arr, anat_arr, harm_arr]):
                im_show = np.rot90(arr)
                im_norm = (im_show - im_show.min()) / (im_show.max() - im_show.min() + 1e-8)
                axes[row, col].imshow(im_norm, cmap="gray", vmin=0, vmax=1)
                axes[row, col].axis("off")

            # Nom du site en label de ligne
            axes[row, 0].set_ylabel(stem, fontsize=7, rotation=0,
                                    labelpad=120, va="center")

        plt.tight_layout()
        fig_path = os.path.join(
            output_dir,
            f"control_sub{sub_id}_vers_{target_label}_ep{epoch_loaded}.png"
        )
        fig.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[_save_control_figure] figure sauvegardée -> {fig_path}")