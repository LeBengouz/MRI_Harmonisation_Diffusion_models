'''
Boucle de train du modèle de diffusion 2D (slices)

Dépend de :
    - models/diffusion/unet.py        = UNet2DConditionModel_Optimized
    - dataset.py                      = SliceDataset, build_label_mapping, build_label_mapping_from_csv, collate_fn_with_label_ids
    - outils/checkpoints.py           = save_checkpoint, load_checkpoint_if_exists


Utilise aussi pour le training et suivi:
    - accelerate
    - diffusers
    - torch
    - tenserboard

Lancement :
    accelerate launch training_diffusion.py
    accelerate launch --num_processes 4 training_diffusion.py   # multi-GPU
'''



import os
import random
from functools import partial
from pathlib import Path
 
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import DDIMScheduler
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
 
from dataset import (
    SliceDataset,
    build_label_mapping,
    build_label_mapping_from_csv,
    collate_fn_with_label_ids,
)
from models.diffusion.unet import UNet2DConditionModel_Optimized
from outils.checkpoints import load_checkpoint_if_exists, save_checkpoint, load_checkpoint_for_eval

from outils.visualization import plot_eval_batch
import matplotlib.pyplot as plt



# setup des dataloaders 

def build_dataloaders(cfg, ds2id):
    """
    Construit les DataLoaders de train train et test à partir des classes de dataset.py
    """

    collate = partial(
        collate_fn_with_label_ids,
        ds2id=ds2id,
        p_uncond=cfg["p_uncond"],
    )


    train_ds = SliceDataset(slice_dir=cfg["train_dir"], train=True, anatomy_mode=cfg["anatomy_mode"],anatomy_csv_path=cfg["anatomy_csv_path_train"])
    test_ds = SliceDataset(slice_dir=cfg["test_dir"], train=False, anatomy_mode=cfg["anatomy_mode"], anatomy_csv_path=cfg["anatomy_csv_path_test"])

    # Data Loaders
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"], 
                              pin_memory=True, persistent_workers=cfg["num_workers"] > 0, prefetch_factor=2 if cfg["num_workers"] > 0 else None,
                              collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], 
                             pin_memory=True, persistent_workers=cfg["num_workers"] > 0, prefetch_factor=2 if cfg["num_workers"] > 0 else None,
                             collate_fn=collate)
    return train_loader, test_loader


def build_model(cfg):
    return UNet2DConditionModel_Optimized(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        block_out_channels=cfg["block_out_channels"],
        cross_attention_dim=cfg["cross_attention_dim"],
        attention_head_dim=cfg["attention_head_dim"],
        time_embedding_dim=cfg["time_embedding_dim"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        attention_pool=cfg["attention_pool"],
        attn_pool_kernel=cfg["attn_pool_kernel"],
        attn_on_resolutions=cfg["attn_on_resolutions"],
    )


def build_label_mapping_for_cfg(cfg):
    if cfg["anatomy_mode"] == "encoded":
        csv_paths = [
            p for p in (cfg["anatomy_csv_path_train"], cfg["anatomy_csv_path_test"])
            if p is not None
        ]
        if len(csv_paths) == 0:
            raise ValueError(
                "[build_label_mapping_for_cfg] anatomy_mode='encoded' requiert au moins "
                "anatomy_csv_path_train ou anatomy_csv_path_test dans la config"
            )
        ds2id, _ = build_label_mapping_from_csv(csv_paths)
    else:
        ds2id, _ = build_label_mapping(cfg["train_dir"])
    return ds2id




# INFERENCE DDIM (eval)
@torch.no_grad()
def ddim_inference(model: nn.Module, noise_scheduler: DDIMScheduler, eval_slices, anat_map, cond_emb, uncond_emb, accelerator, num_inference_steps = 50, guidance_scale = 1.0):
    """
    Génération DDIM guidée par classifier-free guidance.

    Classifier-Free Guidance : On a entrainé un seul modèle pouvant générer avec et sans condition
                               -> Inférence : combiner les deux prédictions
 
    Args:
        model:               UNet2DConditionModel_Optimized
        noise_scheduler:     DDIMScheduler configuré avec num_inference_steps
        eval_slices:         (B, 1, H, W) - batch eval, utilisé pour torch.randn_like
        anat_map:            (B, 1, H, W) - carte anatomique du batch eval
        cond_emb:            (B, 1, cross_attention_dim) - embedding conditionné
        uncond_emb:          (B, 1, cross_attention_dim) - embedding non-conditionné
        device:              device cible
        num_inference_steps: nombre de pas DDIM
        guidance_scale:      lambda du CFG (1.0 = pas de guidance)
 
    Returns:
        noise : (B, 1, H, W) - slices générées
    """
    batch_size = anat_map.shape[0]
 
    noise_scheduler.set_timesteps(num_inference_steps)
    timesteps_iter = list(noise_scheduler.timesteps)
 
    num_train_timesteps = noise_scheduler.config.num_train_timesteps
    step_offset = num_train_timesteps // num_inference_steps
 
    alphas_cumprod      = noise_scheduler.alphas_cumprod.to(accelerator.device)
    final_alpha_cumprod = noise_scheduler.final_alpha_cumprod.to(accelerator.device)
 
    # Bruit initial
    gaussian_noise = torch.randn_like(eval_slices)
 
    for t in timesteps_iter:
        t_b = torch.tensor([int(t)] * batch_size, device=accelerator.device, dtype=torch.long)
 
        slice_in       = torch.cat([gaussian_noise, gaussian_noise], dim=0) # Double input : uncond + cond (classifier-free guidance)
        anat_in    = torch.cat([anat_map, anat_map], dim=0)                 # embedding 0 + embedding réel
        model_in   = torch.cat([slice_in, anat_in], dim=1)
        emb_in     = torch.cat([uncond_emb, cond_emb], dim=0) 
        t_in       = torch.cat([t_b, t_b], dim=0)               # transformé en embedding dans le model
 
        noise_pred_both = model(model_in, t_in, encoder_hidden_states=emb_in)
        uncond_pred, cond_pred = noise_pred_both.chunk(2, dim=0)
 
        # Classifier-free guidance : combinaison des deux
        pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
 
        prev_t    = t - step_offset
        alpha_t   = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[prev_t] if prev_t >= 0 else final_alpha_cumprod
        beta_t    = 1 - alpha_t
 
        x0_pred               = (gaussian_noise - beta_t**0.5 * pred) / alpha_t**0.5
        pred_sample_direction = (1 - alpha_prev)**0.5 * uncond_pred         # coeff_dir * uncond_pred
        gaussian_noise        = alpha_prev**0.5 * x0_pred + pred_sample_direction
 
    return gaussian_noise


# Boucle training DDIM


def train(cfg):
    print("[train] démarrage de l'entraînement")

    # Label mapping
    ds2id = build_label_mapping_for_cfg(cfg)

    # Debug - print des classes
    # print("labels:")
    # for label in list(ds2id.keys())[:]:
    #    print(f"  {label} -> {ds2id[label]}")
 
    n_classes = len(ds2id)
    print(f"[dataset] {n_classes} classes")
 
    accelerator = Accelerator(mixed_precision="fp16")
 
    model_diffusion = build_model(cfg)
    embedder        = nn.Embedding(n_classes + 1, cfg["cross_attention_dim"])  # +1 pour token car 0 réservé pour non-conditionné
    optimizer       = Adam(list(model_diffusion.parameters()) + list(embedder.parameters()), lr=cfg["lr"], eps=1e-8)
 
    noise_scheduler = DDIMScheduler(num_train_timesteps=cfg["num_train_timesteps"])
    noise_scheduler.set_timesteps(cfg["num_train_timesteps"])
    mse_loss = nn.MSELoss()
    
    train_loader, test_loader = build_dataloaders(cfg, ds2id)
 
    # Préparation accelerator -> fait automatiquement l'équivalent de .to(device)
    model_diffusion, embedder, optimizer, train_loader, test_loader = accelerator.prepare(
        model_diffusion, embedder, optimizer, train_loader, test_loader
    )
 
    # Reprise depuis checkpoint; si resume_from=None, la fonction retourne (0, 0) -> start_epoch = 0, global_step=0
    start_epoch, global_step = load_checkpoint_if_exists(
        cfg["resume_from"], model_diffusion, embedder, optimizer, accelerator
    )
 
    # Logger TensorBoard (Main process only)
    writer = None
    if accelerator.is_main_process:
        os.makedirs(cfg["tb_log_dir"], exist_ok=True)
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
        writer = SummaryWriter(cfg["tb_log_dir"])
        print(f"[logger] TensorBoard writer créé dans {cfg['tb_log_dir']}")
 

    # TRAINING LOOP 
    for epoch in range(start_epoch, cfg["num_epochs"]):
        model_diffusion.train()
        epoch_loss = 0.0
 
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}", ncols=120, disable=not accelerator.is_main_process)
 
        for step_in_epoch, batch in progress_bar:
            global_step += 1
 
            # batch = (slices, anat_maps, label_ids) - produit par collate_fn_with_label_ids
            # label_ids sont déjà convetis en int et masqués CFG (p_uncond) par le collate (cf. fonction dans dataset.py)
            slices, anat_maps, label_ids = batch
            slices    = slices.to(accelerator.device, non_blocking=True).float()
            anat_maps = anat_maps.to(accelerator.device, non_blocking=True).float()
            label_ids = label_ids.to(accelerator.device)
 
            batch_size = slices.shape[0]
 
            # Timesteps aléatoires
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (batch_size,),
                device=accelerator.device,
                dtype=torch.long,
            )
 
            noise         = torch.randn_like(slices)
            noisy_latents = noise_scheduler.add_noise(slices, noise, timesteps)
 
            # Embedding label (label_ids déjà masqués CFG par collate)
            label_embedding = embedder(label_ids).unsqueeze(1)
 
            # Forward UNet -> input = slice bruitée + carte anatomique (2 canaux)
            model_input = torch.cat([noisy_latents, anat_maps], dim=1)  # (B, 2, H, W)
            noise_pred  = model_diffusion(
                model_input,
                timesteps,
                encoder_hidden_states=label_embedding,
            )
 
            # Loss
            loss = mse_loss(noise_pred.float(), noise.float())
            epoch_loss += loss.item()
 
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad()
 
            avg_loss = epoch_loss / float(step_in_epoch + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.6f}"})
 
        # TensorBoard logging epoch
        if accelerator.is_main_process and writer is not None:
            writer.add_scalar("train/loss_epoch", epoch_loss / len(train_loader), epoch)
 

        # EVAL
        if (epoch + 1) % cfg["eval_every_epoch"] == 0 or epoch == start_epoch:
            model_diffusion.eval()
 
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
 
                # Récupérer un seul batch de test
                eval_slices, eval_anat_maps, eval_label_ids = next(iter(test_loader))
                eval_slices    = eval_slices.to(accelerator.device, non_blocking=True).float()
                eval_anat_maps = eval_anat_maps.to(accelerator.device, non_blocking=True).float()
                eval_label_ids = eval_label_ids.to(accelerator.device)
 
                # Embeddings conditionné et non-conditionné
                cond_emb   = embedder(eval_label_ids).unsqueeze(1)                                       # (B, 1, cross_dim)
                uncond_ids = torch.zeros_like(eval_label_ids, dtype=torch.long)
                uncond_emb = embedder(uncond_ids).unsqueeze(1)                                           # (B, 1, cross_dim)
 
                # Génération DDIM
                diffused_latents = ddim_inference(
                    model=accelerator.unwrap_model(model_diffusion),
                    noise_scheduler=noise_scheduler,
                    eval_slices=eval_slices,
                    anat_map=eval_anat_maps,
                    cond_emb=cond_emb,
                    uncond_emb=uncond_emb,
                    accelerator=accelerator,
                    num_inference_steps=cfg["num_inference_steps"],
                    guidance_scale=1.0,
                )

                if accelerator.is_main_process:
                    fig = plot_eval_batch(eval_slices, eval_anat_maps, diffused_latents, epoch)
                    if writer is not None:
                        writer.add_figure("eval/inference_visualization", fig, epoch)
                    fig.savefig(os.path.join(cfg["checkpoint_dir"], f"vis_epoch{epoch+1:04d}.png"), bbox_inches="tight")
                    plt.close(fig)
            
 
        # Sauvegarde checkpoint
        if (epoch + 1) % cfg["save_every_epoch"] == 0 or epoch == cfg["num_epochs"] - 1:
            save_checkpoint(
                epoch + 1,
                global_step,
                model_diffusion,
                embedder,
                optimizer,
                cfg["checkpoint_dir"],
                accelerator,
            )
 
    # Fermeture TensorBoard
    if accelerator.is_main_process and writer is not None:
        writer.close()


# Evaluation seule, sans entraînement
def evaluate(cfg, checkpoint_path):
    """
    Charge un checkpoint existant et évalue le modèle :
        - calcule la loss MSE moyenne (bruit prédit vs bruit réel) sur tout test_loader,
        - sauvegarde une figure d'un batch.
 
    /!\ checkpoint_path est obligatoire ici
    """
    print(f"[Eval] Evaluation depuis {checkpoint_path}")
 
    if checkpoint_path is None:
        raise ValueError("[evaluate] checkpoint_path est requis pour évaluer un modèle")
 
    ds2id = build_label_mapping_for_cfg(cfg)
    n_classes = len(ds2id)
    print(f"[dataset] {n_classes} classes")
 
    accelerator = Accelerator(mixed_precision="fp16")
 
    model_diffusion = build_model(cfg)
    embedder        = nn.Embedding(n_classes + 1, cfg["cross_attention_dim"])
 
    noise_scheduler = DDIMScheduler(num_train_timesteps=cfg["num_train_timesteps"])
    mse_loss = nn.MSELoss()
 
    _, test_loader = build_dataloaders(cfg, ds2id) # si jeu eval, remplacer ici

    model_diffusion, embedder, test_loader = accelerator.prepare( model_diffusion, embedder, test_loader)
 
    epoch_loaded, step_loaded = load_checkpoint_for_eval(checkpoint_path, model_diffusion, embedder, accelerator)
    print(f"[Eval] checkpoint chargé (epoch={epoch_loaded}, step={step_loaded})")
 
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
 
    model_diffusion.eval()
 
    # 1) Loss MSE moyenne sur tout le test set
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="evaluate - loss", disable=not accelerator.is_main_process):
            slices, anat_maps, label_ids = batch
            slices = slices.to(accelerator.device, non_blocking=True).float()
            anat_maps = anat_maps.to(accelerator.device, non_blocking=True).float()
            label_ids = label_ids.to(accelerator.device)
 
            batch_size = slices.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.num_train_timesteps, (batch_size,),
                device=accelerator.device, dtype=torch.long,
            )
 
            noise = torch.randn_like(slices)
            noisy_latents = noise_scheduler.add_noise(slices, noise, timesteps)
            label_embedding = embedder(label_ids).unsqueeze(1)
 
            model_input = torch.cat([noisy_latents, anat_maps], dim=1)
            noise_pred = model_diffusion(model_input, timesteps, encoder_hidden_states=label_embedding)
 
            loss = mse_loss(noise_pred.float(), noise.float())
            total_loss += loss.item()
            n_batches += 1
 
    avg_loss = total_loss / max(n_batches, 1)
    if accelerator.is_main_process:
        print(f"[evaluate] loss MSE moyenne sur test set : {avg_loss:.6f}")
 
    # 2) Génération DDIM sur un batch + sauvegarde de la figure
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        eval_slices, eval_anat_maps, eval_label_ids = next(iter(test_loader))
        eval_slices    = eval_slices.to(accelerator.device, non_blocking=True).float()
        eval_anat_maps = eval_anat_maps.to(accelerator.device, non_blocking=True).float()
        eval_label_ids = eval_label_ids.to(accelerator.device)
 
        cond_emb   = embedder(eval_label_ids).unsqueeze(1)
        uncond_ids = torch.zeros_like(eval_label_ids, dtype=torch.long)
        uncond_emb = embedder(uncond_ids).unsqueeze(1)
 
        diffused_latents = ddim_inference(
            model=accelerator.unwrap_model(model_diffusion),
            noise_scheduler=noise_scheduler,
            eval_slices=eval_slices,
            anat_map=eval_anat_maps,
            cond_emb=cond_emb,
            uncond_emb=uncond_emb,
            accelerator=accelerator,
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=1.0,
        )
 
        if accelerator.is_main_process:
            fig = plot_eval_batch(eval_slices, eval_anat_maps, diffused_latents, epoch_loaded)
            fig.savefig(os.path.join(cfg["checkpoint_dir"], f"vis_eval_only_ep{epoch_loaded:04d}.png"), bbox_inches="tight")
            plt.close(fig)
            print(f"[evaluate] figure sauvegardée dans {cfg['checkpoint_dir']}")
 
    return avg_loss

