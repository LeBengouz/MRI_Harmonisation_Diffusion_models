"""
Saving and loading checkpoints from a directory and pt files.
"""

import torch
import os

def save_checkpoint(epoch, step, model, embedder, optimizer, checkpoint_dir, accelerator, filename=None, ds2id=None):
    if filename is None:
        filename = f"ckpt_ep{epoch:04d}_full.pt"
    fname = os.path.join(checkpoint_dir, filename)

    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        ckpt = {
            "model_state_dict": unwrapped.state_dict(),
            "embedder_state_dict": embedder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "ds2id": ds2id,   # None si non fourni, géré en lecture par .get()
        }
        torch.save(ckpt, fname)
        print(f"[checkpoint] saved -> {fname}")

def load_checkpoint_if_exists(resume_from, model_diffusion, embedder, optimizer, accelerator):
    """
    Charge checkpoint si resume_from n'est pas None et existe.
    Doit être appelé APRES accelerator.prepare(...) car on load dans les objets préparés.
    Retourne: start_epoch (int), global_step (int)
    """
    if resume_from is None:
        return 0, 0  # start at epoch 0, global_step 0

    ckpt = torch.load(resume_from, map_location=accelerator.device)

    # Unwrap models préparés par accelerator pour charger les state_dict
    unwrapped_model = accelerator.unwrap_model(model_diffusion)
    unwrapped_model.load_state_dict(ckpt["model_state_dict"])

    embedder.load_state_dict(ckpt["embedder_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("step", 0)

    print(f"Resumed from checkpoint {resume_from} -> start_epoch={start_epoch}, global_step={global_step}")

    # return the epoch index to start from and the global step count
    return start_epoch, global_step


def load_checkpoint_for_eval(checkpoint_path, model_diffusion, embedder, accelerator):
    """
    Charge un checkpoint pour évaluation seulement.
 
    Contrairement à load_checkpoint_if_exists, ne prend PAS d'optimizer.
    Doit être appelé APRES accelerator.prepare(model_diffusion, embedder, ...).

    Il faut absolument que le checkpoint existe :
    la validation de checkpoint_path (None ou non) est faite en amont par
    l'appelant (cf. evaluate() dans train_diffusion.py) 
    """
    ckpt = torch.load(checkpoint_path, map_location=accelerator.device)
 
    unwrapped_model = accelerator.unwrap_model(model_diffusion)
    unwrapped_model.load_state_dict(ckpt["model_state_dict"])
    embedder.load_state_dict(ckpt["embedder_state_dict"])
 
    epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("step", 0)

    ds2id = ckpt.get("ds2id", None)   # None pour les anciens checkpoints qui ont pas ds2id (penser à relancer les entraînements précédents)
    if ds2id is None:
        print("[load_checkpoint_for_eval] /!\ ds2id absent du checkpoint (ancien format) - à reconstruire depuis la config")
    else:
        print(f"[load_checkpoint_for_eval] ds2id chargé ({len(ds2id)} classes)")

    print(f"Checkpoint chargé depuis {checkpoint_path} à partir de l'epoch ={epoch}, global_step={global_step}")
    return epoch, global_step, ds2id


def read_checkpoint_metadata(checkpoint_path, accelerator):
    """
    Lit uniquement les métadonnées d'un checkpoint (epoch, step, ds2id) mais ne charge PAS les poids.
    Utile pour connaître ds2id avant de construire l'embedder à la bonne taille lorsqu'on fait l'inference (tâche d'harmonisation)
    """
    ckpt   = torch.load(checkpoint_path, map_location=accelerator.device)
    epoch  = ckpt.get("epoch", 0)
    step   = ckpt.get("step", 0)
    ds2id  = ckpt.get("ds2id", None)
    if ds2id is None:
        raise ValueError(
            f"[read_checkpoint_metadata] ds2id absent de {checkpoint_path}.\n"
            "Ce checkpoint a été sauvegardé avant l'ajout de ds2id dans save_checkpoint."
        )
    print(f"[read_checkpoint_metadata] ds2id lu ({len(ds2id)} classes, epoch={epoch})")
    return epoch, step, ds2id