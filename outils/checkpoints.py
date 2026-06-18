"""
Saving and loading checkpoints from a directory and pt files.
"""

import torch
import os

def save_checkpoint(epoch, step, model, embedder, optimizer, checkpoint_dir, accelerator):
    accelerator.wait_for_everyone()
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        ckpt = {
            "model_state_dict": unwrapped.state_dict(),
            "embedder_state_dict": embedder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }
        fname = os.path.join(checkpoint_dir, f"ckpt_ep{epoch:04d}_full.pt")
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
    print(f"Checkpoint chargé depuis {checkpoint_path} à partir de l'epoch ={epoch}, global_step={global_step}")
    return epoch, global_step
