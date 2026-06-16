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