'''
Pour utiliser le modèle de diffusion 2D.

Ce fichier sert à
    - Choisir le fichier de config (json) utiliser,
    - Choisir si on entraîne le model (train) ou bien on l'évalue seulement (eval)

Lancement :
    accelerate launch main_diffusion_2d.py
    accelerate launch --num_processes 4 main_diffusion_2d.py   # multi-GPU
'''
 
from config_loader import load_config
from train_diffusion import train, evaluate
 

"""
CONFIG_PATH = "configs/config1.json" 
MODE = "train" # train ou eval
EVAL_CHECKPOINT_PATH = None # (obligatoire si MODE == "eval")
"""


# Eval reverse beta encoded anat
"""
CONFIG_PATH = "configs/eval_configs/eval_config_reverse.json" 
MODE = "eval" # train ou eval
EVAL_CHECKPOINT_PATH = "/NAS/coolio/benolive/Diffusion_beta_encoder/checkpoints_diffusion/diffusion_2d/config_reverse_encoded/best_ckpt.pt" 
"""


# Eval beta encoded anatomy
#"""
CONFIG_PATH = "configs/config2.json" 
MODE = "eval" # train ou eval
EVAL_CHECKPOINT_PATH = "/NAS/coolio/benolive/Diffusion_beta_encoder/checkpoints_diffusion/diffusion_2d/config2/best_ckpt.pt" 
#"""


# Eval anat naive
"""
CONFIG_PATH = "configs/config1.json" 
MODE = "eval" # train ou eval
EVAL_CHECKPOINT_PATH = "/NAS/coolio/benolive/Diffusion_beta_encoder/checkpoints_diffusion/diffusion_2d/config1/best_ckpt.pt" 
"""


def main():
    cfg = load_config(CONFIG_PATH)
 
    if MODE == "train":
        train(cfg)
 
    elif MODE == "eval":
        if EVAL_CHECKPOINT_PATH is None:
            raise ValueError("EVAL_CHECKPOINT_PATH doit être renseigné quand MODE='eval'")
        evaluate(cfg, EVAL_CHECKPOINT_PATH)
 
    else:
        raise ValueError(f"MODE inconnu : '{MODE}' (Utiliser uniquement : 'train' ou 'eval')")
    


 
 
if __name__ == "__main__":
    main()