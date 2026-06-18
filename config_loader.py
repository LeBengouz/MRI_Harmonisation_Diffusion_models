"""
Charge les fichiers de paramétrage (JSON) pour utiliser le modèle de diffusion (2d)
"""
 
import json
from pathlib import Path
 
# Clés obligatoires attendues dans tout fichier de config.
REQUIRED_KEYS = [
    "train_dir",
    "test_dir",
    "checkpoint_dir",
    "tb_log_dir",
    "resume_from",
    "anatomy_mode",
    "anatomy_csv_path_train",
    "anatomy_csv_path_test",
    "num_epochs",
    "lr",
    "batch_size",
    "eval_every_epoch",
    "save_every_epoch",
    "p_uncond",
    "num_workers",
    "num_train_timesteps",
    "num_inference_steps",
    "in_channels",
    "out_channels",
    "block_out_channels",
    "cross_attention_dim",
    "attention_head_dim",
    "time_embedding_dim",
    "gradient_checkpointing",
    "attention_pool",
    "attn_pool_kernel",
    "attn_on_resolutions",
]
 
 
def load_config(config_path):
    """
    Charge un fichier de config JSON et retourne un dict utilisable comme CFG.

    config_path = path vers fichier.json
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"[config_loader] Fichier de config n'existe pas : {config_path}")
 
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
 
    # Enlever des clés de commentaire
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
 
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"[config_loader] Clés manquantes dans {config_path.name} : {missing}")
 
    
    cfg["block_out_channels"] = tuple(cfg["block_out_channels"])    # ex : tuple -> (32, 64, 256, 512)
 
    print(f"[config_loader] Config chargée")
    return cfg
