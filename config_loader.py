"""
Charge les fichiers de paramétrage (JSON) pour utiliser le modèle de diffusion (2d)
"""
 
import json
from pathlib import Path
 

import json
from pathlib import Path
 
# Clés communes à tous les modes (architecture du modèle)
_REQUIRED_MODEL_KEYS = [
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
    "num_train_timesteps",
    "num_inference_steps",
    "batch_size",
    "num_workers",
    "checkpoint_dir",
    "anatomy_mode",
]
 
# Clés supplémentaires requises uniquement en mode train ou evaluate
_REQUIRED_TRAIN_KEYS = [
    "train_dir",
    "test_dir",
    "tb_log_dir",
    "resume_from",
    "anatomy_csv_path_train",
    "anatomy_csv_path_test",
    "num_epochs",
    "lr",
    "p_uncond",
    "eval_every_epoch",
    "save_every_epoch",
    "early_stopping_patience",
    "early_stopping_min_delta",
]
 
# Clés supplémentaires requises uniquement en mode inference
_REQUIRED_INFERENCE_KEYS = [
    "test_dir",
    "anatomy_csv_path_test",
    "target_label",
    "guidance_scale",
    "p_uncond",
]

REQUIRED_KEYS_PER_MODE = {
    "train":     _REQUIRED_MODEL_KEYS + _REQUIRED_TRAIN_KEYS,
    "inference": _REQUIRED_MODEL_KEYS + _REQUIRED_INFERENCE_KEYS,
}


 
 
def load_config(config_path,  mode="train"):
    """
    Charge un fichier de config JSON et retourne un dict utilisable comme CFG.

    config_path = path vers fichier.json

    mode: "train" (défaut) ou "inference" -> cela détermine les clés requises dans la config
    """
    if mode not in REQUIRED_KEYS_PER_MODE:
        raise ValueError(f"[config_loader] mode inconnu : '{mode}' (attendu : 'train' ou 'inference')")


    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"[config_loader] Fichier de config n'existe pas : {config_path}")
 
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
 
    # Enlever des clés de commentaire
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
 
    missing = [k for k in REQUIRED_KEYS_PER_MODE[mode] if k not in cfg]
    if missing:
        raise ValueError(
            f"[config_loader] Clés manquantes dans {config_path.name} (mode='{mode}') : {missing}"
        )

 
    
    cfg["block_out_channels"] = tuple(cfg["block_out_channels"])    # ex : tuple -> (32, 64, 256, 512)
 
    print(f"[config_loader] Config chargée")
    return cfg
