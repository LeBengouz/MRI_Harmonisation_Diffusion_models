"""
Diagnostic de mémoire et de temps de traitement.
Lance ce script à la place d'extract_slices pour identifier où le ralentissement se produit.
    -> Chercher solution pour accélérer le processus.


Diagnostic :
- RAM stable à 0.44G -> pas de fuite mémoire du tout
- temps_load passe de ~0.011s à ~1.77s exactement à partir de 120 -> le goulot est entièrement dans torch.load
- temps_total ≈ temps_load -> tout le reste (extraction slice, numpy) est négligeable

C'est le cache disque du NAS qui se sature à ~120 fichiers. 
Les 119 premiers sont servis depuis le cache RAM du système (quasi-instantané), ensuite chaque fichier est lu depuis le réseau/disque réel (~1.77s par fichier).
"""

import csv
import gc
import time
from pathlib import Path

import psutil
import torch
import numpy as np

# ── À adapter ─────────────────────────────────────────────────────────────────
INPUT_CSV      = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/data/csv_files/listing_data_train.csv")
PT_PATH_COLUMN = "volume_path"
N_DIAG         = 150   # nombre de volumes à tester
# ──────────────────────────────────────────────────────────────────────────────

def get_ram_gb():
    return psutil.Process().memory_info().rss / 1e9

with open(INPUT_CSV, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"{'i':>5}  {'temps_load':>12}  {'temps_total':>12}  {'RAM_GB':>8}  {'shape'}")
print("-" * 60)

for i, row in enumerate(rows[:N_DIAG], start=1):
    pt_path = Path(row[PT_PATH_COLUMN])
    t0 = time.perf_counter()

    data   = torch.load(pt_path, map_location="cpu", weights_only=True)
    t_load = time.perf_counter() - t0

    volume = data["volume"].numpy().astype(np.float32)
    shape  = volume.shape

    Z      = shape[2]
    z_idx  = int(round(Z * 0.5))
    slice_2d = volume[:, :, z_idx]

    t_total = time.perf_counter() - t0
    ram     = get_ram_gb()

    print(f"{i:>5}  {t_load:>11.3f}s  {t_total:>11.3f}s  {ram:>7.2f}G  {shape}")

    del data, volume, slice_2d
    gc.collect()
