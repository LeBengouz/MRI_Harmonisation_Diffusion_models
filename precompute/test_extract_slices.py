"""
Script de test : extrait N slices variées (échantillon aléatoire) depuis le CSV d'entrée.
Contrairement au pipeline principal :
- Ne traite que N volumes (tirés aléatoirement dans le CSV),
- N'enregistre PAS les fichiers .npy (format raw),
- Enregistre uniquement les .png, SANS normalisation des intensités.
"""

import csv
import gc
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

import torch

# ── Paramètres ──────────────────────────────────────────────────────────────

DEFAULT_INPUT_CSV      = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/data/csv_files/listing_data_train.csv")
DEFAULT_PNG_OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/data/brain_slices/experience_png/slices")
DEFAULT_CSV_OUT        = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/data/brain_slices/experience_png/test_slices.csv")
DEFAULT_Z_RATIO        = 0.5
DEFAULT_N              = 5
DEFAULT_NUM_WORKERS    = 4
DEFAULT_RANDOM_SEED    = 307

VOLUME_PATH_COLUMN = "volume_path"


# ── Fonctions (identiques au pipeline principal) ─────────────────────────────

def load_volume_from_pt(volume_path: Path) -> tuple[np.ndarray, str]:
    """
    Charge un fichier .pt contenant un dictionnaire {'volume': Tensor, 'ds_name': str}.
    Retourne le volume sous forme de ndarray float32 et le ds_name associé.
    """
    volume_path = Path(volume_path)
    if not volume_path.suffix == ".pt":
        raise ValueError(f"Fichier .pt attendu, obtenu : {volume_path}")

    data = torch.load(volume_path, map_location="cpu", weights_only=True)

    if not isinstance(data, dict):
        raise ValueError(f"Dictionnaire attendu dans le .pt, obtenu : {type(data)}")
    if "volume" not in data:
        raise KeyError(f"Clé 'volume' absente dans {volume_path.name}. Clés présentes : {list(data.keys())}")

    volume  = data["volume"]
    ds_name = data.get("ds_name", "")

    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()
    volume = volume.astype(np.float32)

    return volume, ds_name


def extract_central_slice(volume: np.ndarray, target_z_ratio: float = 0.5) -> tuple[np.ndarray, int]:
    """
    Extrait la slice axiale à la position relative `target_z_ratio` sur l'axe Z
    (0.5 = milieu du volume). Accepte un ndarray 3D de shape (X, Y, Z).
    """
    if volume.ndim != 3:
        raise ValueError(f"Volume 3D attendu, obtenu shape={volume.shape}")

    X, Y, Z = volume.shape
    z_idx   = int(round(Z * target_z_ratio))
    z_idx   = max(0, min(z_idx, Z - 1))

    slice_2d = volume[:, :, z_idx].astype(np.float32)

    return slice_2d, z_idx


def save_slice_as_rgb_png(array_2d: np.ndarray, output_path: Path) -> None:
    """
    Sauvegarde une slice 2D NumPy au format PNG RGB.
    Normalise les intensités sur [0, 255], puis duplique le canal grayscale
    en 3 canaux RGB : R=G=B.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array_2d = np.asarray(array_2d, dtype=np.float32)
    array_2d = np.nan_to_num(array_2d, nan=0.0, posinf=0.0, neginf=0.0)

    vmin, vmax = array_2d.min(), array_2d.max()
    
    if vmax > vmin:
        normalized = (array_2d - vmin) / (vmax - vmin) * 255.0
    else:
        normalized = np.zeros_like(array_2d)

    gray_uint8 = normalized.astype(np.uint8)

    rgb_uint8 = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    img = Image.fromarray(rgb_uint8, mode="RGB")
    img.save(str(output_path))


def save_slice_as_rgb_png_no_norm(array_2d: np.ndarray, output_path: Path) -> None:
    """
    Sauvegarde une slice 2D NumPy au format PNG RGB SANS normalisation.
    Les valeurs sont simplement castées en uint8 (clip sur [0, 255]).
    R=G=B (grayscale répliqué).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    array_2d = np.asarray(array_2d, dtype=np.float32)
    array_2d = np.nan_to_num(array_2d, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip et cast direct, sans remise à l'échelle
    gray_uint8 = np.clip(array_2d, 0, 255).astype(np.uint8)

    rgb_uint8 = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    img = Image.fromarray(rgb_uint8, mode="RGB")
    img.save(str(output_path))


def build_output_paths(volume_path: Path, output_dir: Path, png_output_dir: Path) -> tuple[Path, Path]:
    """
    Construit les chemins de sortie .npy et .png en remplaçant l'extension .pt
    par les extensions cibles, en conservant le stem du fichier source.
    """
    stem     = volume_path.stem
    npy_path = output_dir     / f"{stem}.npy"
    png_path = png_output_dir / f"{stem}.png"
    return npy_path, png_path


# ── Traitement d'un volume (test : PNG sans normalisation, pas de .npy) ──────

def process_one_volume_test(
    row:            dict,
    png_output_dir: Path,
    z_ratio:        float,
    index:          int,
    total:          int,
) -> tuple[dict | None, str | None]:
    """
    Traite un seul volume pour le mode test :
    - Charge le volume,
    - Extrait la slice centrale,
    - Enregistre UNIQUEMENT le .png, SANS normalisation.
    Retourne (row_out, None) en cas de succès, (None, message_erreur) sinon.
    """
    pt_path = Path(row.get(VOLUME_PATH_COLUMN, ""))
    try:
        volume, ds_name = load_volume_from_pt(pt_path)
        slice_2d, z_idx = extract_central_slice(volume, target_z_ratio=z_ratio)

        # On réutilise build_output_paths mais on n'utilisera que png_path
        _, png_path = build_output_paths(pt_path, png_output_dir, png_output_dir)

        # PNG sans normalisation
        #save_slice_as_rgb_png_no_norm(slice_2d, png_path)
        save_slice_as_rgb_png(slice_2d, png_path)

        print(
            f"[{index}/{total}] {pt_path.name}  "
            f"shape={volume.shape}  z_idx={z_idx}  "
            f"min={slice_2d.min():.1f}  max={slice_2d.max():.1f}"
        )

        row_out = {
            (k if k != VOLUME_PATH_COLUMN else "png_path"): v
            for k, v in row.items()
        }
        row_out["png_path"]  = str(png_path)
        row_out["slice_idx"] = z_idx

        return row_out, None

    except Exception as e:
        print(f"[{index}/{total}] [ERREUR] {pt_path.name} : {e}")
        return None, f"{pt_path.name} : {e}"

    finally:
        try:
            del volume, slice_2d
        except NameError:
            pass
        gc.collect()


# ── Pipeline de test ─────────────────────────────────────────────────────────

def process_test_volumes(
    input_csv:      Path = DEFAULT_INPUT_CSV,
    png_output_dir: Path = DEFAULT_PNG_OUTPUT_DIR,
    csv_out:        Path = DEFAULT_CSV_OUT,
    z_ratio:        float = DEFAULT_Z_RATIO,
    n:              int   = DEFAULT_N,
    num_workers:    int   = DEFAULT_NUM_WORKERS,
    random_seed:    int   = DEFAULT_RANDOM_SEED,
) -> None:
    """
    Sélectionne N volumes aléatoirement dans le CSV d'entrée, extrait leur slice
    centrale, et enregistre uniquement les .png (sans normalisation).
    """
    input_csv      = Path(input_csv)
    png_output_dir = Path(png_output_dir)
    csv_out        = Path(csv_out)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV d'entrée introuvable : {input_csv}")

    png_output_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Lecture du CSV d'entrée
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader        = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        all_rows      = list(reader)

    if not all_rows:
        raise RuntimeError(f"CSV vide : {input_csv}")

    if VOLUME_PATH_COLUMN not in fieldnames_in:
        raise KeyError(
            f"Colonne '{VOLUME_PATH_COLUMN}' absente du CSV. "
            f"Colonnes disponibles : {fieldnames_in}"
        )

    # Échantillonnage aléatoire de N lignes variées
    n_actual = min(n, len(all_rows))
    random.seed(random_seed)
    sampled_rows = random.sample(all_rows, n_actual)

    print(f"{len(all_rows)} volume(s) disponible(s) dans {input_csv.name}")
    print(f"Échantillon aléatoire : {n_actual} volume(s) (seed={random_seed})")
    print(f"Parallélisation       : {num_workers} workers")
    print(f"Normalisation         : désactivée\n")

    # Colonnes de sortie
    fieldnames_out = list(fieldnames_in)
    idx_pt = fieldnames_out.index(VOLUME_PATH_COLUMN)
    fieldnames_out[idx_pt] = "png_path"
    if "slice_idx" not in fieldnames_out:
        fieldnames_out.append("slice_idx")

    results = [None] * n_actual

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(
                process_one_volume_test,
                row, png_output_dir, z_ratio, i + 1, n_actual,
            ): i
            for i, row in enumerate(sampled_rows)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()

    # Écriture du CSV dans l'ordre de l'échantillon
    ok, errors = 0, []
    with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames_out)
        writer.writeheader()
        for row_out, err in results:
            if row_out is not None:
                writer.writerow(row_out)
                ok += 1
            else:
                errors.append(err)

    print(f"\n{'='*60}")
    print(f"Test terminé — {ok}/{n_actual} volumes traités avec succès.")
    print(f"Slices .png enregistrées dans : {png_output_dir}")
    print(f"CSV de slices                 : {csv_out}")

    if errors:
        print(f"\n{len(errors)} erreur(s) :")
        for msg in errors:
            print(f"  {msg}")


if __name__ == "__main__":
    process_test_volumes(
        input_csv      = DEFAULT_INPUT_CSV,
        png_output_dir = DEFAULT_PNG_OUTPUT_DIR,
        csv_out        = DEFAULT_CSV_OUT,
        z_ratio        = DEFAULT_Z_RATIO,
        n              = DEFAULT_N,
    )
