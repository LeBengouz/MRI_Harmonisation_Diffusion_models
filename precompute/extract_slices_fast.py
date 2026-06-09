"""
Pour chaque volume référencé dans le CSV d'entrée (colonne `volume_path`) :
- Copie les fichiers .pt du NAS vers un dossier local temporaire par batches,
- Charge le volume depuis le fichier .pt local,
- Extrait la slice axiale centrale du tenseur 3D,
- Enregistre la slice au format .npy dans DEFAULT_OUTPUT_DIR,
- Enregistre la slice au format .png dans DEFAULT_PNG_OUTPUT_DIR,
- Constitue un dataset au format .csv pour MR-CLIP.

Stratégie NAS :
  Les fichiers .pt sont copiés en local par batches (DEFAULT_BATCH_SIZE) avant
  traitement. Cela contourne la saturation du cache page et des connexions du NAS
  en concentrant les I/O réseau sur une seule opération rsync/shutil par batch,
  puis en traitant entièrement depuis le disque local.
  Chaque batch est supprimé après traitement pour ne pas saturer le disque local.
"""

import csv
import gc
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# Paramètres (par défault pour le train)

DEFAULT_INPUT_CSV       = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/listing_data_train.csv")
DEFAULT_OUTPUT_DIR      = Path("../data/brain_slices/train/raw")          # slices .npy
DEFAULT_PNG_OUTPUT_DIR  = Path("../data/brain_slices/train/png")          # slices .png
DEFAULT_CSV_OUT         = Path("../data/csv_files/raw_slices_and_json_paths_train.csv")
DEFAULT_Z_RATIO         = 0.5

# Dossier local temporaire pour les copies de fichiers .pt
# Doit être sur un disque **LOCAL** (pas le NAS) 
DEFAULT_LOCAL_CACHE_DIR = Path("/home/benolive/Documents/tmp/pt_cache")

# Taille de batch : nombre de fichiers .pt copiés+traités en une fois.
# Doit rester sous le seuil de ralentissement du NAS (~100 ici).
DEFAULT_BATCH_SIZE      = 80

# Workers pour le traitement CPU/écriture (pas les I/O NAS)
DEFAULT_NUM_WORKERS     = 4

VOLUME_PATH_COLUMN = "volume_path"


# Fonctions 

def copy_batch_from_nas(volume_paths: list[Path], local_cache_dir: Path) -> dict[Path, Path]:
    """
    Copie une liste de fichiers .pt depuis le NAS vers local_cache_dir.
    Retourne un dict {nas_path: local_path}.
    La copie est séquentielle mais groupée : le NAS reçoit une rafale de
    requêtes continues plutôt que des requêtes entrelacées avec du traitement.
    """
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for volume_path in volume_paths:
        local_path = local_cache_dir / volume_path.name
        if not local_path.exists():
            shutil.copy2(volume_path, local_path)
        mapping[volume_path] = local_path
    return mapping


def load_volume_from_pt(volume_path: Path) -> tuple[np.ndarray, str]:
    """
    Charge un fichier .pt contenant un dictionnaire {'volume': Tensor, 'ds_name': str}.
    Retourne le volume sous forme de ndarray float32 et le ds_name associé.
    """
    volume_path = Path(volume_path)
    if volume_path.suffix != ".pt":
        raise ValueError(f"Fichier .pt attendu, obtenu : {volume_path}")

    data = torch.load(volume_path, map_location="cpu", weights_only=True)

    if not isinstance(data, dict):
        raise ValueError(f"Dictionnaire attendu dans le .pt, obtenu : {type(data)}")
    if "volume" not in data:
        raise KeyError(
            f"Clé 'volume' absente dans {volume_path.name}. "
            f"Clés présentes : {list(data.keys())}"
        )

    volume  = data["volume"]
    ds_name = data.get("ds_name", "")

    if isinstance(volume, torch.Tensor):
        volume = volume.numpy()
    volume = volume.astype(np.float32)

    return volume, ds_name


def extract_central_slice(volume: np.ndarray, target_z_ratio: float = 0.5) -> tuple[np.ndarray, int]:
    """Extrait la slice axiale centrale d'un ndarray 3D (X, Y, Z)."""
    if volume.ndim != 3:
        raise ValueError(f"Volume 3D attendu, obtenu shape={volume.shape}")

    X, Y, Z = volume.shape
    z_idx   = int(round(Z * target_z_ratio))
    z_idx   = max(0, min(z_idx, Z - 1))

    return volume[:, :, z_idx].astype(np.float32), z_idx


def save_slice_as_npy(array_2d: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), array_2d)


def save_slice_as_png(array_2d: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = array_2d.min(), array_2d.max()
    normalized = (
        (array_2d - vmin) / (vmax - vmin) * 255.0
        if vmax > vmin
        else np.zeros_like(array_2d)
    )
    Image.fromarray(normalized.astype(np.uint8)).save(str(output_path))


def build_output_paths(volume_path: Path, output_dir: Path, png_output_dir: Path) -> tuple[Path, Path]:
    stem = volume_path.stem
    return output_dir / f"{stem}.npy", png_output_dir / f"{stem}.png"


def process_one_volume(
    row:            dict,
    local_volume_path:  Path,
    output_dir:     Path,
    png_output_dir: Path,
    z_ratio:        float,
    index:          int,
    total:          int,
) -> tuple[dict | None, str | None]:
    """Traite un volume depuis sa copie locale. Appelé depuis un thread."""
    nas_volume_path = Path(row.get(VOLUME_PATH_COLUMN, ""))
    try:
        volume, ds_name = load_volume_from_pt(local_volume_path)
        slice_2d, z_idx = extract_central_slice(volume, target_z_ratio=z_ratio)

        # Les chemins de sortie utilisent le stem NAS original (pas /tmp)
        npy_path, png_path = build_output_paths(nas_volume_path, output_dir, png_output_dir)
        save_slice_as_npy(slice_2d, npy_path)
        save_slice_as_png(slice_2d, png_path)

        print(
            f"[{index}/{total}] {nas_volume_path.name}  "
            f"shape={volume.shape}  z_idx={z_idx}"
        )

        row_out = {
            (k if k != VOLUME_PATH_COLUMN else "slice_path"): v
            for k, v in row.items()
        }
        row_out["slice_path"] = str(npy_path)
        row_out["slice_idx"]  = z_idx
        row_out["png_path"]   = str(png_path)
        return row_out, None

    except Exception as e:
        print(f"[{index}/{total}] [ERREUR] {nas_volume_path.name} : {e}")
        return None, f"{nas_volume_path.name} : {e}"

    finally:
        try:
            del volume, slice_2d
        except NameError:
            pass
        gc.collect()


# Pipeline principale

def process_all_volumes(
    input_csv:       Path,
    output_dir:      Path,
    png_output_dir:  Path,
    csv_out:         Path,
    z_ratio:         float,
    local_cache_dir: Path = DEFAULT_LOCAL_CACHE_DIR,
    batch_size:      int  = DEFAULT_BATCH_SIZE,
    num_workers:     int  = DEFAULT_NUM_WORKERS,
) -> None:
    input_csv      = Path(input_csv)
    output_dir     = Path(output_dir)
    png_output_dir = Path(png_output_dir)
    csv_out        = Path(csv_out)
    local_cache_dir = Path(local_cache_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV d'entrée introuvable : {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_output_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader        = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        rows          = list(reader)

    if not rows:
        raise RuntimeError(f"CSV vide : {input_csv}")
    if VOLUME_PATH_COLUMN not in fieldnames_in:
        raise KeyError(
            f"Colonne '{VOLUME_PATH_COLUMN}' absente du CSV. "
            f"Colonnes disponibles : {fieldnames_in}"
        )

    total = len(rows)
    n_batches = (total + batch_size - 1) // batch_size
    print(f"{total} volume(s) — {n_batches} batch(es) de {batch_size} — {num_workers} workers\n")

    fieldnames_out = list(fieldnames_in)
    idx_pt = fieldnames_out.index(VOLUME_PATH_COLUMN)
    fieldnames_out[idx_pt] = "slice_path"
    for col in ("slice_idx", "png_path"):
        if col not in fieldnames_out:
            fieldnames_out.append(col)

    all_results = [None] * total  # résultats dans l'ordre original

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, total)
        batch_rows = rows[start:end]

        print(f"── Batch {batch_idx + 1}/{n_batches}  "
              f"(volumes {start + 1}–{end}) ──────────────────────────")

        # 1. Copie NAS → local (une seule rafale réseau par batch)
        nas_paths   = [Path(r[VOLUME_PATH_COLUMN]) for r in batch_rows]
        print(f"   Copie de {len(nas_paths)} fichiers depuis le NAS...")
        local_map   = copy_batch_from_nas(nas_paths, local_cache_dir)
        print(f"   Copie terminée. Traitement...\n")

        # 2. Traitement depuis le disque local (rapide, parallélisé)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_local_idx = {
                executor.submit(
                    process_one_volume,
                    row,
                    local_map[Path(row[VOLUME_PATH_COLUMN])],
                    output_dir,
                    png_output_dir,
                    z_ratio,
                    start + local_i + 1,
                    total,
                ): local_i
                for local_i, row in enumerate(batch_rows)
            }
            for future in as_completed(future_to_local_idx):
                local_i = future_to_local_idx[future]
                all_results[start + local_i] = future.result()

        # 3. Nettoyage du cache local après chaque batch
        for local_path in local_map.values():
            try:
                local_path.unlink()
            except FileNotFoundError:
                pass
        print(f"\n   Cache local nettoyé.\n")

    # Écriture du CSV dans l'ordre original
    ok, errors = 0, []
    with open(csv_out, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames_out)
        writer.writeheader()
        for row_out, err in all_results:
            if row_out is not None:
                writer.writerow(row_out)
                ok += 1
            else:
                errors.append(err)

    print(f"\n{'='*60}")
    print(f"Pipeline terminé — {ok}/{total} volumes traités avec succès.")
    print(f"Slices .npy : {output_dir}")
    print(f"Slices .png : {png_output_dir}")
    print(f"CSV         : {csv_out}")

    if errors:
        print(f"\n{len(errors)} erreur(s) :")
        for msg in errors:
            print(f"  • {msg}")



if __name__ == "__main__":
    process_all_volumes(
        input_csv       = DEFAULT_INPUT_CSV,
        output_dir      = DEFAULT_OUTPUT_DIR,
        png_output_dir  = DEFAULT_PNG_OUTPUT_DIR,
        csv_out         = DEFAULT_CSV_OUT,
        z_ratio         = DEFAULT_Z_RATIO,
        local_cache_dir = DEFAULT_LOCAL_CACHE_DIR,
        batch_size      = DEFAULT_BATCH_SIZE,
        num_workers     = DEFAULT_NUM_WORKERS,
    )
