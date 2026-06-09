"""
Pour chaque volume référencé dans le CSV d'entrée (colonne `volume_path`) :
- Charge le volume depuis un fichier .pt (dictionnaire avec clés 'volume' et 'ds_name'),
- Extrait la slice axiale centrale du tenseur 3D,
- Enregistre la slice au format .npy dans DEFAULT_OUTPUT_DIR,
- Enregistre la slice au format .png dans DEFAULT_PNG_OUTPUT_DIR,
- Constitue un dataset au format .csv servant à construire un dataset pour MR-CLIP.
"""

import csv
from pathlib import Path
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# Paramètres (par défault pour le train)

DEFAULT_INPUT_CSV       = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/listing_data_train.csv")
DEFAULT_OUTPUT_DIR      = Path("../data/brain_slices/train/raw")          # slices .npy
DEFAULT_PNG_OUTPUT_DIR  = Path("../data/brain_slices/train/png")          # slices .png
DEFAULT_CSV_OUT         = Path("../data/csv_files/raw_slices_and_json_paths_train.csv")
DEFAULT_Z_RATIO         = 0.5

# Nom de la colonne du CSV d'entrée qui pointe vers les fichiers .pt
VOLUME_PATH_COLUMN = "volume_path"


DEFAULT_NUM_WORKERS = 4


# Fonctions 

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

    # Tensor -> numpy float32
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


def save_slice_as_npy(array_2d: np.ndarray, output_path: Path) -> None:
    """Sauvegarde une slice 2D NumPy au format .npy."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), array_2d)
    #print(f"   - .npy sauvegardé : {output_path}")


def save_slice_as_png(array_2d: np.ndarray, output_path: Path) -> None:
    """
    Sauvegarde une slice 2D NumPy au format .png.
    Normalise les intensités sur [0, 255] avant l'export.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vmin, vmax = array_2d.min(), array_2d.max()
    if vmax > vmin:
        normalized = (array_2d - vmin) / (vmax - vmin) * 255.0
    else:
        normalized = np.zeros_like(array_2d)

    img = Image.fromarray(normalized.astype(np.uint8))
    img.save(str(output_path))
    #print(f"   - .png sauvegardé : {output_path}")

def save_slice_as_rgb_png(array_2d: np.ndarray, output_path: Path) -> None:
    """
    Sauvegarde une slice 2D NumPy au format PNG RGB.
    Normalise les intensités sur [0, 255], puis duplique le canal grayscale
    en 3 canaux RGB : R=G=B.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sécurité : conversion en float et gestion des NaN/inf
    array_2d = np.asarray(array_2d, dtype=np.float32)
    array_2d = np.nan_to_num(array_2d, nan=0.0, posinf=0.0, neginf=0.0)

    vmin, vmax = array_2d.min(), array_2d.max()

    if vmax > vmin:
        normalized = (array_2d - vmin) / (vmax - vmin) * 255.0
    else:
        normalized = np.zeros_like(array_2d)

    gray_uint8 = normalized.astype(np.uint8)

    # Passage de [H, W] à [H, W, 3]
    rgb_uint8 = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    img = Image.fromarray(rgb_uint8, mode="RGB")
    img.save(str(output_path))


def build_output_paths(volume_path: Path, output_dir: Path, png_output_dir: Path) -> tuple[Path, Path]:
    """
    Construit les chemins de sortie .npy et .png en remplaçant l'extension .pt
    par les extensions cibles, en conservant le stem du fichier source.
    """
    stem     = volume_path.stem                          # ex: "aibl_450385"
    npy_path = output_dir     / f"{stem}.npy"
    png_path = png_output_dir / f"{stem}.png"
    return npy_path, png_path

def process_one_volume(
    row:            dict,
    output_dir:     Path,
    png_output_dir: Path,
    z_ratio:        float,
    index:          int,
    total:          int,
) -> tuple[dict | None, str | None]:
    """
    Traite un seul volume (chargement, extraction, sauvegarde).
    Retourne (row_out, None) en cas de succès, (None, message_erreur) sinon.
    Conçu pour être appelé depuis un thread.
    """
    pt_path = Path(row.get(VOLUME_PATH_COLUMN, ""))
    try:
        volume, ds_name = load_volume_from_pt(pt_path)
        slice_2d, z_idx = extract_central_slice(volume, target_z_ratio=z_ratio)
        npy_path, png_path = build_output_paths(pt_path, output_dir, png_output_dir)
 
        save_slice_as_npy(slice_2d, npy_path)
        #save_slice_as_png(slice_2d, png_path)
        save_slice_as_rgb_png(slice_2d, png_path) # on utilise checkpoint mr-clip adapté au rgb, donc adapter le format d'enregistrement
 
        print(
            f"[{index}/{total}] {pt_path.name}  "
            f"shape={volume.shape}  z_idx={z_idx}  "
            f"min={slice_2d.min():.1f}  max={slice_2d.max():.1f}"
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
        print(f"[{index}/{total}] [ERREUR] {pt_path.name} : {e}")
        return None, f"{pt_path.name} : {e}"
 
    finally:
        try:
            del volume, slice_2d
        except NameError:
            pass
        gc.collect()



# Pipeline principale

def process_all_volumes(
    input_csv:      Path,
    output_dir:     Path,
    png_output_dir: Path,
    csv_out:        Path,
    z_ratio:        float,
    num_workers:    int = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Itère sur toutes les lignes du CSV d'entrée, charge chaque volume .pt,
    extrait la slice centrale, la sauvegarde en .npy et .png, et écrit le CSV de slices.
    """
    input_csv      = Path(input_csv)
    output_dir     = Path(output_dir)
    png_output_dir = Path(png_output_dir)
    csv_out        = Path(csv_out)

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV d'entrée introuvable : {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    png_output_dir.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Lecture du CSV d'entrée
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader        = csv.DictReader(f)
        fieldnames_in = reader.fieldnames or []
        rows          = list(reader)

    if not rows:
        raise RuntimeError(f"CSV vide : {input_csv}")

    # Vérification de la colonne attendue
    if VOLUME_PATH_COLUMN not in fieldnames_in:
        raise KeyError(
            f"Colonne '{VOLUME_PATH_COLUMN}' absente du CSV. "
            f"Colonnes disponibles : {fieldnames_in}"
        )

    total = len(rows)
    print(f"{total} volume(s) trouvé(s) dans {input_csv.name}")
    print(f"Parallélisation : {num_workers} workers\n")


    # Colonnes de sortie : identiques à l'entrée, volume_path remplacé par slice_path + slice_idx + png_path
    fieldnames_out = list(fieldnames_in)
    # Remplace la colonne volume_path par slice_path (chemin .npy)
    idx_pt = fieldnames_out.index(VOLUME_PATH_COLUMN)
    fieldnames_out[idx_pt] = "slice_path"
    for extra_col in ("slice_idx", "png_path"):
        if extra_col not in fieldnames_out:
            fieldnames_out.append(extra_col)

    # results[i] contiendra (row_out | None, erreur | None) dans l'ordre original
    results = [None] * total

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(
                process_one_volume,
                row, output_dir, png_output_dir, z_ratio, i + 1, total,
            ): i
            for i, row in enumerate(rows)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()
 
    # Écriture du CSV dans l'ordre original
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
 
    # Résumé
    print(f"\n{'='*60}")
    print(f"Pipeline terminé — {ok}/{total} volumes traités avec succès.")
    print(f"Slices .npy enregistrées dans : {output_dir}")
    print(f"Slices .png enregistrées dans : {png_output_dir}")
    print(f"CSV de slices                 : {csv_out}")
 
    if errors:
        print(f"\n{len(errors)} erreur(s) :")
        for msg in errors:
            print(f"  {msg}")



if __name__ == "__main__":
    process_all_volumes(
        input_csv      = DEFAULT_INPUT_CSV,
        output_dir     = DEFAULT_OUTPUT_DIR,
        png_output_dir = DEFAULT_PNG_OUTPUT_DIR,
        csv_out        = DEFAULT_CSV_OUT,
        z_ratio        = DEFAULT_Z_RATIO,
    )
