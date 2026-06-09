"""
Fichier main du précompute :
- Appelle extract_slices.py pour extraire l'ensemble des slices à partir des path entré en dur 
- Appelle ensuite beta_encoder_runner.py pour calculer les anatomies de chaque donné et les enregistre dans le dossier data/brain_slices/encoded
- Enfin appelle build_mrclip_dataset.py pour construir le csv nécessaire à l'utilisation de Mr-CLIP dans le repo MARAI.


Attention : 
Trois autres étapes du pré-processing ne sont pas pris en compte ici :
    1) Listing des volumes ainsi que leur metadonnées dicom associées (préliminaires géré par listing_data.py)
    2) Faire tourner MR-CLIP et produire les embeddings associés à chaque slice
La deuxième dernière étapes étapes sont gérées par l'utilisation de MR-CLIP dans le repôsitory MarAI_2. Ces deux étapes doivent être exécutées après l'exécution 
de ce fichier.


Les paramètres sont écrits en dur ci-dessous.
Passer --skip-extract pour sauter l'étape 1 si les slices existent déjà.
"""


import argparse
import sys
from pathlib import Path

# Résolution du dossier racine du projet (parent de "precompute" et "models")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from extract_slices import process_all_volumes
from beta_encoder_runner import iter_npy_slices, apply_beta_encoder
from build_mrclip_dataset import generate_mrclip_csv

# Import depuis models/ (dossier frère de precompute/)
from models.beta_encoder import load_beta_encoder

import torch
from tqdm import tqdm


# ================================ PARAMÈTRES ===============================

# -------------- Version train --------------
'''
SUFFIX = "_train"   # suffixe commun à tous les fichiers de sortie nommés

# 1) extract_slices 
EXTRACT_INPUT_CSV      = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/listing_data_train.csv")
EXTRACT_OUTPUT_DIR     = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/train/raw")
EXTRACT_PNG_OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/train/png")
EXTRACT_CSV_OUT        = Path(f"/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/raw_slices_and_json_paths{SUFFIX}.csv")
EXTRACT_Z_RATIO        = 0.5
EXTRACT_NUM_WORKERS    = 4

# 2) beta_encoder_runner
ENCODER_FOLDER_RAW = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/train/raw")     # = EXTRACT_OUTPUT_DIR
ENCODER_OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/train/encoded")
ENCODER_CKPT       = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/models/anatomy_encoder.pt")

# 3) build_mrclip_dataset
MRCLIP_INPUT_DIR  = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/train/png")      # = EXTRACT_PNG_OUTPUT_DIR
MRCLIP_OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/prerequis_MRCLIP") 
MRCLIP_DEFAULT_TEXT = (
    "A brain MRI, plane NONE, "
    "Scanner (Manufacturer, Model, Field Strength): (NONE, NONE, NONE), "
    "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), "
    "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): "
    "(NONE, NONE, NONE, NONE)"
)
'''



# -------------- Version test --------------
#'''
SUFFIX = "_test"   # suffixe commun à tous les fichiers de sortie nommés

# 1) extract_slices 
EXTRACT_INPUT_CSV      = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/listing_data_test.csv")
EXTRACT_OUTPUT_DIR     = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/test/raw")
EXTRACT_PNG_OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/test/png")
EXTRACT_CSV_OUT        = Path(f"/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/raw_slices_and_json_paths{SUFFIX}.csv")
EXTRACT_Z_RATIO        = 0.5
EXTRACT_NUM_WORKERS    = 4

# 2) beta_encoder_runner
ENCODER_FOLDER_RAW = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/test/raw")    # = EXTRACT_OUTPUT_DIR
ENCODER_OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/test/encoded")
ENCODER_CKPT       = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/models/anatomy_encoder.pt")

# 3) build_mrclip_dataset
MRCLIP_INPUT_DIR  = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/test/png")      # = EXTRACT_PNG_OUTPUT_DIR
MRCLIP_OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files/prerequis_MRCLIP") 
MRCLIP_DEFAULT_TEXT = (
    "A brain MRI, plane NONE, "
    "Scanner (Manufacturer, Model, Field Strength): (NONE, NONE, NONE), "
    "Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), "
    "Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): "
    "(NONE, NONE, NONE, NONE)"
)
#'''



# ===========================================================================


def step_extract_slices() -> None:
    """Étape 1 - extraction des slices depuis les volumes .pt."""
    print("\n" + "=" * 60)
    print("ÉTAPE 1 - Extraction des slices")
    print("=" * 60)
    process_all_volumes(
        input_csv=EXTRACT_INPUT_CSV,
        output_dir=EXTRACT_OUTPUT_DIR,
        png_output_dir=EXTRACT_PNG_OUTPUT_DIR,
        csv_out=EXTRACT_CSV_OUT,
        z_ratio=EXTRACT_Z_RATIO,
        num_workers=EXTRACT_NUM_WORKERS,
    )


def step_beta_encoder() -> None:
    """Étape 2 - encodage anatomique des slices via le beta-encoder."""
    print("\n" + "=" * 60)
    print("ÉTAPE 2 - Encodage anatomique (beta-encoder)")
    print("=" * 60)

    ENCODER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    beta_encoder = load_beta_encoder(ENCODER_CKPT, device)

    total = sum(1 for _ in ENCODER_FOLDER_RAW.glob("*.npy"))
    for item in tqdm(iter_npy_slices(ENCODER_FOLDER_RAW), desc="Encodage beta-encoder", unit="slice", total=total):
        apply_beta_encoder(
            file=item["file"],
            path=item["path"],
            slice=item["slice"],
            beta_encoder=beta_encoder,
            device=device,
            output_dir=ENCODER_OUTPUT_DIR,
        )


def step_build_mrclip_dataset() -> None:
    """Étape 3 - construction du CSV MR-CLIP."""
    print("\n" + "=" * 60)
    print("ÉTAPE 3 -  Construction du dataset MR-CLIP")
    print("=" * 60)

    csv_path = generate_mrclip_csv(
        input_dir=MRCLIP_INPUT_DIR,
        output_dir=MRCLIP_OUTPUT_DIR,
        suffix=SUFFIX,
        default_text=MRCLIP_DEFAULT_TEXT,
    )
    print(f"CSV MR-CLIP enregistré : {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de précompute : extract -> encode -> mrclip-csv"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Sauter l'étape 1 (extract_slices) si les slices existent déjà.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_extract:
        step_extract_slices()
    else:
        print("\n[INFO] Étape 1 (extract_slices) ignorée (--skip-extract activé).")


    step_beta_encoder()
    step_build_mrclip_dataset()

    print("\n" + "=" * 60)
    print("Pipeline de précompute terminé avec succès.")
    print("=" * 60)


if __name__ == "__main__":
    main()
