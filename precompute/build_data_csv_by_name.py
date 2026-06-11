"""
build_data_by_name_csv.py

Construit un fichier CSV associant chaque fichier .npy brut à son équivalent encodé.

CSV généré :
    name, raw_path, encoded_path
"""

from pathlib import Path
import csv


def build_data_by_name_csv(source_raw, source_encoded, output_path, suffixe = "", check_encoded_exists = False):
    """
    Parcourt source_raw et crée un CSV qui relie chaque fichier .npy brut
    à son fichier encodé correspondant dans source_encoded.
    """

    source_raw = Path(source_raw)
    source_encoded = Path(source_encoded)
    output_path = Path(output_path)

    if not source_raw.is_dir():
        raise NotADirectoryError(f"source_raw n'est pas un dossier valide : {source_raw}")

    if not source_encoded.is_dir():
        raise NotADirectoryError(f"source_encoded n'est pas un dossier valide : {source_encoded}")

    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"data_by_name{suffixe}.csv"

    raw_files = sorted(source_raw.glob("*.npy"))

    with csv_path.open(mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["name", "raw_path", "encoded_path"])

        for raw_file in raw_files:
            name = raw_file.stem

            encoded_file = source_encoded / raw_file.name

            if check_encoded_exists and not encoded_file.exists():
                raise FileNotFoundError(
                    f"Fichier encodé manquant pour {raw_file.name} : {encoded_file}"
                )

            writer.writerow([
                name,
                str(raw_file),
                str(encoded_file),
            ])

    return csv_path


if __name__ == "__main__":
    # Utilisation train:
    csv_file = build_data_by_name_csv(
        source_raw="/NAS/coolio/benolive/Diffusion_beta_encoder/data/brain_slices/train/raw",
        source_encoded="/NAS/coolio/benolive/Diffusion_beta_encoder/data/brain_slices/train/encoded",
        output_path="/NAS/coolio/benolive/Diffusion_beta_encoder/data/csv_files/diffusion_data",
        suffixe="_train",
        check_encoded_exists=False,
    )

    print(f"CSV généré : {csv_file}")
