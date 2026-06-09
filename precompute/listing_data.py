import os
import csv
import torch

from tqdm import tqdm
 
 
def generate_listing_csv(data_dir_path: str, output_csv_path: str, suffixe: str | None = None) -> str:
    """
    Parcourt un dossier à la recherche de fichiers .pt et génère un fichier listing_data.csv.
    """
    rows = []

    print(f"Début du listing des fichiers .pt dans : {data_dir_path}")

    for filename in os.listdir(data_dir_path):
        if filename.endswith(".pt"):
            volume_path = os.path.join(data_dir_path, filename)

            file_name = filename

            rows.append({
                "file_name": file_name,
                "volume_path": volume_path,
            })

    if suffixe is not None:
        csv_filename = f"listing_data{suffixe}.csv"
    else:
        csv_filename = "listing_data.csv"

    os.makedirs(output_csv_path, exist_ok=True)
    output_file = os.path.join(output_csv_path, csv_filename)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "volume_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"listing_data.csv généré avec {len(rows)} entrée(s) -> {output_file}")
    print()

    return output_file
 
 
# === Appel direct du script ===
if __name__ == "__main__":
    # Version train
    print("Computing train set")
    DATA_DIR_PATH = "/NAS/coolio/Barnabe/CODES/diffusion_classifier_guidance/iguane_pt_train_dataset"
    OUTPUT_CSV_PATH = "/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/csv_files"
 
    generate_listing_csv(DATA_DIR_PATH, OUTPUT_CSV_PATH, suffixe="_train")
    

    # Version test
    print("Computing test set")
    DATA_DIR_PATH_TEST = "/NAS/coolio/Barnabe/CODES/diffusion_classifier_guidance/iguane_pt_test_dataset"
 
    generate_listing_csv(DATA_DIR_PATH_TEST, OUTPUT_CSV_PATH, suffixe="_test")