'''
Scan le dossier où se trouvent les .png et produis le fichier .csv utilisable par MR-CLIP 2D
text : default text


Le CSV produit contient 3 colonnes :
- filepath : chemin du fichier .png
- text     : texte constant donné par DEFAULT_TEXT
- label     : tout le temps 0 pour le moment
'''


from pathlib import Path
import csv



def generate_mrclip_csv(input_dir, output_dir, suffix, default_text):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Le dossier input_dir n'existe pas : {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir n'est pas un dossier : {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"mrclip_dataset{suffix}.csv"

    png_files = sorted(input_dir.glob("*.png"))

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filepath", "text", "label"],
        )

        writer.writeheader()

        for png_file in png_files:
            writer.writerow(
                {
                    "filepath": str(png_file),
                    "text": default_text,
                    "label": 0,
                }
            )

    return csv_path



if __name__ == "__main__":
    INPUT_DIR = "/NAS/coolio/benolive/Diffusion_beta_encoder/data/brain_slices/train/png"
    OUTPUT_DIR = "/NAS/coolio/benolive/Diffusion_beta_encoder/data/csv_files/prerequis_MRCLIP"
    DEFAULT_TEXT = "A brain MRI, plane NONE, Scanner (Manufacturer, Model, Field Strength): (NONE, NONE, NONE), Acquisition (Description, Sequence, Variant): (NONE, NONE, NONE), Imaging Parameters (Echo Time, Repetition Time, Inversion Time, Flip Angle): (NONE, NONE, NONE, NONE)"
    SUFFIX = "_train"

    csv_path = generate_mrclip_csv(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        suffix=SUFFIX,
        default_text=DEFAULT_TEXT,
    )

    print(f"CSV généré : {csv_path}")
