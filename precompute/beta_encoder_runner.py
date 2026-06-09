"""
Charge le beta_encoder puis créé les anatomies à partir des slices se trouvant dans le raw. 
"""


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn.functional as F

from models.beta_encoder import load_beta_encoder

def iter_npy_slices(folder_path, axis=2, dtype=np.float32):
    """
    Itère sur tous les fichiers .npy du dossier d'input et yield les slices au format numpy.

    Chaque fichier .npy est supposé contenir une slice 2D.
    """

    folder_path = Path(folder_path)

    if not folder_path.is_dir():
        raise ValueError(f"Le chemin donné n'est pas un dossier : {folder_path}")
    
    npy_files = sorted(folder_path.glob("*.npy"))
    print(f"[INFO] Dossier raw : {folder_path}")
    print(f"[INFO] Nombre de fichiers .npy trouvés : {len(npy_files)}")

    print(f"[INFO] Dossier raw : {folder_path}")
    print(f"[INFO] Nombre de fichiers .npy trouvés : {len(npy_files)}")

    for npy_file in npy_files:
        slice_np = np.load(npy_file).astype(dtype)

        yield {
            "file": npy_file.stem,
            "path": npy_file,
            "slice": np.asarray(slice_np),
        }



def apply_beta_encoder(file, path, slice, beta_encoder, device, output_dir):
    #print("File :", file)
    #print("Path :", path)
    #print("Slice shape :", slice.shape)

    # ENCODED : passage dans le UNet (ToDO) (inspiré de precompute_slices.py)
    mask_2d     = (slice != 0)
    masked      = (slice * mask_2d).astype(np.float32)       # Fonctionnement en attendant d'avoir les données avec mask

    tensor_in, pad_info = pad_to_multiple(
        torch.from_numpy(masked).unsqueeze(0).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        tensor_out = beta_encoder(tensor_in)
    result = unpad(tensor_out, pad_info).squeeze().cpu().numpy().astype(np.float32)
    result = (result * mask_2d).astype(np.float32)

    out_encoded = output_dir / f"{file}.npy"
    np.save(out_encoded, result)




# --- outils ---
def pad_to_multiple(tensor, multiple=16):
    _, _, H, W = tensor.shape
    H_pad = (multiple - H % multiple) % multiple
    W_pad = (multiple - W % multiple) % multiple
    pad_top, pad_bottom = H_pad // 2, H_pad - H_pad // 2
    pad_left, pad_right = W_pad // 2, W_pad - W_pad // 2
    padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return padded, (pad_top, pad_left, H, W)

def unpad(tensor, pad_info):
    pad_top, pad_left, H, W = pad_info
    return tensor[:, :, pad_top:pad_top + H, pad_left:pad_left + W]






if __name__ == "__main__":
    folder_path_raw = "/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/train/raw"
    output_dir = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/data/brain_slices/train/encoded")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    encoder_ckpt = Path("/NAS/coolio/benolive/Diffusion_beta_encoder_2D/models/anatomy_encoder.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    beta_encoder = load_beta_encoder(encoder_ckpt, device)


    for item in tqdm(iter_npy_slices(folder_path_raw), desc="Encodage des slices"):
        apply_beta_encoder(file=item["file"], path=item["path"], slice=item["slice"], beta_encoder=beta_encoder, device=device, output_dir=output_dir)

    

