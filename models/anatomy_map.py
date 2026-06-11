"""
Anatomy extractor.

Two possible anatomy map :
- naive edge map
- loading encoded slices

"""

import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def _gaussian_kernel_2d(sigma: float, device: torch.device, dtype=torch.float32):
    """ Noyeau Gaussien 2D (1, 1, H, W) """
    if sigma <= 0:
        k = torch.zeros((1,1,1,1), device=device, dtype=dtype)
        k[0,0,0,0] = 1.0
        return k
    radius = int(max(1, torch.ceil(3 * torch.tensor(sigma)).item()))
    coords = torch.arange(-radius, radius+1, device=device, dtype=dtype)
    g1d = torch.exp(-(coords**2) / (2 * sigma**2))
    g1d = g1d / g1d.sum()
    g2d = g1d[:, None] * g1d[None, :]          # outer product -> (2r+1, 2r+1)    
    return g2d.unsqueeze(0).unsqueeze(0)       # (1, 1, H, W)

def _pad_for_conv2d(kernel):
    kH, kW = kernel.shape[-2:]
    return (kW // 2, kW // 2, kH // 2, kH // 2)

def _conv2d_same(x, kernel):
    pad = _pad_for_conv2d(kernel)
    x_p = F.pad(x, pad, mode='replicate')
    return F.conv2d(x_p, kernel)


def gaussian_blur2d(x, sigma: float):
    if sigma <= 0:
        return x
    kernel = _gaussian_kernel_2d(sigma, device=x.device, dtype=x.dtype)
    return _conv2d_same(x, kernel)


# Gradient
def _sobel_kernels_2d(device, dtype=torch.float32):
    kx = torch.zeros((1,1,3,3), device=device, dtype=dtype)
    ky = torch.zeros_like(kx)
    # simple central difference kernels
    kx[0, 0, 1, 0] = -1.0;  kx[0, 0, 1, 2] = 1.0   # différence centrale horizontal
    ky[0, 0, 0, 1] = -1.0;  ky[0, 0, 2, 1] = 1.0   # différence centrale vertical
    return kx, ky

def gradient_magnitude_torch_2d(x: torch.Tensor, sigma: float = 0.0) -> torch.Tensor:
    """
    Magnitude du gradient en 2D,
    x: (B, 1, H, W)
    """
    if sigma > 0:
        x = gaussian_blur2d(x, sigma)
    kx, ky = _sobel_kernels_2d(x.device, dtype=x.dtype)
    pad = _pad_for_conv2d(kx)
    xpad = F.pad(x, pad, mode='replicate')
    dx = F.conv2d(xpad, kx)
    dy = F.conv2d(xpad, ky)
    return torch.sqrt(dx * dx + dy * dy + 1e-12)


def _global_quantile(tensor, q, max_samples = 2_000_000):
    """
    Compute approximate quantile q of `tensor` without flattening the entire huge tensor.
    If the total number of elements <= max_samples we compute exact quantile,
    otherwise we take a strided subsample (deterministic, GPU-friendly) and compute quantile on it.
    q: float in [0,1]
    """
    flat = tensor.view(-1)
    n = flat.numel()
    if n <= max_samples:
        return torch.quantile(flat, q)
    # deterministic strided subsample to limit memory
    step = int(n // max_samples) + 1
    sample = flat[::step]
    return torch.quantile(sample, q)


# main function : Canny edge like anatomy extractor
def make_structural_anatomy_map_2d(
    batch_imgs: torch.Tensor,
    grad_sigmas: tuple = (0.5, 2.0),
    hf_sigma: float = 1.0,
    smooth_sigma: float = 1.0,
    normalize_percentiles: tuple = (1.0, 99.0),
    ):
    """
    Crates a 2D anatomical map (1 canal) same dimension as the input.

    Args:
        batch_imgs:            (B, 1, H, W)  - axiales slices
        grad_sigmas:           sigmas for multi-scale gradient
        hf_sigma:              sigma for high frequency map |I - Gσ(I)|
        smooth_sigma:          sigma final smoothing
        normalize_percentiles: percentiles for rubust normalisation -> [-1, 1]

    Returns:
        anatomy_map: (B, 1, H, W), torch.Tensor
    """
    assert batch_imgs.ndim == 4 and batch_imgs.shape[1] == 1    # Attendu : (B, 1, H, W)

    # --- 1) Gradients multi-échelles
    g1 = gradient_magnitude_torch_2d(batch_imgs, sigma=grad_sigmas[0])  # Contours fins
    g2 = gradient_magnitude_torch_2d(batch_imgs, sigma=grad_sigmas[1])  # formes générales

    # --- 2) Carte haute fréquence
    blurred = gaussian_blur2d(batch_imgs, sigma=hf_sigma)
    hf = torch.abs(batch_imgs - blurred)

    # 3) Combinaison pondérée (mêmes poids qu'en 3D)
    combined = 0.5 * g1 + 0.3 * g2 + 0.2 * hf

    # --- 4) Normalisation robuste [-1,1] via percentiles globaux
    p1, p99 = normalize_percentiles
    lo = _global_quantile(combined, p1/100.0)
    hi = _global_quantile(combined, p99/100.0)
    normed = (combined - lo) / (hi - lo + 1e-6)
    normed = normed.clamp(0,1) * 2 - 1  # -> [-1,1]

    # --- 5) Lissage optionnel pour continuité anatomique
    if smooth_sigma > 0:
        normed = gaussian_blur2d(normed, sigma=smooth_sigma)

    return normed


# Anatomy beta encoder
def load_beta_encoded_anatomy(csv_path, slice_names):
    """
    Charge les slices encodées (anatomie pré-calculée par l'anatomy mapper de DISt-CLIP) depuis un CSV.

    Args:
        csv_path:    chemin vers le .csv (colonnes: name, raw_path, encoded_path)
        slice_names: liste des noms de slices à charger (colonne 'name')
    Returns:
        anatomy_map: (B, 1, H, W) torch.Tensor, une entrée par slice indiquée par slice_names
    """
    df = pd.read_csv(csv_path)
    assert {"name", "raw_path", "encoded_path"}.issubset(df.columns)

    slices = []
    for name in slice_names:
        row = df[df["name"] == name]
        assert len(row) == 1, f"Slice '{name}' : {len(row)} correspondance(s) trouvée(s) dans le CSV"

        encoded_path = Path(row["encoded_path"].values[0])
        assert encoded_path.exists(), f"Fichier introuvable : {encoded_path}"

        arr = np.load(encoded_path)
        assert arr.ndim == 2, f"On attend shape : (H, W) pour '{name}', reçu {arr.shape}"
        slices.append(torch.from_numpy(arr).float().unsqueeze(0))  # (1, H, W)

    return torch.stack(slices, dim=0)




if __name__ == "__main__":
    # test d'utilisation :

    # ====== Paramètres ======

    # canny edge like
    SLICE_PATH = "/NAS/coolio/benolive/Diffusion_beta_encoder/data/brain_slices/train/raw/oas-trio_53Sd0428R2.npy"
    OUTPUT_DIR  = "/NAS/coolio/benolive/Diffusion_beta_encoder/tests/anatomy_map_tests"

    # beta encoder
    CSV_PATH    = "/NAS/coolio/benolive/Diffusion_beta_encoder/data/csv_files/diffusion_data/data_by_name_train.csv"
    SLICE_NAMES = ["oas-trio_53Sd0428R2"]          # liste ou nom unique
    # ========================

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_np = np.load(SLICE_PATH)
    assert slice_np.ndim == 2, f"Attendu (H, W), reçu {slice_np.shape}"
    print(f"[LOAD] shape: {slice_np.shape}, dtype: {slice_np.dtype}")

    tensor = torch.from_numpy(slice_np).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        anatomy = make_structural_anatomy_map_2d(tensor)   # (1,1,H,W)

    slice_out   = tensor[0, 0].numpy()
    anatomy_out = anatomy[0, 0].numpy()


    # test beta encoder
    anatomy_beta_encoded = load_beta_encoded_anatomy(CSV_PATH, SLICE_NAMES)  
    assert anatomy_beta_encoded.ndim == 4, (
        f"Attendu anatomy_beta_encoded shape (B, 1, H, W), reçu {anatomy_beta_encoded.shape}"
    )
    anatomy_beta_encoded_out = anatomy_beta_encoded[0, 0].detach().cpu().numpy()

    
    # sauvegardes
    np.save(output_dir / "slice_input.npy",   slice_out)
    np.save(output_dir / "anatomy_map.npy",   anatomy_out)
    np.save(output_dir / "anatomy_beta_encoded.npy", anatomy_beta_encoded_out)
    print(f"[save] .npy sauvegardés dans {output_dir}")



    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(slice_out, cmap="gray", vmin=-1, vmax=1)
    axes[0].set_title("Slice brute\nnormalisée")
    axes[0].axis("off")

    im1 = axes[1].imshow(anatomy_out, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Carte anatomique\nstructurelle")
    axes[1].axis("off")

    im2 = axes[2].imshow(anatomy_beta_encoded_out, cmap="coolwarm", vmin=-1, vmax=1)
    axes[2].set_title("Carte anatomique\nbeta-encoded")
    axes[2].axis("off")

    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()

    fig_path = output_dir / "anatomy_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"[fig] comparaison sauvegardée dans {fig_path}")