"""
Visualisation des générations du modèle de diffusion (print des slices)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def normalize_for_display(im: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """
    Normalisation percentile (p1, p99) d'une image 2D numpy -> [0, 1].
    Identique au notebook - vis dans le training
 
    Args:
        im:     array 2D (H, W)
        p_low:  percentile bas
        p_high: percentile haut
    Returns:
        array 2D normalisé dans [0, 1], ou im inchangé si p_low == p_high
    """
    p1, p99 = np.percentile(im, [p_low, p_high])
    if p1 == p99:
        return im
    im_clipped = np.clip(im, p1, p99)
    return (im_clipped - p1) / (p99 - p1 + 1e-5)
 
 
def plot_eval_batch(slice_origine, anatomy_map, generated_slice, epoch, sample_idx = 0, mse=None):
    """
    Produit une figure matplotlib 3*1 pour suivre visuellement l'entraînement

 
    Args:
        slice_origine:  (B, 1, H, W) = slices originales du batch eval
        anatomy_map:       (B, 1, H, W) = cartes anatomiques
        generated_slice:      (B, 1, H, W) = slices générées par DDIM
        sample_idx:     index dans le batch (par default à 0)
        mse:            Valeur MSE sur test set
 
    Returns:
        figure (à logger dans TensorBoard ou sauvegarder)
    """
    def to_np(t):
        """ Tenseur shape : (B, 1, H, W) -> (H, W) numpy"""
        return t[sample_idx, 0].detach().cpu().float().numpy()
 
    imgs = [to_np(slice_origine), to_np(anatomy_map), to_np(generated_slice)]
    row_titles = ["Original", "Anat map", "DDIM output"]
 
    fig, axes = plt.subplots(3, 1, figsize=(5, 12))
    mse_str = f" | MSE : {mse:.6f}" if mse is not None else ""
    fig.suptitle(f"Epoch {epoch}{mse_str}", fontsize=13, fontweight="bold")
    plt.subplots_adjust(hspace=0.3)
 
    for r, (im, title) in enumerate(zip(imgs, row_titles)):
        ax = axes[r]
        im_show = np.rot90(im)  # pour faciliter lecture
        im_norm = normalize_for_display(im_show)
        ax.imshow(im_norm, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
 
    return fig
 
