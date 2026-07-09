import numpy as np
from pathlib import Path
from collections import defaultdict
import re
from scipy.ndimage import uniform_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import math 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def group_npy_by_subject(folder_path):
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Dossier introuvable : {folder_path}")

    subject_files = defaultdict(list)
    pattern = "*.npy"

    for file_path in sorted(folder_path.glob("*.npy")):
        filename = file_path.name

        match = re.match(r"^(sub\d+)", filename)

        if match:
            subject = match.group(1)
            subject_files[subject].append(file_path)

    return dict(subject_files)


def generate_pairs(liste):
    n = len(liste)

    for i in range(n):
        for j in range(i + 1, n):
            yield liste[i], liste[j]


def load_npy_2d(path, dtype=np.float32):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    image = np.load(path)
    image = np.squeeze(image) # au cas ou dimension égale à 1

    if image.ndim != 2:
        raise ValueError(
            f"L'image n'est pas 2D après squeeze. Shape obtenue : {image.shape}"
        )

    if not np.all(np.isfinite(image)):
        raise ValueError(f"Valeurs non finies (NaN/Inf) détectées dans {path.name}")

    return image.astype(dtype)

def compute_global_data_range(paths_list, p_low=1.0, p_high=99.0):
    """
    Calculé au début sur tout le dataset. Sert de ref pour le PSNR
    """
    all_lows, all_highs = [], []
    for path in paths_list:
        img = load_npy_2d(path)
        nz = img[img > 0]
        if len(nz) == 0:
            continue
        all_lows.append(np.percentile(nz, p_low))
        all_highs.append(np.percentile(nz, p_high))

    if not all_lows:
        raise ValueError("Aucun pixel non-nul trouvé dans le dataset.")

    global_low = min(all_lows)
    global_high = max(all_highs)
    R = global_high - global_low
    if R <= 0:
        raise ValueError(f"Dynamique invalide : R={R}.")
    return float(R), float(global_low), float(global_high)


def robust_pair_data_range(img1, img2, p_low=1.0, p_high=99.0):
    ref1 = img1[img1 > 0]
    ref2 = img2[img2 > 0]
    if len(ref1) == 0 or len(ref2) == 0:
        raise ValueError("Une des deux images n'a aucun pixel non-nul (masque vide ?).")

    p_lo = min(np.percentile(ref1, p_low), np.percentile(ref2, p_low))
    p_hi = max(np.percentile(ref1, p_high), np.percentile(ref2, p_high))
    dr = p_hi - p_lo

    if dr <= 0:
        raise ValueError(f"data_range invalide (paire) : {dr}")
    return float(dr)



def compute_PSNR(img1, img2, R):
    # R a été déterminé sur la dataset
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float("inf")
    return 10*math.log10((R**2)/mse)



def run_experiment(input_dir, label):
    """
    Traite un dossier de slices .npy et retourne un dict de résultats, ou None si le dossier n'existe pas ou vide.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"[{label}] Dossier inexistant ignoré : {input_dir}")
        return None

    dic_paths_per_subject = group_npy_by_subject(input_dir)
    if not dic_paths_per_subject:
        print(f"[{label}] Aucun fichier sub*.npy trouvé, ignoré : {input_dir}")
        return None

    all_paths = [p for path_list in dic_paths_per_subject.values() for p in path_list]

    try:
        R_GLOBAL, global_low, global_high = compute_global_data_range(all_paths)
    except ValueError as e:
        print(f"[{label}] Impossible de calculer R global ({e}), ignoré.")
        return None

    print(f"[{label}] R global (percentile 1-99, fond exclu) = {R_GLOBAL:.4f} "
          f"(low={global_low:.4f}, high={global_high:.4f})")

    ssim_mean_per_subject = {}
    psnr_mean_per_subject = {}

    for subject, path_list in dic_paths_per_subject.items():
        n = len(path_list)
        if n < 2:
            print(f"  [{label}] {subject} : pas assez d'images (n={n}), ignoré.")
            ssim_mean_per_subject[subject] = None
            psnr_mean_per_subject[subject] = None
            continue

        sum_ssim = 0.0
        sum_psnr = 0.0
        n_pairs = 0

        for path1, path2 in generate_pairs(path_list):
            img1 = load_npy_2d(path1)
            img2 = load_npy_2d(path2)

            if img1.shape != img2.shape:
                raise ValueError(
                    f"[{label}] Shapes incompatibles : {path1.name} {img1.shape} alors que {path2.name} {img2.shape}"
                )

            pair_data_range = robust_pair_data_range(img1, img2)
            ssim_value, _ = ssim(img1, img2, win_size=7, data_range=pair_data_range, full=True)
            psnr_value = compute_PSNR(img1, img2, R=R_GLOBAL)

            sum_ssim += ssim_value
            sum_psnr += psnr_value
            n_pairs += 1

        ssim_mean_per_subject[subject] = sum_ssim / n_pairs
        psnr_mean_per_subject[subject] = sum_psnr / n_pairs

    valid_ssim = [v for v in ssim_mean_per_subject.values() if v is not None]
    valid_psnr = [v for v in psnr_mean_per_subject.values() if v is not None]

    if not valid_ssim or not valid_psnr:
        print(f"[{label}] Aucun sujet exploitable (toutes les valeurs sont None), ignoré.")
        return None

    total_mean_ssim = sum(valid_ssim) / len(valid_ssim)
    total_mean_psnr = sum(valid_psnr) / len(valid_psnr)

    print(f"[{label}] SSIM moyenne globale = {total_mean_ssim:.4f}")
    print(f"[{label}] PSNR moyenne globale = {total_mean_psnr:.4f}")
    print("---")

    return {
        "ssim_per_subject": ssim_mean_per_subject,
        "psnr_per_subject": psnr_mean_per_subject,
        "total_mean_ssim": total_mean_ssim,
        "total_mean_psnr": total_mean_psnr,
        "n_subjects": len(valid_ssim),
    }

def plot_comparison(results_per_method, output_path):
    """
    results_per_method : dict {label: résultat de run_experiment (ou absent si None)}
    Ne trace que les méthodes réellement disponibles (naive, encoded et/ou reverse-encoded).
    """
    available = {k: v for k, v in results_per_method.items() if v is not None}

    if not available:
        print("Pas de méthode disponible")
        return

    labels = list(available.keys())
    ssim_values = [available[l]["total_mean_ssim"] for l in labels]
    psnr_values = [available[l]["total_mean_psnr"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = plt.cm.tab10.colors[: len(labels)]

    axes[0].bar(labels, ssim_values, color=colors)
    axes[0].set_title("SSIM moyenne par méthode")
    axes[0].set_ylabel("SSIM")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(ssim_values):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

    axes[1].bar(labels, psnr_values, color=colors)
    axes[1].set_title("PSNR moyen par méthode")
    axes[1].set_ylabel("PSNR (dB)")
    for i, v in enumerate(psnr_values):
        axes[1].text(i, v + 0.3, f"{v:.2f}", ha="center")

    fig.suptitle("Comparaison SSIM / PSNR : naive vs encoded vs reverse_encoded")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Graphique sauvegardé : {output_path}")




if __name__ == "__main__":
    BASE_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/data/experiment_srpbs")
    OUTPUT_DIR = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/experiments/experiment_similarite_anatomy")

    # Liste des dossiers d'inputs
    INPUT_DIRS = {
        "naive": BASE_DIR / "outputs_harmonized_naive",
        "encoded": BASE_DIR / "outputs_harmonized_encoded",
        "reverse_encoded": BASE_DIR / "outputs_harmonized_reverse_encoded",
    }

    print("====== Experience : similarite moyenne, patients voyageurs (Comparaison entre méthodes) ======")

    results_per_method = {}
    for label, dir_path in INPUT_DIRS.items():
        print(f"\n--- Traitement de la méthode : {label} ---")
        results_per_method[label] = run_experiment(dir_path, label)

    print("\n=== Résultats globaux ===")
    for label, res in results_per_method.items():
        if res is None:
            print(f"{label} : indisponible")
        else:
            print(f"{label} : SSIM={res['total_mean_ssim']:.4f}, "
                  f"PSNR={res['total_mean_psnr']:.4f} (n={res['n_subjects']} sujets)")

    plot_comparison(results_per_method, OUTPUT_DIR / "comparison_ssim_psnr.png")

