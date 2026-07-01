import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import csv
 
#ANATOMY_OUT_PATH = Path("/NAS/coolio/benolive/MaRaI_2/MaRaI/logs/anatomy_encoder.pt")
ANATOMY_OUT_PATH = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/models/anatomy_encoder.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
 
VOLUME_DIR        = Path("/NAS/coolio/Barnabe/SRPBS_post_process/native/brain")
MASK_DIR          = Path("/NAS/coolio/Barnabe/SRPBS_post_process/native/brain_mask")
OUTPUT_PATH_RAW     = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/data/experiment_srpbs/raw")
OUTPUT_PATH_ENCODED = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/data/experiment_srpbs/encoded")
OUTPUT_CSV_PATH = Path("/NAS/coolio/benolive/Diffusion_beta_encoder/experiments/experiment_similarite_anatomy/data_by_name_test.csv") 
 
# ==== U-Net (anatomy encoder) ====
 
class ConvBlock2d(nn.Module):
    """Bloc double-convolution avec InstanceNorm + LeakyReLU.
    Génère les clés .conv.0 (1ère Conv) et .conv.3 (2ème Conv)
    car les indices 1,2 sont InstanceNorm+LeakyReLU."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),   # index 0  → .conv.0.weight
            nn.InstanceNorm2d(mid_ch),             # index 1
            nn.LeakyReLU(0.1),                     # index 2
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),   # index 3  → .conv.3.weight
            nn.InstanceNorm2d(out_ch),             # index 4
            nn.LeakyReLU(0.1)                      # index 5
        )
 
    def forward(self, in_tensor):
        return self.conv(in_tensor)
 
 
class Upsample(nn.Module):
    """Interpolation bilinéaire *2, puis Conv+InstanceNorm+LeakyReLU,
    puis concaténation avec le skip correspondant."""
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1)
        )
 
    def forward(self, in_tensor, encoded_feature):
        # Upsample spatial *2
        x = F.interpolate(in_tensor, scale_factor=2, mode='bilinear', align_corners=False)
        # Réduction de canaux par convolution
        x = self.conv(x)
        # Concaténation avec le skip de l'encodeur → double les canaux
        return torch.cat([encoded_feature, x], dim=1)
 
 
class UNet(nn.Module):
    """U-Net de network.py (MaRaI / DIST-CLIP).
 
    Instanciation pour le beta_encoder du checkpoint :
        UNet(in_ch=1, out_ch=1, base_ch=16, num_lvs=4, final_act='noact')
 
    Architecture résultante :
        in_conv       : Conv2d(1→16, k=3)
        contrast_norm : InstanceNorm2d(1)       <-- présente mais COMMENTÉE dans forward
        down_convs[0] : ConvBlock2d(16, 32, 32)   + MaxPool
        down_convs[1] : ConvBlock2d(32, 64, 64)   + MaxPool
        down_convs[2] : ConvBlock2d(64, 128, 128) + MaxPool
        down_convs[3] : ConvBlock2d(128, 256, 256)+ MaxPool
        bottleneck    : ConvBlock2d(256, 512, 512)
        up_samples[3] : Upsample(512)  → Conv(512→256) + cat skip
        up_convs[3]   : ConvBlock2d(512, 256, 256)
        up_samples[2] : Upsample(256)  → Conv(256→128) + cat skip
        up_convs[2]   : ConvBlock2d(256, 128, 128)
        up_samples[1] : Upsample(128)  → Conv(128→64)  + cat skip
        up_convs[1]   : ConvBlock2d(128, 64, 64)
        up_samples[0] : Upsample(64)   → Conv(64→32)   + cat skip
        up_convs[0]   : ConvBlock2d(64, 32, 32)
        out_conv      : Conv(32→16) + LeakyReLU + Conv(16→out_ch)
    """
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact', use_contrast_norm=False):
        super().__init__()
        self.final_act = final_act
        self.use_contrast_norm = use_contrast_norm
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.contrast_norm = nn.InstanceNorm2d(in_ch, affine=True)  # poids présents, non utilisés
 
        self.down_convs   = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples   = nn.ModuleList()
        self.up_convs     = nn.ModuleList()
 
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
 
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch, out_ch, 3, 1, 1)
        )
 
    def forward(self, in_tensor, condition=None):
        if self.use_contrast_norm:
            in_tensor = self.contrast_norm(in_tensor)
        encoded_features = []
        x = self.in_conv(in_tensor)
 
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(
                    torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1)
                )
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
 
        x = self.bottleneck_conv(x)
 
        for encoded_feature, up_conv, up_sample in zip(
            reversed(encoded_features),
            reversed(self.up_convs),
            reversed(self.up_samples)
        ):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
 
        x = self.out_conv(x)
 
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == 'relu':
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            # 'noact' : pas d'activation
            x = x
 
        return x
 
 
# ==== Fonctions utilitaires ====
 
def find_mask_path(volume_path: Path, mask_dir: Path) -> Path:
    stem = volume_path.name
    # Suppression de toutes les extensions (.nii.gz ou .nii)
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if stem.endswith("_brain"):
        stem = stem[: -len("_brain")]
    mask_name = f"{stem}_mask.nii.gz"
    return mask_dir / mask_name
 
 
def load_mask_slice(mask_path: Path, z_idx: int) -> np.ndarray:
    """
    Charge le masque NIfTI fourni et extrait la slice à l'indice z_idx.
    """
    if not mask_path.exists():
        raise FileNotFoundError(f"Masque introuvable : {mask_path}")
 
    nii_mask = nib.load(str(mask_path))
    mask_vol = nii_mask.get_fdata(dtype=np.float32)
 
    if mask_vol.ndim == 4:
        mask_vol = mask_vol[..., 0]
 
    if mask_vol.ndim != 3:
        raise ValueError(f"Masque 3D attendu, obtenu shape={mask_vol.shape}")
 
    mask_2d = mask_vol[:, :, z_idx] > 0.5  # binarisation
    return mask_2d.astype(bool)
 
 
def apply_mask(slice_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    """
    Applique le masque sur la slice : fond mis à 0, contrastes inchangés.
    Les deux tableaux doivent avoir la même shape (X, Y).
    """
    if slice_2d.shape != mask_2d.shape:
        raise ValueError(
            f"Shapes incompatibles : slice={slice_2d.shape}, masque={mask_2d.shape}"
        )
    return (slice_2d * mask_2d).astype(np.float32)
 
 
def extract_central_slice(volume_path: Path, target_z_ratio: float = 0.5) -> tuple[np.ndarray, int]:
    """
    Extrait la slice axiale à une position relative fixe dans le volume.
 
    La position est définie par `target_z_ratio` (0.5 = milieu) appliqué sur
    l'axe Z *après* réorientation RAS standard via nibabel.
 
    Retourne :
        slice_2d (np.ndarray, float32) : slice axiale brute, shape (X, Y)
        z_idx    (int)                  : indice de slice utilisé
    """
    volume_path = Path(volume_path)
    if not (str(volume_path).endswith(".nii.gz") or str(volume_path).endswith(".nii")):
        raise ValueError(f"Fichier NIfTI attendu, obtenu : {volume_path}")
 
    nii    = nib.load(str(volume_path))
    volume = nii.get_fdata(dtype=np.float32)
 
    if volume.ndim == 4:
        print("  Volume 4D => on garde t=0.")
        volume = volume[..., 0]
 
    if volume.ndim != 3:
        raise ValueError(f"Volume 3D attendu, obtenu shape={volume.shape}")
 
    X, Y, Z = volume.shape
    z_idx   = int(round(Z * target_z_ratio))
    z_idx   = max(0, min(z_idx, Z - 1))  # garde-fou aux bornes
 
    slice_2d = volume[:, :, z_idx].astype(np.float32)
 
    print(f"  Dimensions : X={X}, Y={Y}, Z={Z}")
    print(f"  Ratio cible={target_z_ratio:.2f}  --> slice z_idx={z_idx}")
    print(f"  Intensités : min={slice_2d.min():.2f}  max={slice_2d.max():.2f}  "
          f"mean={slice_2d.mean():.2f}")
 
    return slice_2d, z_idx
 
 
def pad_to_multiple(tensor, multiple=16):
    """
    Padde un tenseur (1, 1, H, W) pour que H et W soient divisibles par `multiple`.
    Utilise un padding symétrique avec des zéros (fond = 0 en IRM).
    Retourne (tensor_padded, (pad_top, pad_left, H_orig, W_orig)) pour permettre
    de recadrer la sortie du réseau exactement sur la taille originale.
    """
    _, _, H, W = tensor.shape
    H_pad = (multiple - H % multiple) % multiple
    W_pad = (multiple - W % multiple) % multiple
    pad_top    = H_pad // 2
    pad_bottom = H_pad - pad_top
    pad_left   = W_pad // 2
    pad_right  = W_pad - pad_left
    # F.pad : (left, right, top, bottom)
    tensor_padded = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return tensor_padded, (pad_top, pad_left, H, W)
 
 
def unpad(tensor, pad_info):
    """
    Supprime le padding ajouté par pad_to_multiple pour revenir à la taille originale.
    """
    pad_top, pad_left, H_orig, W_orig = pad_info
    return tensor[:, :, pad_top:pad_top + H_orig, pad_left:pad_left + W_orig]
 
 
def prepare_tensor(masked_slice: np.ndarray, multiple: int = 16) -> tuple:
    """
    Convertit une slice 2D masquée en tenseur prêt pour le modèle.
 
    Étapes :
        numpy (H, W) → torch (1, 1, H, W) → padding divisible par `multiple`
 
    Retourne (tensor_padded, pad_info).
    """
    if not isinstance(masked_slice, np.ndarray) or masked_slice.ndim != 2:
        raise ValueError(f"np.ndarray 2D attendu, obtenu shape={getattr(masked_slice, 'shape', '?')}")
 
    tensor = torch.from_numpy(masked_slice).unsqueeze(0).unsqueeze(0)
    tensor_padded, pad_info = pad_to_multiple(tensor, multiple=multiple)
 
    pad_top, pad_left, H, W = pad_info
    if pad_top > 0 or pad_left > 0:
        print(f"  Padding appliqué : {H}×{W} --> {tuple(tensor_padded.shape[2:])}  "
              f"(top={pad_top}, left={pad_left})")
    else:
        print(f"  Shape finale : {tuple(tensor_padded.shape[2:])} (aucun padding nécessaire)")
 
    return tensor_padded, pad_info
 
 
def save_slice_as_npy(array_2d: np.ndarray, output_path: Path) -> None:
    """Sauvegarde une slice 2D au format .npy."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), array_2d)
    print(f"  => Sauvegardé : {output_path}")
 
 
# ==== Chargement du modèle ====
 
def load_beta_encoder(checkpoint_path: Path) -> UNet:
    reloaded    = torch.load(checkpoint_path, map_location="cpu")
    reloaded_sd = reloaded["state_dict"]
 
    in_ch   = reloaded_sd["in_conv.weight"].shape[1]
    base_ch = reloaded_sd["in_conv.weight"].shape[0]
    out_ch  = reloaded_sd["out_conv.2.weight"].shape[0]
    num_lvs = sum(
        1 for k in reloaded_sd
        if k.startswith("down_convs.") and k.endswith(".conv.0.weight")
    )
 
    print("Hyperparamètres détectés depuis le state_dict :")
    print(f"  in_ch={in_ch}  out_ch={out_ch}  base_ch={base_ch}  num_lvs={num_lvs}")
 
    model = UNet(
        in_ch=in_ch,
        out_ch=out_ch,
        base_ch=base_ch,
        num_lvs=num_lvs,
        final_act="noact",
        use_contrast_norm=True,
    )
 
    missing, unexpected = model.load_state_dict(reloaded_sd, strict=True)
    print("load_state_dict strict=True réussi")
 
    model.eval()
    model.to(device)
    print("Mode eval activé, modèle sur", device)
    return model
 
 
# ==== Outil exploration des volumes ====
 
def collect_volumes(volume_dir: Path) -> list[Path]:
    candidates = sorted(
        p for p in volume_dir.rglob("*")
        if p.is_file()
        and (p.name.endswith(".nii.gz") or p.name.endswith(".nii"))
        and not p.name.endswith("_mask.nii.gz")
        and not p.name.endswith("_mask.nii")
    )
    print(f"{len(candidates)} volume(s) trouvé(s) dans : {volume_dir}")
    return candidates
 
 
# ==== Pipeline principal ====
 
def process_volume(
    volume_path: Path,
    beta_encoder: UNet,
    mask_dir: Path,
    output_dir_raw: Path,
    output_dir_encoded: Path,
    target_z_ratio: float = 0.5,
) -> None:
    """
    Traite un volume complet :
        extraction slice - chargement masque - application masque - inférence - recadrage - sauvegarde : slice brute + slice encodée
    """
    volume_path         = Path(volume_path)
    output_dir_raw      = Path(output_dir_raw)
    output_dir_encoded  = Path(output_dir_encoded)
    print(f"\n{'='*60}")
    print(f"Volume : {volume_path.name}")
 
    # Extraction de la slice centrale cohérente
    slice_raw, z_idx = extract_central_slice(volume_path, target_z_ratio=target_z_ratio)
    slice_raw = np.flip(slice_raw, axis=0).copy()
 
    # Chargement du masque fourni et extraction de la même slice
    mask_path = find_mask_path(volume_path, mask_dir)
    print(f"  Masque  : {mask_path.name}")
    mask_2d = load_mask_slice(mask_path, z_idx=z_idx)
    mask_2d = np.flip(mask_2d, axis=0).copy()
 
    # Application du masque : fond=0, contrastes bruts conservés
    masked_slice = apply_mask(slice_raw, mask_2d)
 
    # --- Sauvegarde de la slice brute ---
    stem = volume_path.name
    for ext in (".nii.gz", ".nii"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
 
    raw_out_path = output_dir_raw / f"{stem}.npy"
    save_slice_as_npy(masked_slice, raw_out_path)
 
    # --- Inférence beta_encoder ---
    tensor_input, pad_info = prepare_tensor(masked_slice, multiple=16)
    tensor_input = tensor_input.to(device)
    with torch.no_grad():
        anatomy_out = beta_encoder(tensor_input)   # (1, 1, H_pad, W_pad)
 
    # Recadrage à la taille originale (suppression du padding)
    anatomy_out_cropped = unpad(anatomy_out, pad_info)   # (1, 1, H, W)
    result_np = anatomy_out_cropped.squeeze().cpu().numpy()  # (H, W)
    result_np = apply_mask(result_np, mask_2d)  # Application mask sur la sortie
 
    print(f"  Sortie modèle : shape={result_np.shape}  "
          f"min={result_np.min():.4f}  max={result_np.max():.4f}")
 
    # --- Sauvegarde de la slice encodée ---
    anatomy_out_path = output_dir_encoded / f"{stem}.npy"
    save_slice_as_npy(result_np, anatomy_out_path)

    return stem, raw_out_path, anatomy_out_path 
 
 
if __name__ == "__main__":
    OUTPUT_PATH_RAW.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH_ENCODED.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
 
    volume_paths = collect_volumes(VOLUME_DIR)
 
    beta_encoder = load_beta_encoder(ANATOMY_OUT_PATH)
 
    ok, errors = 0, []
    csv_rows = [] 
    for volume_path in volume_paths:
        try:
            stem, raw_path, enc_path = process_volume(  
                volume_path        = volume_path,
                beta_encoder       = beta_encoder,
                mask_dir           = MASK_DIR,
                output_dir_raw     = OUTPUT_PATH_RAW,
                output_dir_encoded = OUTPUT_PATH_ENCODED,
                target_z_ratio     = 0.5,
            )
            csv_rows.append({
                "name":         stem,
                "raw_path":     str(raw_path),
                "encoded_path": str(enc_path),
            })
            ok += 1
        except Exception as e:
            print(f"  [ERREUR] {volume_path.name} : {e}")
            errors.append((volume_path.name, str(e)))

    # CSV :
    with open(OUTPUT_CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "raw_path", "encoded_path"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV sauvegardé   : {OUTPUT_CSV_PATH}")

 
    print(f"\nPipeline terminé - {ok}/{len(volume_paths)} volumes traités avec succès.")
    if errors:
        print(f"\n{len(errors)} erreur(s) :")
        for name, msg in errors:
            print(f"  - {name} : {msg}")
