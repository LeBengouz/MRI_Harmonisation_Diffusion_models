# Pipeline de Précompute

Ce document décrit la pipeline de précompute du projet. Il couvre le rôle de chaque étape, l'ordre d'exécution, les paramètres configurables et les dépendances inter-dépôts.

---

## Vue d'ensemble

Globalement les fichiers du dossier précompute préparent les données brutes (volumes IRM `.pt`) pour leur utilisation par le modèle de diffusion. Celui-ci utilise les données brutes mais aussi leurs versions encodées par des modèles pré-entraînés. Donc l'objectif est de produire :

- des **slices 2D** extraites de chaque volume (`.npy` + `.png`),
- des **encodages anatomiques** de chaque slice via le beta-encoder,
- un **CSV d'entrée pour MR-CLIP** (utilisé pour construire les embeddings).
- des **embeddings** grâce à l'utilisation de MR-CLIP (dans le dépôt MaRAI_2)

La pipeline complète comporte **5 étapes**, dont 2 sont gérées en dehors de *run_precompute.py* :

```
[0] listing_data.py          ← prérequis, exécuté au préalable
        ↓
[1] extract_slices.py        ┐
[2] beta_encoder_runner.py   ├─ run_precompute.py 
[3] build_mrclip_dataset.py  ┘
        ↓
[4] MR-CLIP + embeddings     ← géré dans MarAI_2
```

> **Important :** Les étapes 0 et 4 ne sont **pas** exécutées par `run_precompute.py`. Elles doivent être lancées manuellement avant et après ce script, respectivement.


De plus, MaRai fait référence au projet : https://github.com/myigitavci/MaRaI 

---

## Structure des dossiers

```
project_root/
├── logs_diffusion/
├── checkpoints_diffusion/
├── outils/
│   ├── checkpoints.py          (save models, load models)
│   └── visualization.py        (plotting)
├── precompute/
│   ├── run_precompute.py        ← script principal
│   ├── extract_slices.py
│   ├── beta_encoder_runner.py
│   ├── build_mrclip_dataset.py
│   └── listing_data.py
├── models/
│   ├── beta_encoder.py
│   └── anatomy_encoder.pt       ← checkpoint du beta-encoder
└── data/
    ├── csv_files/
    │   ├── listing_data_train.csv
    │   ├── raw_slices_and_json_paths_train.csv
    │   └── prerequis_MRCLIP/
    │       └── mrclip_dataset_train.csv
    └── brain_slices/
        └── train/
            ├── raw/             ← slices .npy
            ├── png/             ← slices .png
            └── encoded/         ← encodages .npy
```

---

## Étape préliminaire — `listing_data.py`

> **À exécuter avant `run_precompute.py`, géré séparément.**

Ce script liste l'ensemble des paths des volumes disponibles ainsi que leurs nom. Il produit le CSV d'entrée consommé par l'étape 1 :

```
data/csv_files/listing_data_train.csv
```

Ce fichier CSV contient une colonne `volume_path` pointant vers chaque fichier `.pt`.

---

## Pipeline principal — `run_precompute.py`

### Lancement

Depuis le dossier `precompute/` :

```bash
# Pipeline complet (étapes 1 → 2 → 3)
python run_precompute.py

# Sauter l'extraction des slices si déjà effectuée
python run_precompute.py --skip-extract
```

### Option disponible

| Option | Description |
|---|---|
| `--skip-extract` | Ignore l'étape 1 (`extract_slices`).

---

### Étape 1 — Extraction des slices (`extract_slices.py`)

Charge chaque volume `.pt` listé dans le CSV d'entrée, extrait la slice centrale selon l'axe Z, et la sauvegarde en `.npy` et `.png`.

**Paramètres dans `run_precompute.py` :**

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `EXTRACT_INPUT_CSV` | `data/csv_files/listing_data_train.csv` | CSV d'entrée produit par `listing_data.py`. Doit contenir une colonne `volume_path`. |
| `EXTRACT_OUTPUT_DIR` | `data/brain_slices/train/raw` | Dossier de sortie des slices `.npy`. |
| `EXTRACT_PNG_OUTPUT_DIR` | `data/brain_slices/train/png` | Dossier de sortie des slices `.png` au format rgb comme le nécessite le checkpoint de MR-CLIP plus tard. |
| `EXTRACT_CSV_OUT` | `data/csv_files/raw_slices_and_json_paths_train.csv` | CSV de sortie listant les chemins des slices extraites. Le suffixe `SUFFIX` est appliqué automatiquement. |
| `EXTRACT_Z_RATIO` | `0.5` | Position relative de la slice extraite sur l'axe Z (0 = début, 1 = fin, 0.5 = centre). |
| `EXTRACT_NUM_WORKERS` | `4` | Nombre de threads parallèles pour le traitement des volumes. |

**Sorties :**
- `data/brain_slices/train/raw/*.npy`
- `data/brain_slices/train/png/*.png`
- `data/csv_files/raw_slices_and_json_paths_train.csv`

---

### Étape 2 — Encodage anatomique (`beta_encoder_runner.py`)

Pour chaque slice `.npy` extraite, passe la donnée dans le beta-encoder (UNet) afin de produire une représentation anatomique encodée. Les résultats sont sauvegardés en `.npy` dans le dossier `encoded/`.

**Paramètres dans `run_precompute.py` :**

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `ENCODER_FOLDER_RAW` | `data/brain_slices/train/raw` | Dossier contenant les slices `.npy` à encoder (serait au format .nii.gz si on était sur de la 3D pour MR-CLIP). |
| `ENCODER_OUTPUT_DIR` | `data/brain_slices/train/encoded` | Dossier de sortie des encodages `.npy`. Créé automatiquement si absent. |
| `ENCODER_CKPT` | `models/anatomy_encoder.pt` | Chemin vers le checkpoint du beta-encoder. |

Le device (CPU / CUDA) est détecté automatiquement.

**Sorties :**
- `data/brain_slices/train/encoded/*.npy`

---

### Étape 3 — Construction du dataset MR-CLIP (`build_mrclip_dataset.py`)

Génère le CSV d'entrée attendu par MR-CLIP dans le dépôt **MarAI_2**. Ce CSV liste chaque slice `.png` avec un texte descriptif par défaut puisque nous ne disposons pas des métadonnées DICOM.

**Paramètres dans `run_precompute.py` :**

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `MRCLIP_INPUT_DIR` | `data/brain_slices/train/png` | Dossier contenant les slices `.png` à référencer. |
| `MRCLIP_OUTPUT_DIR` | `data/csv_files/prerequis_MRCLIP` | Dossier de sortie du CSV MR-CLIP. Créé automatiquement si absent. |
| `MRCLIP_DEFAULT_TEXT` | `"A brain MRI, plane NONE, ..."` | Texte descriptif par défaut respectant le format attendu (mais sans informations). |
| `SUFFIX` | `_train` | Suffixe apposé au nom du fichier CSV de sortie (`mrclip_dataset_train.csv`). Partagé entre toutes les étapes pour la cohérence des nommages. |

**Sorties :**
- `data/csv_files/prerequis_MRCLIP/mrclip_dataset_train.csv`

---

## Étape finale — MR-CLIP et embeddings (MarAI_2)

> **À exécuter après `run_precompute.py`, dans le dépôt MarAI_2.**

MR-CLIP consomme le CSV produit à l'étape 3 et calcule les embeddings multimodaux (image + texte) associés à chaque slice. Voici des instructions d'exécution :

Il est nécessaire de se placer dans le dossier src :
```
cd /NAS/coolio/benolive/MaRaI_2/MaRaI/src
```
Puis exécuter Mr-CLIP à l'aide de la commande suivante :





Ces paramètres permettent :
- D'utiliser le dataset créé précédemment,
- Choisir le type de modèle adapté,
- De choisir l'emplacement des saves. Ainsi, dans "../logs/mr_clip_3d/checkpoints/checkpoints/" sont mis les saves dont text_embeddings.pkl,
- etc.



---

## Paramètre global — `SUFFIX`

Le paramètre `SUFFIX` (défaut : `_train`) est partagé entre les étapes et s'applique à tous les fichiers de sortie nommés dynamiquement. Pour traiter un jeu de validation ou de test, il suffit de modifier cette valeur ainsi que les chemins associés :

```python
SUFFIX = "_val"
EXTRACT_INPUT_CSV = Path(".../listing_data_val.csv")
# etc.
```

---

## Résumé de l'ordre d'exécution

| Ordre | Script / Action | Dépôt | Prérequis |
|---|---|---|---|
| 0 | `listing_data.py` | Ce dépôt | Données DICOM brutes disponibles |
| 1 | `run_precompute.py` | Ce dépôt | CSV produit par l'étape 0 |
| 2 | MR-CLIP + embeddings | MarAI_2 | CSV produit par l'étape 1 (étape 3 interne) |
