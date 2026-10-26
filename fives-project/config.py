"""Central configuration: paths, constants, class map, seed."""
from pathlib import Path

# The only absolute path in the project.
DATASET_PATH = Path("/home/ghotra/Downloads/dataset")

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

SEED = 42
FIG_DPI = 200

# Dataset layout: <DATASET_PATH>/{train,test}/{Original,Ground truth}/<id>_<L>.png
# The class is the filename suffix letter (e.g. 100_A.png -> AMD).
SPLITS = ("train", "test")
IMAGE_DIR_NAME = "Original"
MASK_DIR_NAME = "Ground truth"  # vessel segmentation masks, not class labels
IMAGE_EXT = ".png"  # also skips the stray Thumbs.db files

LABEL_LETTER_TO_NAME = {"N": "Normal", "A": "AMD", "D": "Diabetic Retinopathy", "G": "Glaucoma"}
CLASS_NAMES = ["Normal", "AMD", "Diabetic Retinopathy", "Glaucoma"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}
N_CLASSES = len(CLASS_NAMES)

for _d in (DATA_DIR, MODELS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
