"""
Paths to data and models.
"""
import os
from pathlib import Path

# If running on Google Colab, set the root directory to the Google Drive (mounted at /content/drive)
if 'COLAB_GPU' in os.environ:
    # Note: this assumes that the Google Drive has data and models in the following directory structure
    ROOT = Path('/content/drive/MyDrive/ENSF619')
# Otherwise, set the root directory to the current project directory
else:
    ROOT = Path(__file__).parent.resolve()

DATA_PATH = ROOT / 'dataset'
ORIGINALS = DATA_PATH / 'adult/original'
NEGATIVES = DATA_PATH / 'adult/negative'
MASKS = DATA_PATH / 'adult/mask'
NEWBORNS = DATA_PATH / 'newborn/original'

MODEL_PATH = ROOT / 'models'
PRETRAINED_PATH = MODEL_PATH / 'pretrained'
