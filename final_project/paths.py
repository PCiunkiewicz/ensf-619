import os
from pathlib import Path


if 'COLAB_GPU' in os.environ:
    ROOT = Path('/content/drive/MyDrive/ENSF619')
else:
    ROOT = Path(__file__).parent.resolve()

DATA_PATH = ROOT / 'dataset'
ORIGINALS = DATA_PATH / 'adult/original'
NEGATIVES = DATA_PATH / 'adult/negative'
MASKS = DATA_PATH / 'adult/mask'
NEWBORNS = DATA_PATH / 'newborn/original'

MODEL_PATH = ROOT / 'models'
PRETRAINED_PATH = MODEL_PATH / 'pretrained'
