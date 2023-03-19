from pathlib import Path


DATA_PATH = Path(__file__).parent.resolve() / 'dataset'
ORIGINALS = DATA_PATH / 'adult/original'
NEGATIVES = DATA_PATH / 'adult/negative'
MASKS = DATA_PATH / 'adult/mask'
NEWBORNS = DATA_PATH / 'newborn/original'

MODEL_PATH = Path(__file__).parent.resolve() / 'models'
PRETRAINED_PATH = MODEL_PATH / 'pretrained'
