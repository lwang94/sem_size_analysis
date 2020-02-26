"""training.py configurations"""

import numpy as np
from pathlib import Path

MODEL = 'stage-2_bs16.pkl'
MODEL_URL = 'https://drive.google.com/uc?id=1ClEUOwFhIOMNxvedps8WQN8Koy6NT5eI'
VERSION = '0.1'

PATH_IMG = Path('data') / 'dataset' / 'good' / 'train_x'
PATH_LBL = Path('data') / 'dataset' / 'good' / 'train_y_png'
CODES = np.array(['background', 'particle'], dtype='<U17')
INPUT_SIZE = (192, 256)
BATCH_SIZE = 16

FREEZE_LAYER = 2
EPOCHS = 50
LEARNING_RATE = slice(2e-5, 5e-5)
WEIGHT_DECAY = 1e-1

SAVE_MODEL = 'stage-2_bs16'
PATH_TO_TESTING = Path('data') / 'dataset' / 'good' / 'testing'
