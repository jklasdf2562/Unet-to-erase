import os
import torch

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# attempt to infer workspace root two levels up from this file
_this_dir = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(_this_dir, '..', '..'))

# data directories: try to locate '手写笔记擦除实验/deli/<date>' under workspace root
DATA_DIRS = [
    os.path.join(WORKSPACE_ROOT, '手写笔记擦除实验', 'deli', d)
    for d in ['20250211', '20250212', '20250213']
]

INPUT_SIZE = 512
# Allow overriding via environment for quick tests
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '8'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', '100'))
LR = 1e-3
WEIGHT_DECAY = 1e-5
CHECKPOINT_PATH = os.path.join(_this_dir, 'checkpoint.pth')
BEST_MODEL_PATH = os.path.join(_this_dir, 'best_model.pth')

SEED = 42
