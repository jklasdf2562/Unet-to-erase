import os
import torch

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The author's dataset is divided into three parts and placed in a folder. Each part is further divided into input and output. When reproducing, you can modify this part according to your own file management habits.

DATA_DIRS = [
    os.path.join(WORKSPACE_ROOT, 'U_net_eraser', 'deli', d)
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
