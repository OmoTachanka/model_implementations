import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR  = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMG_SIZE = 256
CHANNELS = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 100
LOAD_MODEL = False
SAVE_MODEL = True
# CHCKPT_GEN = "gen.pth.tar"
# CHCKPT_DISC = "disc.pth.tar"

both_trans = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.HorizontalFlip(p = 0.5),
], additional_targets={"image0":"image"})

trans_input = A.Compose([
    A.ColorJitter(0.1),
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2(),
])

trans_mask = A.Compose([
    A.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2(),
])

