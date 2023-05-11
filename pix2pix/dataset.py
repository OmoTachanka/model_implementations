from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import config

class MapDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.files = os.listdir(self.root)
        # print(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        fname = self.files[index]
        path = os.path.join(self.root, fname)
        img = np.array(Image.open(path))
        x = img[:, :600, :]
        y = img[:, 600:, :]

        augs = config.both_trans(image = x, image0 = y)
        x, y = augs["image"], augs["image0"]
        x = config.trans_input(image = x)
        y = config.trans_mask(image = y)
        return x["image"], y["image"]


# if __name__ == "__main__":
#     data = MapDataset("./maps/train/")
#     print(data)