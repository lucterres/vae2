import cv2
from torch.utils.data import Dataset
from torchvision import transforms as tv
from PIL import Image


class Seismic(Dataset):
    def __init__(self, img_root, mask_root, x_db, transform=None):
        self.img_dir = img_root
        self.mask_dir = mask_root
        self.X = x_db
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_id = self.X[idx]

        img_path = self.img_dir + str(img_id) + ".png"
        mask_path = self.mask_dir + str(img_id) + ".png"

        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (128, 128))

        if self.transform is not None:
            # transform image and mask
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug['image'])
            # convert to 1channel
            mask = Image.fromarray(aug['mask']).convert('L')
        if self.transform is None:
            img = Image.fromarray(img)
            # convert to 1channel
            mask = Image.fromarray(mask).convert('L')
        t = tv.ToTensor()
        img = t(img)
        mask = t(mask)
        return img, mask

class SeismicMaskDS(Dataset):

    def __init__(self, x_db, transform=None):

        self.X = x_db
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        figs = self.X['id'].iloc[idx]
        file_path = f'./train/masks/{figs}.png'
        mask = cv2.imread(file_path)
        #mask = self.transform(mask)
        mask = cv2.resize(mask, (32, 32))
        mask = Image.fromarray(mask).convert('L')
        t = tv.ToTensor()
        mask = t(mask)
        return mask
