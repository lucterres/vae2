import pandas as pd
# from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from torchvision import transforms

from MVAE import *
from SeismicT import *

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

# X_train = datasets.ImageFolder(root=TRAIN_MASK_DIR, transform=transform)

TRAIN_CSV = f"data\saltMaskOk.csv"
saltGood = pd.read_csv(TRAIN_CSV)

maskDS = SeismicMaskDS(saltGood, transform=transform)

train_loader = DataLoader(maskDS, batch_size=300, shuffle=True)

Decoders = nn.ModuleList([Decoder_MLP(latent_dim=100, in_channel=1, im_size=32, hiddens=[128, 256, 512]),
                          Decoder_MLP(latent_dim=100, in_channel=1, im_size=32, hiddens=[256, 512, 1024]),
                          Decoder_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[1024, 512, 256, 128], init=2),
                          Decoder_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[1024, 512, 256], init=4),
                          Decoder_Linear_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[512, 256, 128, 64],
                                              init=2)])

# Creation of a MabVAE instance
MVAE = MabVAE(train_loader, Decoders, eps=0.3, i=0)

# Use the GPU for making computations faster
trainer = Trainer(gpus=1, max_epochs=5)
trainer.fit(MVAE)
