from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from MVAE import *

TRAIN_MASK_DIR = './train/'

transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor()])

X_train = datasets.ImageFolder(root=TRAIN_MASK_DIR, transform=transform)
train_loader = DataLoader(X_train, batch_size=128, shuffle=True)

Decoders = nn.ModuleList([Decoder_MLP(latent_dim=100, in_channel=1, im_size=32, hiddens=[128, 256, 512]),
                          Decoder_MLP(latent_dim=100, in_channel=1, im_size=32, hiddens=[256, 512, 1024]),
                          Decoder_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[1024, 512, 256, 128], init=2),
                          Decoder_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[1024, 512, 256], init=4),
                          Decoder_Linear_Conv(latent_dim=100, in_channel=1, im_size=32, hiddens=[512, 256, 128, 64],
                                              init=2)])

MabVae = MabVAE(train_loader, Decoders, eps=0.3, i=0)

#device = torch.device(1 if torch.cuda.is_available() else 0)
trainer = Trainer(gpus=1, max_epochs=5)

trainer.fit(MabVae)
