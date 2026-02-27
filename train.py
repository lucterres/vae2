"""
Treinamento do MabVAE via script (mais estável que notebook no Windows).

Uso:
    python train.py                        # treino padrão
    python train.py --epochs 50            # mais épocas
    python train.py --batch 64             # batch maior
    python train.py --resume               # retoma do último checkpoint do Lightning
    python train.py --workers 4            # mais workers no DataLoader
"""

import argparse
import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from PIL import Image

from MVAE import MabVAE, Decoder_MLP


# ── Dataset ────────────────────────────────────────────────────────────────────

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, dtype=str)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx, 0].strip('"') + '.png'
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1] if self.img_labels.shape[1] > 1 else 0
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Configuração ───────────────────────────────────────────────────────────────

TRAIN_CSV      = 'data/saltMaskOk.csv'        #saltMaskOk.csv' mask10k_files.csv

TRAIN_MASK_DIR =  '/nethome/atena_projetos/cym7/dataset/tgsSalt/train/mask10k/mask10k'
CHECKPOINT_OUT = 'vae_checkpoint10k_c.pth'


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    # DataLoader com num_workers > 0 funciona corretamente em script (não em notebook)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset = CustomImageDataset(TRAIN_CSV, TRAIN_MASK_DIR, transform)
    print(f"Dataset carregado: {len(dataset)} amostras")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,   # > 0 é seguro em script com __main__
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    print(f"Batches por época: {len(dataloader)}")

    # Modelo
    Decoders = nn.ModuleList([
        Decoder_MLP(latent_dim=100, in_channel=1, im_size=64, hiddens=[256, 512, 1024])
    ])
    model = MabVAE(dataloader, Decoders, eps=0.3, i=0)

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        monitor=None,           # salva toda época
        save_top_k=-1,          # mantém todos os checkpoints
        every_n_epochs=1,
        filename='epoch={epoch:02d}',
        verbose=True,
    )

    callbacks = [checkpoint_cb]
    try:
        callbacks.append(RichProgressBar())  # barra colorida no terminal
    except Exception:
        pass  # fallback para barra padrão se rich não estiver instalado

    # Trainer
    ckpt_path = None
    if args.resume:
        # Encontra o checkpoint mais recente do Lightning
        import glob
        ckpts = sorted(glob.glob('lightning_logs/version_*/checkpoints/*.ckpt'))
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f"Retomando de: {ckpt_path}")
        else:
            print("Nenhum checkpoint encontrado — iniciando do zero.")

    trainer = Trainer(
        accelerator='gpu',
        devices=args.devices,
        strategy='ddp' if args.devices > 1 else 'auto',
        max_epochs=args.epochs,
        log_every_n_steps=10,
        callbacks=callbacks,
        enable_progress_bar=True,
    )

    print(f"\nIniciando treinamento: {args.epochs} épocas, batch={args.batch}, workers={args.workers}\n")
    trainer.fit(model, ckpt_path=ckpt_path)

    # Salva checkpoint no formato usado pelo generate_masks.py
    checkpoint = {
        'decoder_state_dict': model.decoders[0].state_dict(),
        'encoder_state_dict': model.encoder.state_dict(),
        'latent_dim': model.latent_dim,
        'im_size': model.im_size,
        'in_channel': model.in_channel,
        'history': model.history,
        'NbDraws': model.NbDraws,
        'epochs': model.i,
    }
    torch.save(checkpoint, CHECKPOINT_OUT)
    print(f"\n✓ Checkpoint salvo: {CHECKPOINT_OUT}")
    print(f"  Latent dim : {model.latent_dim}")
    print(f"  Image size : {model.im_size}")
    print(f"  Épocas     : {model.i}")
    print(f"\nPara gerar máscaras: python generate_masks.py")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Treina o MabVAE')
    parser.add_argument('--epochs',  type=int, default=20,  help='Número de épocas (padrão: 20)')
    parser.add_argument('--batch',   type=int, default=32,  help='Tamanho do batch (padrão: 32)')
    parser.add_argument('--workers', type=int, default=4,   help='Workers do DataLoader (padrão: 0, seguro no Windows)')
    parser.add_argument('--devices', type=int, default=1,   help='Número de GPUs (padrão: 1)')
    parser.add_argument('--resume',  action='store_true',   help='Retoma do último checkpoint Lightning')
    args = parser.parse_args()
    main(args)
