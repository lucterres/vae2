"""
Script para gerar 100 amostras de máscaras sintéticas usando o modelo VAE treinado.
Reproduz o workflow do notebook maskCustomDS.ipynb.
"""

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from MVAE import *
import matplotlib.pyplot as plt

# Disable matplotlib display
plt.ioff()


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, label_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        filename = self.img_labels.iloc[idx, 0] + '.png'
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)
        return image, label


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


def create_output_directory(base_dir='./generated_masks'):
    """Cria diretório de saída se não existir."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Diretório criado: {base_dir}")
    else:
        print(f"Usando diretório existente: {base_dir}")
    return base_dir


def load_model(checkpoint_path=None):
    """Carrega ou cria o modelo VAE."""
    # Configuração padrão do modelo
    latent_dim = 100
    in_channel = 1
    im_size = 64
    hiddens = [256, 512, 1024]
    
    # Se checkpoint existe, carrega configurações dele
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Carregando checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Usa configurações do checkpoint se disponíveis
        if 'latent_dim' in checkpoint:
            latent_dim = checkpoint['latent_dim']
            im_size = checkpoint['im_size']
            in_channel = checkpoint['in_channel']
            print(f"  - Latent dim: {latent_dim}")
            print(f"  - Image size: {im_size}")
            print(f"  - Epochs treinadas: {checkpoint.get('epochs', 'N/A')}")
    
    # Cria decoder
    decoder = Decoder_MLP(latent_dim=latent_dim, in_channel=in_channel, 
                         im_size=im_size, hiddens=hiddens)
    
    # Carrega pesos se checkpoint fornecido
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print("✓ Pesos do decoder carregados com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar pesos: {e}")
            print("Usando modelo não treinado")
    else:
        print("⚠ Aviso: Usando modelo não treinado (checkpoint não encontrado)")
        print("  Para melhores resultados, treine o modelo primeiro no notebook")
    
    return decoder


def generate_masks(decoder, num_samples=100, latent_dim=100, output_dir='./generated_masks'):
    """Gera máscaras sintéticas e salva como PNG."""
    
    # Define transformação de saída (blur + threshold)
    transformOut = transforms.Compose([
        transforms.Resize((101, 101)),
        transforms.GaussianBlur((5, 5), sigma=(0.1, 2.0)),
        ThresholdTransform(thr_255=127)
    ])
    
    # Configura device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Nota: Não podemos usar decoder.to(device) devido a um bug no código MVAE
    # onde self.modules sobrescreve o método modules() do PyTorch
    # Então trabalhamos em CPU
    decoder.eval()
    
    print(f"Gerando {num_samples} máscaras...")
    print(f"Device: CPU (devido a limitação do código MVAE)")
    
    with torch.no_grad():
        # Gera em batches para eficiência
        batch_size = 16
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        mask_count = 0
        
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_samples - mask_count)
            
            # Gera ruído latente
            fixed_noise = torch.randn(current_batch_size, latent_dim)
            
            # Gera imagens fake
            fake = decoder(fixed_noise).detach()
            
            # Aplica transformações (blur + threshold)
            blurredAndBinarized = transformOut(fake)
            
            # Salva cada máscara individualmente
            for i in range(current_batch_size):
                mask_tensor = blurredAndBinarized[i]
                
                # Converte tensor para numpy e normaliza para [0, 255]
                mask_np = mask_tensor.squeeze().numpy()
                mask_np = (mask_np * 255).astype(np.uint8)
                
                # Salva como PNG
                output_path = os.path.join(output_dir, f'generated_mask_{mask_count:04d}.png')
                Image.fromarray(mask_np).save(output_path)
                
                mask_count += 1
                
                # Progresso
                if mask_count % 10 == 0:
                    print(f"Geradas: {mask_count}/{num_samples}")
    
    print(f"\n✓ Concluído! {num_samples} máscaras salvas em: {output_dir}")
    return output_dir


def create_visualization(output_dir, num_display=16):
    """Cria visualização em grade das máscaras geradas."""
    print(f"\nCriando visualização...")
    
    mask_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])[:num_display]
    
    if len(mask_files) == 0:
        print("Nenhuma máscara encontrada para visualização")
        return
    
    # Cria grade de visualização
    grid_size = int(np.ceil(np.sqrt(num_display)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, mask_file in enumerate(mask_files):
        mask_path = os.path.join(output_dir, mask_file)
        img = Image.open(mask_path)
        axes[idx].imshow(img, cmap='gray')
        axes[idx].axis('off')
    
    # Remove eixos vazios
    for idx in range(len(mask_files), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'visualization_grid.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualização salva: {viz_path}")


def main():
    """Função principal."""
    print("="*60)
    print("Gerador de Máscaras Sintéticas - VAE")
    print("="*60)
    
    # Configurações
    NUM_SAMPLES = 100
    OUTPUT_DIR = './generated_masks'
    CHECKPOINT_PATH = 'vae_checkpoint.pth'  # Especifique o caminho do checkpoint se disponível
    
    # Cria diretório de saída
    output_dir = create_output_directory(OUTPUT_DIR)
    
    # Carrega modelo
    print("\nCarregando modelo...")
    decoder = load_model(CHECKPOINT_PATH)
    
    # Gera máscaras
    generate_masks(decoder, num_samples=NUM_SAMPLES, output_dir=output_dir)
    
    # Cria visualização
    create_visualization(output_dir, num_display=16)
    
    print("\n" + "="*60)
    print("Processo concluído!")
    print("="*60)


if __name__ == "__main__":
    main()
