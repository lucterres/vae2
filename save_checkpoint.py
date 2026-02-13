"""
Script auxiliar para salvar checkpoint do modelo treinado.
Execute este código no notebook após o treinamento.
"""

import torch

def save_model_checkpoint(MabVae, checkpoint_path='vae_checkpoint.pth'):
    """
    Salva checkpoint do modelo VAE treinado.
    
    Parâmetros:
        MabVae: Modelo MabVAE treinado
        checkpoint_path: Caminho para salvar o checkpoint
    """
    checkpoint = {
        'decoder_state_dict': MabVae.decoders[0].state_dict(),
        'encoder_state_dict': MabVae.encoder.state_dict(),
        'latent_dim': MabVae.latent_dim,
        'im_size': MabVae.im_size,
        'in_channel': MabVae.in_channel,
        'history': MabVae.history,
        'NbDraws': MabVae.NbDraws,
        'epochs': MabVae.i
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint salvo: {checkpoint_path}")
    print(f"  - Latent dim: {MabVae.latent_dim}")
    print(f"  - Image size: {MabVae.im_size}")
    print(f"  - Epochs: {MabVae.i}")
    return checkpoint_path


def load_model_checkpoint(checkpoint_path='vae_checkpoint.pth'):
    """
    Carrega checkpoint do modelo.
    
    Retorna:
        dict: Checkpoint com state_dicts e configurações
    """
    checkpoint = torch.load(checkpoint_path)
    print(f"✓ Checkpoint carregado: {checkpoint_path}")
    print(f"  - Latent dim: {checkpoint['latent_dim']}")
    print(f"  - Image size: {checkpoint['im_size']}")
    print(f"  - Epochs: {checkpoint['epochs']}")
    return checkpoint


# Exemplo de uso no notebook:
"""
# Após treinar o modelo no notebook:
from save_checkpoint import save_model_checkpoint

# Salvar checkpoint
save_model_checkpoint(MabVae, 'vae_checkpoint.pth')

# O arquivo pode então ser usado no script generate_masks.py
"""
