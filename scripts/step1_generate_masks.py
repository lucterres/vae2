"""
step1_generate_masks.py
-----------------------
PASSO 1 do pipeline de dois estágios:
  Gera N máscaras sintéticas usando o VAE treinado (MabVAE / maskCustomDS.ipynb)
  e salva em um diretório para APROVAÇÃO MANUAL antes de gerar as imagens sísmicas.

Fluxo de trabalho:
  1. Execute este script  →  máscaras salvas em <out_dir>/  (padrão: result/masks_pending)
  2. Revise as máscaras e remova as indesejadas do diretório
  3. Execute textureSSD/scripts/step2_synthesize_from_masks.py apontando para o mesmo diretório

Uso:
    python vae2/scripts/step1_generate_masks.py \\
        --vae_checkpoint vae2/outputs/vae_checkpoint.pt \\
        --out_dir result/masks_pending \\
        --num_masks 1000

Parâmetros:
    --vae_checkpoint    Checkpoint PyTorch do VAE treinado (.pt ou .ckpt)  [obrigatório]
    --out_dir           Diretório de saída das máscaras  [padrão: result/masks_pending]
    --num_masks         Quantidade de máscaras a gerar   [padrão: 1000]
    --latent_dim        Dimensão do espaço latente       [padrão: 100]
    --img_size          Tamanho interno do decoder       [padrão: 64]
    --output_size       Tamanho final da máscara (px)    [padrão: 101]
    --threshold         Limiar de binarização [0-1]      [padrão: 0.1]
    --batch_size        Máscaras por batch de inferência [padrão: 64]
    --resume            Retoma ignorando as já existentes [flag]
"""

import os
import sys

# Garante que o diretório do vae2 esteja no path (para importar MVAE.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_VAE2  = os.path.dirname(_SCRIPT_DIR)   # .../vae2
if _ROOT_VAE2 not in sys.path:
    sys.path.insert(0, _ROOT_VAE2)

import argparse
import uuid

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Passo 1: Gera máscaras via VAE e salva para aprovação manual."
    )
    p.add_argument("--vae_checkpoint", type=str, required=True,
                   help="Caminho do checkpoint PyTorch do VAE (.pt ou .ckpt).")
    p.add_argument("--out_dir", type=str, default="result/masks_pending",
                   help="Diretório de saída das máscaras (padrão: result/masks_pending).")
    p.add_argument("--num_masks", type=int, default=1000,
                   help="Número de máscaras a gerar (padrão: 1000).")
    p.add_argument("--latent_dim", type=int, default=100,
                   help="Dimensão do espaço latente do VAE (padrão: 100).")
    p.add_argument("--img_size", type=int, default=64,
                   help="Tamanho interno de saída do decoder (padrão: 64).")
    p.add_argument("--output_size", type=int, default=101,
                   help="Tamanho final (quadrado) da máscara salva em pixels (padrão: 101).")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Limiar de binarização sobre a saída do decoder, em [0,1] (padrão: 0.1).")
    p.add_argument("--batch_size", type=int, default=64,
                   help="Máscaras geradas por batch de inferência (padrão: 64).")
    p.add_argument("--resume", action="store_true",
                   help="Retoma geração ignorando máscaras já existentes em out_dir.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _count_existing(out_dir: str) -> int:
    """Conta máscaras já salvas (arquivos *_mask.png no diretório)."""
    if not os.path.isdir(out_dir):
        return 0
    return len([f for f in os.listdir(out_dir) if f.endswith("_mask.png")])


def _tensor_to_mask(tensor_img: torch.Tensor, output_size: int, threshold: float) -> np.ndarray:
    """
    Converte tensor [1 x H x W] (valores 0-1) em máscara binária uint8 (0 ou 255).
    Aplica Gaussian blur para suavizar bordas e binariza pelo limiar informado.
    Redimensiona para output_size × output_size.
    """
    img_np = tensor_img.squeeze().cpu().numpy().astype(np.float32)  # [H x W]

    # Gaussian blur para suavizar contornos gerados pelo VAE
    blurred = cv2.GaussianBlur(img_np, (7, 7), 0)

    # Binarização
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # Redimensiona para o tamanho esperado pela síntese
    if binary.shape[0] != output_size or binary.shape[1] != output_size:
        binary = cv2.resize(binary, (output_size, output_size),
                            interpolation=cv2.INTER_NEAREST)
    return binary


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Validações --------------------------------------------------------
    if not os.path.isfile(args.vae_checkpoint):
        raise FileNotFoundError(f"Checkpoint não encontrado: {args.vae_checkpoint}")

    # ---- Diretório de saída ------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)

    already_done = _count_existing(args.out_dir) if args.resume else 0
    remaining    = args.num_masks - already_done
    if remaining <= 0:
        print(f"Já existem {already_done} máscaras em '{args.out_dir}'. Nada a fazer.")
        return
    if already_done > 0:
        print(f"Retomando: {already_done} máscaras existentes. Gerando mais {remaining}.")

    print("=" * 60)
    print(f"Checkpoint VAE : {args.vae_checkpoint}")
    print(f"Saída          : {os.path.abspath(args.out_dir)}")
    print(f"Total máscaras : {args.num_masks}  (gerando: {remaining})")
    print(f"Latent dim     : {args.latent_dim}")
    print(f"Tamanho saída  : {args.output_size}×{args.output_size}")
    print(f"Limiar binário : {args.threshold}")
    print("=" * 60)

    # ---- Dispositivo -------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsando dispositivo: {device}\n")

    # ---- Carrega modelo VAE ------------------------------------------------
    print("Carregando modelo VAE...")
    try:
        from MVAE import Decoder_MLP, MabVAE

        checkpoint = torch.load(args.vae_checkpoint, map_location=device)

        # Reconstrói o decoder padrão usado em maskCustomDS.ipynb
        decoder = Decoder_MLP(
            latent_dim=args.latent_dim,
            in_channel=1,
            im_size=args.img_size,
            hiddens=[256, 512, 1024],
        )
        decoders = torch.nn.ModuleList([decoder])

        # Aceita checkpoint completo do pl.LightningModule ou state_dict puro
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        # Tenta carregar pesos dos decoders (prefixo "decoders.")
        decoder_state = {k.replace("decoders.", "", 1): v
                         for k, v in state.items()
                         if k.startswith("decoders.")}
        if decoder_state:
            decoders.load_state_dict(decoder_state, strict=False)
            print(f"  Pesos do decoder carregados ({len(decoder_state)} tensores).")
        else:
            # Fallback: tenta carregar diretamente no decoder
            decoder.load_state_dict(state, strict=False)
            print("  Pesos carregados diretamente no decoder.")

    except Exception as e:
        raise RuntimeError(
            f"Falha ao carregar o VAE: {e}\n"
            "Verifique se --vae_checkpoint aponta para um arquivo gerado por maskCustomDS.ipynb."
        ) from e

    decoder = decoders[0].to(device).eval()
    print("Modelo VAE carregado com sucesso.\n")

    # ---- Loop de geração em batches ----------------------------------------
    saved  = 0
    errors = 0

    print(f"Iniciando geração de {remaining} máscaras em batches de {args.batch_size}...\n")

    with torch.no_grad():
        while saved < remaining:
            current_batch = min(args.batch_size, remaining - saved)

            # Amostra aleatória do espaço latente
            z    = torch.randn(current_batch, args.latent_dim).to(device)
            fake = decoder(z).detach().cpu()   # [B, 1, H, W]

            for j in range(current_batch):
                try:
                    mask_img = _tensor_to_mask(fake[j], args.output_size, args.threshold)

                    # Descarta máscaras triviais (quase totalmente pretas ou brancas)
                    coverage = np.count_nonzero(mask_img) / mask_img.size
                    if coverage < 0.02 or coverage > 0.98:
                        continue

                    name     = str(uuid.uuid4())[:8]
                    out_path = os.path.join(args.out_dir, f"{name}_mask.png")
                    cv2.imwrite(out_path, mask_img)
                    saved += 1

                    if saved % 100 == 0 or saved == remaining:
                        print(f"  {saved + already_done:>5}/{args.num_masks} máscaras salvas...")

                except Exception as e:
                    errors += 1
                    print(f"  [WARN] Erro ao processar máscara: {e}")

    # ---- Resumo ------------------------------------------------------------
    total_in_dir = _count_existing(args.out_dir)
    print("\n" + "=" * 60)
    print("Geração de máscaras concluída.")
    print(f"  Salvas nesta execução  : {saved}")
    print(f"  Erros / descartes      : {errors}")
    print(f"  Total no diretório     : {total_in_dir}")
    print(f"  Diretório              : {os.path.abspath(args.out_dir)}")
    print("=" * 60)
    print()
    print("PRÓXIMO PASSO:")
    print(f"  1. Revise as máscaras em:  {os.path.abspath(args.out_dir)}")
    print(f"  2. Remova as indesejadas")
    print(f"  3. Execute o passo 2:")
    print()
    print(f"     python textureSSD/scripts/step2_synthesize_from_masks.py \\")
    print(f"         --masks_dir {args.out_dir} \\")
    print(f"         --sample_path textureSSD/tgs_salt/0bdd44d530.png \\")
    print(f"         --sample_semantic_mask_path textureSSD/tgs_salt/0bdd44d530Mask.png \\")
    print(f"         --out_dir result/images_synthesized")
    print()


if __name__ == "__main__":
    main()
