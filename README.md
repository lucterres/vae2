# VAE para Geração de Máscaras Sísmicas (TGS Salt)

Implementação de um VAE com seleção de decoder via Multi-Armed Bandit (MabVAE) para geração sintética de máscaras de sal em dados sísmicos do dataset TGS Salt.

## Estrutura do Projeto

```
vae2/
├── MVAE.py               # Arquitetura: Encoder, Decoder_MLP, Decoder_Conv, MabVAE
├── train.py              # Script de treinamento (recomendado para Windows)
├── generate_masks.py     # Geração de máscaras a partir do checkpoint
├── test_generate.py      # Teste rápido sem modelo treinado
├── maskCustomDS.ipynb    # Notebook alternativo de treinamento
├── dataLoader.py         # Utilitários de carregamento de dados
├── constDirectories.py   # Constantes de caminhos
├── save_checkpoint.py    # Utilitário para salvar checkpoints
├── requirements.txt      # Dependências
└── data/
    ├── train.csv
    └── saltMaskOk.csv
```

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uso Rápido

### 1. Preparar o CSV de IDs

```powershell
$files = Get-ChildItem 'D:/dataset/tgs-salt/train/mask10k' |
         ForEach-Object { '"' + $_.BaseName + '",True' }
@('id,pass') + $files | Set-Content 'D:/dataset/tgs-salt/train/mask10k_files.csv'
```

### 2. Treinar o modelo

```powershell
# CPU básica (≤8GB RAM)
python train.py --batch 16

# CPU com +RAM (≥16GB)
python train.py --batch 64 --workers 4

# GPU NVIDIA
python train.py --batch 128 --workers 4 --gpu

# Retomar treino interrompido
python train.py --resume
```

| Hardware | Tempo (~20 épocas) |
|----------|--------------------|
| CPU básica | ~60–120 min |
| CPU ≥16GB RAM | ~30–60 min |
| GPU NVIDIA | ~7–20 min |

### 3. Gerar máscaras

```powershell
python generate_masks.py
```

Saída: `generated_masks/` com 100 máscaras PNG (101×101 px)

## Arquitetura

```
Imagem (1×64×64)
     │
  Encoder (Conv2d)
     │
  z ~ N(μ, σ²)   ← espaço latente (dim=100)
     │
  Decoder_MLP     ← selecionado via ε-greedy (MAB)
  [256→512→1024]
     │
  Imagem gerada (1×64×64)
     │
  GaussianBlur + Threshold
     │
  Máscara binária (1×101×101)
```

## Parâmetros de Treinamento

| Parâmetro | Valor |
|-----------|-------|
| Latent dim | 100 |
| Image size | 64×64 px |
| Canais | 1 (grayscale) |
| Hiddens | [256, 512, 1024] |
| Batch size | 32 |
| Épocas | 20 |
| Epsilon (MAB) | 0.3 |

## Dataset

- **Fonte:** [TGS Salt Identification Challenge (Kaggle)](https://www.kaggle.com/c/tgs-salt-identification)
- **Máscaras usadas:** ~11.600 imagens de `mask10k`
- **Formato:** PNG 101×101 px, escala de cinza

## Treinamento no Windows

> O notebook Jupyter pode crashar com datasets grandes no Windows por limitações de multiprocessing.
> Use `train.py` que resolve isso com `num_workers` correto e `if __name__ == '__main__'`.

Ver [QUICKSTART.md](QUICKSTART.md) para instruções detalhadas.
