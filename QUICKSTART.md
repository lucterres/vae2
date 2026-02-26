# Quick Start - Geração de Máscaras

## Teste Rápido (sem treinar)

```powershell
# Gera 5 máscaras de teste para validar o script
python test_generate.py
```

Saída: `test_masks/` com 5 máscaras (modelo não treinado)

## 1. Salvar modelo treinado (no notebook)

Execute a nova célula após treinar:
```python
# Célula adicionada automaticamente ao notebook
# Salva checkpoint após `trainer.fit(MabVae)`
```

Isso cria o arquivo `vae_checkpoint.pth`.

## 2. Gerar 100 máscaras

```powershell
# Na raiz do projeto
python generate_masks.py
```

**Saída:**
- `generated_masks/` - 100 máscaras PNG (101x101px)
- `generated_masks/visualization_grid.png` - Preview

## 3. Customizar

Edite `generate_masks.py`:

```python
NUM_SAMPLES = 500  # Gerar 500 em vez de 100
OUTPUT_DIR = './my_masks'  # Diretório customizado
```

## Retreinar o Modelo

### Pré-requisito: gerar o CSV de IDs

```powershell
# Lista nomes de arquivo sem extensão, preservando como texto (colunas: id, pass)
$files = Get-ChildItem 'D:/dataset/tgs-salt/train/mask10k' |
         ForEach-Object { '"' + $_.BaseName + '",True' }
@('id,pass') + $files | Set-Content 'D:/dataset/tgs-salt/train/mask10k_files.csv'
```

---

### Opção A — Script `train.py` ✓ Recomendado

Mais estável no Windows: usa `num_workers > 0` e não causa crash de kernel.

```powershell
# Treino padrão: 20 épocas, batch=32, workers=2
python train.py

# Mais épocas
python train.py --epochs 50

# Batch maior (mais RAM, mais rápido)
python train.py --batch 64

# Mais workers no DataLoader
python train.py --workers 4

# Retomar treino interrompido
python train.py --resume
```

**Argumentos disponíveis:**

| Argumento | Padrão | Descrição |
|-----------|--------|-----------|
| `--epochs` | 20 | Número de épocas |
| `--batch` | 32 | Tamanho do batch |
| `--workers` | 2 | Workers do DataLoader |
| `--resume` | False | Retoma do último checkpoint Lightning |

**Saída:** `vae_checkpoint.pth` salvo automaticamente ao final.

**Checkpoints por época** (para retomada com `--resume`):
```
lightning_logs/version_X/checkpoints/epoch=XX.ckpt
```

---

### Opção B — Notebook `maskCustomDS.ipynb`

> ⚠ No Windows, use `num_workers=0` no DataLoader para evitar crash de kernel.

Edite os caminhos na célula de configuração:

```python
TRAIN_CSV      = 'D:/dataset/tgs-salt/train/mask10k_files.csv'
TRAIN_MASK_DIR = 'D:/dataset/tgs-salt/train/mask10k'
```

Execute as células em sequência até a célula de salvamento do checkpoint.

---

### Recomendações por hardware (20 épocas, ~11.600 amostras)

| Hardware | Comando | Tempo estimado |
|----------|---------|---------------|
| CPU básica (≤8GB RAM) | `python train.py --batch 16` | ~60–120 min |
| CPU com +RAM (≥16GB) | `python train.py --batch 64 --workers 4` | ~30–60 min |
| GPU NVIDIA (CUDA) | `python train.py --batch 128 --workers 4 --gpu` | ~7–20 min |
| GPU NVIDIA + muita RAM | `python train.py --batch 256 --workers 8 --gpu` | ~4–10 min |

> **`--workers`**: use número de núcleos físicos / 2 (ex: CPU 8 núcleos → `--workers 4`).  
> **`--gpu`**: requer CUDA instalado. Ignore se não tiver GPU NVIDIA.  
> **Windows com pouca RAM**: mantenha `--batch 16 --workers 0` para evitar crashes.

Para verificar sua GPU:
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'sem GPU')"
```

---

## Workflow Reproduzido

1. ✓ CustomImageDataset
2. ✓ ThresholdTransform
3. ✓ Decoder MLP (latent_dim=100, hiddens=[256,512,1024])
4. ✓ Geração de ruído latente
5. ✓ Decodificação
6. ✓ Transformações: Resize(101,101) → GaussianBlur → Threshold(127)
7. ✓ Salva PNG

---

**Requisitos:**
- Modelo treinado (`vae_checkpoint.pth`)
- MVAE.py no diretório
- Ambiente virtual ativado
