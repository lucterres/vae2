# Configuração do Cluster Atena — CENPES / Petrobras

## Visão Geral

Ambiente de treinamento do projeto **vae2** no cluster de IA **Atena** do CENPES.

---

## Hardware

| Item | Detalhe |
|------|---------|
| Nó | `atn2b01n02.petrobras.biz` |
| OS | Red Hat Enterprise Linux 8.10 |
| GPUs | **8× Tesla V100-SXM2-32GB** |
| VRAM por GPU | 32 GB |
| CUDA | 12.1 |

---

## Filesystems

| Caminho | Servidor | Tamanho | Uso |
|---------|----------|---------|-----|
| `/u/cym7` | `homeunix-rio.petrobras.biz` | 20 GB | Home do usuário |
| `/nethome/atena_projetos` | `srjcipdvfs50502.petrobras.biz` | 7.9 TB | DFS de projetos |

> **Repositório do projeto:** `/nethome/atena_projetos/cym7/code/vae2`  
> **Dataset TGS Salt:** `/nethome/atena_projetos/cym7/dataset/tgsSalt/`

---

## Mapeamento Windows → Linux (DFS)

| Windows | Linux |
|---------|-------|
| `\\dfs.petrobras.biz\cientifico\cenpes\atena_projetos\cym7` | `/nethome/atena_projetos/cym7` |
| `\\dfs.petrobras.biz\cientifico\cenpes\atena_projetos\cym7\dataset` | `/nethome/atena_projetos/cym7/dataset` |

---

## Dataset

```
/nethome/atena_projetos/cym7/dataset/tgsSalt/
├── train/
│   ├── images/
│   ├── masks/              ← 4000 máscaras originais TGS
│   ├── mask10k/
│   │   ├── mask10k/        ← 11.617 máscaras (extraídas de mask10k.tar)
│   │   └── mask10k_files.csv
│   └── mask10k.tar
├── test/
├── train.csv
├── depths.csv
└── tgsSalt.tar
```

---

## Ambiente Python (venv)

| Item | Detalhe |
|------|---------|
| Python | **3.12** (via conda/módulo separado do sistema) |
| venv | `/nethome/atena_projetos/cym7/code/vae2/.venv` |
| torch | `2.5.1+cu121` |
| torchvision | `0.20.1+cu121` |
| pytorch-lightning | `2.6.1` |

### Criar o venv

```bash
cd /nethome/atena_projetos/cym7/code/vae2
bash setup_venv_linux.sh
```

O script cria `.venv/`, instala torch com CUDA 12.1 e verifica a GPU automaticamente.

### Ativar o venv

```bash
source /nethome/atena_projetos/cym7/code/vae2/.venv/bin/activate
```

---

## Treinamento

```bash
cd /nethome/atena_projetos/cym7/code/vae2
source .venv/bin/activate

# Treino padrão (20 épocas, batch 32, 4 workers, GPU 0)
python train.py

# Batch e workers maiores (aproveitar as V100)
python train.py --batch 256 --workers 8 --epochs 50

# Retomar do último checkpoint Lightning
python train.py --resume
```

### Configurações em `train.py`

| Variável | Valor atual | Descrição |
|----------|------------|-----------|
| `TRAIN_CSV` | `data/saltMaskOk.csv` | CSV com IDs das máscaras de treino |
| `TRAIN_MASK_DIR` | `/nethome/atena_projetos/cym7/dataset/tgsSalt/train/masks` | Pasta das máscaras |
| `CHECKPOINT_OUT` | `vae_checkpoint.pth` | Checkpoint final no formato `generate_masks.py` |
| `accelerator` | `gpu` | Usar GPU |
| `devices` | `1` | Número de GPUs |

> Para usar múltiplas GPUs, altere `devices=N` e `accelerator='gpu'` em `train.py`.

---

## Checkpoints Lightning

Salvos automaticamente a cada época em:
```
lightning_logs/version_N/checkpoints/epoch=XX.ckpt
```

Para retomar o mais recente:
```bash
python train.py --resume
```

---

## Verificar GPU

```bash
nvidia-smi

# Ou via Python
source .venv/bin/activate
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_available())"
```

---

## Problemas conhecidos

| Problema | Causa | Solução |
|----------|-------|---------|
| `CUDA out of memory` | `retain_graph=True` no backward | Removido — não usar |
| `input_val >= 0 && input_val <= 1` (BCE assert) | Explosão de gradiente | Gradient clipping adicionado em `MVAE.py` (`clip_val=1.0`) |
| GPU 0 ocupada | Outro usuário no nó | Usar `CUDA_VISIBLE_DEVICES=1` (ou 2..7) |
| `python3.12` não encontrado | Sistema usa 3.8 | Verificar `which python3.12` ou usar conda |

---

## Gerar Máscaras

Após o treino:
```bash
python generate_masks.py
```

As imagens geradas são salvas em `outputs/`.
