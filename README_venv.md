# Configuração do Ambiente Virtual — Terminal Científico CENPES

## Overview

Ambiente virtual Python configurado para execução do projeto no **terminal científico do CENPES (Petrobras)**,  com suporte a GPU via **PyTorch CUDA 12.1**.

A máquina dispõe de GPU virtualizada **NVIDIA GRID T4-4Q** (4 GB) acessível via driver WDDM 539.19.

> **Localização do venv:** `C:\Users\<user>\AppData\Local\venvs\vae2`  
> O venv **não** fica dentro do repositório (`F:\code\vae2`) porque `F:` é um **drive de rede (DFS Petrobras)**.  
>
> ⚠️ **Diferença de desempenho medida nesta máquina:**
>
> | Drive | Tipo | Tempo de instalação completa do venv |
> |-------|------|--------------------------------------|
> | `C:\AppData\Local` | Disco local (SSD) | ~15 minutos |
> | `F:\code\vae2\.venv` | Drive de rede DFS | ~3 horas |
>
> A diferença se deve à latência de rede do DFS para cada operação de escrita de arquivo — o pip grava milhares de pequenos arquivos (`.py`, `.pyc`, `.pyd`, metadados) durante a instalação, tornando a operação ~12× mais lenta no drive de rede.

---

## Hardware

| Item | Detail |
|------|--------|
| GPU | NVIDIA GRID T4-4Q |
| VRAM | 4 GB |
| Driver | 539.19 |
| Max CUDA | 12.2 |

---

## Python & Environment

| Item | Detail |
|------|--------|
| Python | **3.12.10** |
| Environment | venv local (no repositório apenas `.venv` simbólico) |
| Location | `C:\Users\<user>\AppData\Local\venvs\vae2` |

> **Por que Python 3.12 e não 3.13?**  
> O PyTorch não publica wheels `+cu121` para Python 3.13. Os builds `+cu124`/`+cu126` para Python 3.13 exigem driver ≥ 560, e o driver instalado é 539. Portanto, Python 3.12 é necessário para ter CUDA com o driver atual.

---

## Creating the environment

```powershell
# Usar Python 3.12 via py launcher
# O venv vai para C:\AppData\Local para evitar lentidão do drive de rede F:
py -3.12 -m venv "$env:LOCALAPPDATA\venvs\vae2" --without-pip

# Bootstrap pip (ensurepip usa wheel local, não precisa de rede)
& "$env:LOCALAPPDATA\venvs\vae2\Scripts\python.exe" -m ensurepip --upgrade
```

> `--without-pip` é necessário neste ambiente corporativo porque o processo de setup do pip via `venv` é interrompido pelo proxy antes de concluir.

---

## Installing packages

> **Important:** `requirements.txt` lists generic package names/versions.  
> For GPU support, `torch` and `torchvision` **must** be installed with the `+cu121` variant from the PyTorch wheel index — **not** from PyPI (which delivers the CPU-only `+cpu` build by default).

### 1. Torch + Torchvision (CUDA 12.1)

> **Crítico:** usar `--isolated` para ignorar o `pip.ini` corporativo que aponta para o Nexus.  
> Sem `--isolated`, o Nexus intercepta o `--index-url` e retorna apenas o build `+cpu`.

```powershell
& "$env:LOCALAPPDATA\venvs\vae2\Scripts\python.exe" -m pip install `
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" `
    --index-url https://download.pytorch.org/whl/cu121 `
    --isolated
```

> **Por que 2.5.1 e não 2.10.0?**  
> `torch==2.10.0+cu121` não existe. A versão máxima disponível no índice `cu121` para Python 3.12 é `2.5.1`.

### 2. Remaining dependencies

```powershell
& "$env:LOCALAPPDATA\venvs\vae2\Scripts\python.exe" -m pip install `
    pytorch-lightning==2.6.1 pandas==3.0.0 matplotlib==3.10.8 `
    opencv-python==4.13.0.92 pillow==12.1.1 `
    --index-url https://nexus.petrobras.com.br/nexus/repository/pypi-all/simple `
    --trusted-host nexus.petrobras.com.br
```

> Os demais pacotes não são afetados pelo Nexus — o build `cpu`/`gpu` só importa para `torch` e `torchvision`.

---

## Verifying GPU availability

```python
import torch
print(torch.__version__)              # torch: 2.5.1+cu121
print(torch.cuda.is_available())      # True
print(torch.cuda.device_count())      # 1
print(torch.cuda.get_device_name(0))  # GRID T4-4Q
```

---

## Package list

| Package | Version instalada | Notes |
|---------|------------------|-------|
| torch | **2.5.1+cu121** | CUDA build — `--index-url https://download.pytorch.org/whl/cu121` + `--isolated` |
| torchvision | **0.20.1+cu121** | CUDA build — mesma flag |
| pytorch-lightning | 2.6.1 | via Nexus |
| numpy | 2.3.5 | via Nexus (2.4.2 não disponível para Py3.12 no Nexus) |
| pandas | 3.0.0 | via Nexus |
| matplotlib | 3.10.8 | via Nexus |
| opencv-python | 4.13.0.92 | via Nexus |
| pillow | 12.0.0 | instalada como dependência do torch |

---

## Armadilhas comuns

| Problema | Causa | Solução |
|----------|-------|---------|
| `torch.cuda.is_available() == False` | `torch+cpu` instalado pelo Nexus | Usar `--index-url https://download.pytorch.org/whl/cu121 --isolated` |
| `Could not find torch==2.10.0+cu121` | Versão não existe para Py3.12 | Versão máxima cu121/Py3.12 é **2.5.1** |
| `Could not find torch+cu121` (qualquer versão) | Python 3.13 não tem wheels cu121 | Usar **Python 3.12** |
| `ensurepip` trava ao criar venv | Proxy corporativo interrompe subprocess | Usar `--without-pip` + bootstrapar depois |
| Venv lento / travado | Drive de rede `F:` | Criar em `$env:LOCALAPPDATA\venvs\` |
