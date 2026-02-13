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
