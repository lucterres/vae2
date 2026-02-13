# Gerador de Máscaras Sintéticas

Script para gerar máscaras sintéticas usando o modelo VAE treinado.

## Uso Básico

```powershell
# Ativar ambiente virtual
.\.venv\Scripts\Activate.ps1

# Executar geração de máscaras
python generate_masks.py
```

## Configurações

Edite as variáveis no `generate_masks.py`:

```python
NUM_SAMPLES = 100           # Número de máscaras a gerar
OUTPUT_DIR = './generated_masks'  # Diretório de saída
CHECKPOINT_PATH = None      # Caminho do checkpoint (se disponível)
```

## Carregar Modelo Treinado

Se você já treinou o modelo no notebook, pode salvar o checkpoint e carregá-lo:

### No notebook (após treinar):

```python
# Salvar checkpoint
torch.save({
    'decoder_state_dict': MabVae.decoders[0].state_dict(),
    'encoder_state_dict': MabVae.encoder.state_dict(),
}, 'vae_checkpoint.pth')
```

### No script:

```python
CHECKPOINT_PATH = './vae_checkpoint.pth'
```

E adicione no método `load_model()`:

```python
checkpoint = torch.load(checkpoint_path)
Decoders[0].load_state_dict(checkpoint['decoder_state_dict'])
```

## Saída

O script gera:
- 100 máscaras PNG no diretório `./generated_masks/`
- Uma visualização em grade: `visualization_grid.png`

## Estrutura do Workflow

1. **Carrega modelo** (decoder MLP)
2. **Gera ruído latente** (z ~ N(0,1))
3. **Decodifica** para imagens 64x64
4. **Aplica transformações**:
   - Resize para 101x101
   - Gaussian Blur (5x5)
   - Threshold binário (127)
5. **Salva máscaras** como PNG

## Exemplo de Uso Avançado

```python
from generate_masks import generate_masks, load_model

# Carrega modelo
decoders = load_model('vae_checkpoint.pth')

# Gera 500 máscaras customizadas
generate_masks(
    decoder=decoders[0],
    num_samples=500,
    output_dir='./custom_output'
)
```
