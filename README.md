# Transfer Learning e Fine-Tuning em CNN — Classificação de Resíduos Sólidos (RealWaste)

Atividade da disciplina **Inteligência Computacional / Machine Learning** (CESUPA) — Profa. Polyana Santos Fonseca Nascimento.

## Integrantes
- Andrey Lourival Andrade Garcia
- Andrey de Matos Gonçalves
- Igor Cecim Vilhena

## Resumo do Projeto
- **Arquitetura:** `MobileNetV2` (`torchvision.models.mobilenet_v2`, pesos `IMAGENET1K_V2` — Sandler et al., CVPR 2018).
- **Dataset:** [RealWaste (UCI ML Repository, id 908)](https://archive.ics.uci.edu/dataset/908/realwaste) — `4.752` imagens reais (`524×524`, RGB) distribuídas em `9` classes de resíduos.
- **Divisão dos dados:** Holdout estratificado `70 / 15 / 15` (`3.326` treino, `713` validação, `713` teste isolado) registrado em `splits.csv`.
- **Condições comparadas:**
  1. **Base congelada** (`11.529` parâmetros treináveis no classificador) — condição de referência (`frozen.pt`)
  2. **Fine-tuning A** (`features[14:]` descongelado, `1.692.873` parâmetros treináveis, `lr = 1e-4`)
  3. **Fine-tuning B** (base inteira descongelada, `2.235.401` parâmetros treináveis, `lr = 1e-4`) — **modelo escolhido**
- **Explicabilidade:** Grad-CAM próprio com hooks na última convolução (`features[18]`, mapa `7×7`).

## Resultados Principais

| Condição | F1-macro (Val) | F1-macro (Teste) | Recicláveis perdidos (Val / Teste) | Contaminações (Val / Teste) |
|---|---:|---:|---:|---:|
| Base congelada | `0,835` | `0,831` | `20` / `22` | `11` / `12` |
| Fine-tuning A | `0,925` | `0,905` | `7` / `11` | `13` / `14` |
| **Fine-tuning B** | **`0,933`** | **`0,924`** | **`5`** / **`8`** | **`10`** / **`8`** |

- **Inferência externa (`02_inferencia.ipynb`):** `imagens_externas/lata_keystone.jpg` (Tony Webster, Wikimedia Commons, CC BY 2.0) classificada como **`Metal`** (score `0,59`).

## Estrutura do Repositório

```text
README.md                        # Guia de uso e resumo da entrega
ROTEIRO.md                       # Roteiro experimental e checklist da equipe
01_cnn_transfer_learning.ipynb   # Pipeline completo (Etapas 1 a 7, com saídas salvas)
02_inferencia.ipynb              # Carregamento do modelo salvo e previsão externa
splits.csv                       # Divisão estratificada fixa (train / val / test)
requirements.txt                 # Dependências Python 3.12 (PyTorch CPU + visão/dados)
imagens_externas/                # Imagem externa usada no notebook de inferência
```

> Os diretórios `data/`, `models/` e `.venv/` estão no `.gitignore` para não versionar o dataset bruto nem arquivos `.pt` no Git.

## Como Executar Localmente (Windows ou Linux/Ubuntu)

1. **Criar ambiente Python 3.12 e instalar dependências:**
   ```bash
   python -m venv .venv
   # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
   # Linux/macOS:          source .venv/bin/activate
   pip install -r requirements.txt
   ```
   *(No Windows com múltiplas versões do Python instaladas, use `py -3.12 -m venv .venv`.)*

2. **Executar apenas a inferência (`02_inferencia.ipynb`):**
   - Coloque o arquivo `realwaste_mobilenetv2.pt` (disponibilizado na entrega/Drive) dentro de `models/realwaste_mobilenetv2.pt`.
   - Abra e execute `02_inferencia.ipynb` usando o kernel da `.venv` (não requer o dataset RealWaste nem acesso à internet).

3. **Reexecutar o pipeline completo (`01_cnn_transfer_learning.ipynb`):**
   - Se `models/` contiver `frozen.pt`, `finetune_A.pt`, `finetune_B.pt` e seus arquivos `*_history.csv` / `*_info.json` (com `FORCE_RETRAIN = False`), o notebook carrega os checkpoints e executa todas as análises em alguns minutos.
   - Se `data/realwaste-main/RealWaste` não existir, a Célula 3 baixa e extrai automaticamente o dataset da UCI (~656 MB). Sem `models/`, o notebook usa GPU (`cuda`) automaticamente se disponível ou roda em CPU (~1h46min no ambiente original).

> **Nota sobre Google Colab:** o Colab já traz PyTorch com suporte a GPU (T4), NumPy, Pandas, Pillow, Matplotlib e Scikit-learn pré-instalados, e o notebook seleciona `cuda` automaticamente se houver GPU ativa. Não rode `pip install -r requirements.txt` no Colab (para não substituir o PyTorch GPU pela build `+cpu`). Para rodar o `02_inferencia.ipynb` lá, clone o repositório e envie `models/realwaste_mobilenetv2.pt` para `models/`.
