# Roteiro — Transfer Learning e Fine-Tuning em CNN

Lauda: `Lauda CNN 2026_2.pdf` (Profa. Polyana — Inteligência Computacional/ML)

**Tema:** Classificação de imagens de resíduos sólidos para triagem e reciclagem automatizada.
**Target:** classe do material do resíduo.

## 0. Decisões já tomadas / pendentes

| Item | Status |
|---|---|
| Dataset | **RealWaste** — UCI ML Repository (id 908), licença CC BY-NC-SA 4.0 |
| Arquitetura | **MobileNetV2** (Sandler et al., CVPR 2018) — torchvision, pesos `IMAGENET1K_V2` |
| Framework | PyTorch + torchvision (pesos e `transforms` oficiais já vêm juntos) |
| Ambiente | **Local, CPU** (`.venv` + kernel Jupyter "Python (trabalho CNN)"). Colab só se o fine-tuning B ficar lento demais |
| Integrantes | ⏳ |
| Datas (defesa intermediária / entrega) | ⏳ |

### O dataset (conferido)

- Fonte: https://archive.ics.uci.edu/dataset/908/realwaste
- Artigo: Single, S. et al. *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*. Information, 14(12), 633, 2023.
- Fotos reais de resíduos na esteira do aterro Whyte's Gully (Wollongong, Austrália). 524×524, RGB.
- **4.752 imagens, 9 classes:**

| Classe | Imagens |
|---|---|
| Plastic | 921 |
| Metal | 790 |
| Paper | 500 |
| Miscellaneous Trash | 495 |
| Cardboard | 461 |
| Vegetation | 436 |
| Glass | 420 |
| Food Organics | 411 |
| Textile Trash | 318 |

- Desbalanceamento: maior/menor classe ≈ 2,9×. Menor classe (318) atende ao mínimo de 100.
- Sem duplicatas exatas (checagem por CRC: 0 repetidas).
- Por que este e não o `archive.zip` (Kaggle O/R): aquele é binário (orgânico × reciclável), não classifica material,
  é muito documentado e tem 285 imagens repetidas entre treino e teste.

### Argumento para a validação com a professora

Fotos de campo, não de estúdio. Dataset recente (2023), de repositório confiável, com artigo publicado e pouco explorado
em notebooks prontos. Classes com confusão real (Paper × Cardboard, Plastic × Misc Trash), boas para matriz de confusão e Grad-CAM.

---

## 1. Arquitetura e dataset (Etapa 1)

- [ ] Inscrever a equipe na planilha (garantir família + versão).
- [ ] Pesquisa da arquitetura: autores/ano/artigo, diagrama dos blocos, inovação vs CNN convencional, diferencial da versão.
- [ ] Contexto: triagem automatizada em esteira de reciclagem — câmera classifica o material e direciona o resíduo.

**Verificar:** cada integrante consegue explicar o diagrama em 2 minutos sem ler.

## 2. Pré-processamento e análise (Etapa 2)

1. Montar tabela `caminho, classe`.
2. Análise inicial: contagem por classe (gráfico de barras), grade de exemplos por classe, checagem de tamanho/canais.
3. **Holdout estratificado 70/15/15** com seed fixa. Salvar `splits.csv`.
   O teste (~713 imagens, ~48 de Textile) é separado aqui e **não é tocado até a Etapa 7**.
4. Pré-processamento: `weights.transforms()` do torchvision (resize 224, normalização ImageNet, RGB).
   Data augmentation **só no treino**: flip horizontal/vertical, rotação, color jitter leve (resíduo na esteira não tem orientação fixa).
5. Desbalanceamento: **CrossEntropy com peso por classe** (inverso da frequência no treino).
6. Métricas:
   - Principal (seleção de modelo): **F1-macro** na validação — cada classe pesa igual, Textile não some atrás de Plastic.
   - Apoio: acurácia, balanced accuracy, precisão/recall por classe, matriz de confusão.
   - Erros mais graves no contexto:
     - **Reciclável indo para rejeito/orgânico** (Glass, Metal, Plastic, Paper, Cardboard → Misc Trash/Food/Vegetation): perde material de valor.
     - **Rejeito ou orgânico indo para reciclável** (Food Organics/Misc Trash → Paper/Cardboard/Plastic): contamina o lote.
     - Acompanhar **recall** das classes recicláveis e **precisão** delas.

**Verificar:** assert de que nenhum arquivo aparece em dois conjuntos; proporção das classes igual nos três.

## 3. Base congelada (Etapa 3) — condição de referência

- Carregar arquitetura com pesos ImageNet, congelar toda a base, trocar o classificador por `Dropout + Linear(n_features, 9)`.
- Hiperparâmetros: Adam, lr 1e-3, batch 32, até 15 épocas; checkpoint e early stopping pelo F1-macro de validação (paciência 4).
- BatchNorm das camadas congeladas em modo `eval` durante o treino.
- Registrar: arquitetura adaptada, entrada/saída, parâmetros totais vs treináveis, hiperparâmetros, tempo.
- Curvas loss + F1-macro (treino × validação) e análise de over/underfitting.
- Salvar `frozen.pt` (melhor época na validação).

**Verificar:** nº de parâmetros treináveis = só o classificador.

## 4. Fine-tuning A e B (Etapa 4)

As duas partem de **cópias de `frozen.pt`**, mesmos dados, seed, métricas, batch, nº máx. de épocas e regra de early stopping.

| | Estratégia A | Estratégia B |
|---|---|---|
| Ideia | Descongelar `features[14:]` (estágio final 7×7) | Descongelar a base inteira |
| LR | 1e-4 | 1e-4 (igual a A: só a profundidade do descongelamento muda) |
| Hipótese | Ajusta features de alto nível com pouco risco de esquecer o ImageNet | Mais capacidade de adaptação ao domínio (lixo sujo, amassado), mais risco de overfitting com ~3,3 mil imagens de treino e mais custo |


**Verificar:** nº de parâmetros treináveis A < B, ambos > referência.

## 5. Comparação e análise visual de erros (Etapa 5)

- Tabela: métricas de validação, parâmetros treináveis, épocas, tempo de treino — das 3 condições.
- Curvas lado a lado + 3 matrizes de confusão (9×9).
- Galeria de erros e acertos: imagem, classe real, prevista, score — pelo menos um exemplo por classe.
- **Mesmas imagens nos 3 modelos** (fixar ~12 imagens da validação): erros corrigidos, persistentes e novos.
- Discussão: houve ganho? quais classes melhoraram/pioraram? overfitting? o custo extra compensou?
  Ambiguidades de rótulo: papel × papelão, plástico × misc trash, objetos mistos (embalagem com comida).

## 6. Explicabilidade (Etapa 6)

- **Grad-CAM** na última camada convolucional, implementado com hooks (~20 linhas, sem dependência nova e fácil de explicar na arguição).
- Mesmas imagens da Etapa 5 nos 3 modelos; informar camada e classe investigada.
- Analisar se o destaque está no objeto, na esteira/fundo ou em outros objetos da cena.

## 7. Modelo escolhido, teste e uso (Etapa 7)

1. Escolher o modelo **só pela validação** e justificar (desempenho × custo × erros graves).
2. Só então rodar os **3 modelos no teste**, uma vez. Comparar com a validação. Não ajustar nada depois disso.
3. Salvar o escolhido: `state_dict` + nome da arquitetura + lista ordenada das 9 classes + config de pré-processamento.
4. Notebook `02_inferencia.ipynb`: carregar sem treinar, prever uma imagem da internet (mostrar imagem, fonte, classe esperada, previsão, score).

---

## Entregas

- [ ] `01_cnn_transfer_learning.ipynb` — integrantes, **declaração de uso de IA Generativa** (ferramenta, finalidade, extensão, validação), todas as etapas, markdown intercalado, executado em ordem com saídas salvas, dependências listadas.
- [ ] `02_inferencia.ipynb` — carregamento + imagem externa.
- [ ] Modelo salvo (link do Drive no Classroom — **não versionar no git**).
- [ ] Slides (máx. 12 min, todos falam, foco em narrativa e resultados, não em código).

## Checklist da defesa intermediária

- [ ] Arquitetura pesquisada (diagrama + blocos + inovação + diferencial da versão)
- [ ] Dados analisados e preparados
- [ ] Teste reservado + métricas e divisão definidas
- [ ] Modelo congelado treinado, com curvas e resultados na validação
- [ ] Análise visual preliminar
- [ ] Estratégias A e B **planejadas**

## Estrutura do repositório

```
ROTEIRO.md
01_cnn_transfer_learning.ipynb
02_inferencia.ipynb
splits.csv          # split fixo, reprodutível
requirements.txt
.gitignore          # *.zip, RealWaste/, *.pt
```
