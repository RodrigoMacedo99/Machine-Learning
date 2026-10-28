# Visão Computacional

Atividades de visão computacional do curso de Machine Learning (CIMATEC): classificação de imagens do CIFAR-10 com redes convolucionais (CNN).

- **v1** (Aula 14): CNN simples, usada como linha de base. Teste: ~69,6%.
- **v2** (Aula 21): CNN com três blocos convolucionais, BatchNorm, aumento de dados, dropout, ReduceLROnPlateau e EarlyStopping. Teste: ~86,6%.

## Como executar

```bash
uv sync
uv run python -m src.models.train_model --version v2   # treina, avalia e salva o modelo
uv run python -m src.models.train_model --version v1
```

O treino salva o modelo em `models/cifar10_<versão>.keras` e as métricas de teste em `reports/metrics_<versão>.json`. O CIFAR-10 é baixado pelo `tf.keras.datasets` na primeira execução.

O TensorFlow usa a GPU se houver CUDA disponível (`tensorflow[and-cuda]`). `src/gpu.py` pré-carrega as bibliotecas CUDA antes do import do TensorFlow.

Opções: `--epochs`, `--batch-size`, `--model-dir` e `--report-dir`. Sem `--epochs`, a v1 roda 8 épocas e a v2, 40.

Para explorar o treino passo a passo, os notebooks usam o mesmo código de forma didática:

```bash
uv run jupyter lab notebooks/
```

## Código

```
src/
├── gpu.py                <- pré-carrega as bibliotecas CUDA
├── data/cifar.py         <- carrega o CIFAR-10 e separa a validação do teste
├── models/cnn.py         <- arquiteturas v1 e v2
└── models/train_model.py <- CLI de treino, avaliação e gravação do modelo
```

## Estrutura (Cookiecutter Data Science)

```
visao_computacional/
├── data/            <- raw, interim, processed e external (não versionados)
├── docs/            <- documentação (Sphinx)
├── models/          <- modelos treinados (.keras, não versionados)
├── notebooks/       <- Classificador_Imagens v1 e v2 (versões didáticas)
├── references/      <- material de apoio
├── reports/         <- métricas e figuras geradas
├── src/             <- código do projeto
├── Makefile         <- `make requirements` (uv sync), `make clean`, `make lint`
├── pyproject.toml   <- dependências, gerenciadas com uv
└── README.md
```
