# Visão Computacional

Atividades de visão computacional do curso de Machine Learning (CIMATEC).

## Notebooks

- `notebooks/Classificador_Imagens_v1.ipynb`: primeira CNN (Aula 14), sobre o CIFAR-10.
- `notebooks/Classificador_Imagens_v2.ipynb`: CNN com três blocos convolucionais, BatchNorm, aumento de dados, dropout, ReduceLROnPlateau e EarlyStopping.

## Como executar

```bash
uv sync
uv run jupyter lab notebooks/
```

O CIFAR-10 é baixado pelo `tf.keras.datasets` na primeira execução. O TensorFlow usa a GPU se houver CUDA disponível (`tensorflow[and-cuda]`).

## Estrutura (Cookiecutter Data Science)

```
visao_computacional/
├── data/            <- raw, interim, processed e external (não versionados)
├── docs/            <- documentação (Sphinx)
├── models/          <- modelos treinados (não versionados)
├── notebooks/       <- Classificador_Imagens v1 e v2
├── references/      <- material de apoio
├── reports/         <- figuras e relatórios gerados
├── src/             <- código do projeto
├── Makefile         <- `make requirements` (uv sync), `make clean`, `make lint`
├── pyproject.toml   <- dependências, gerenciadas com uv
└── README.md
```
