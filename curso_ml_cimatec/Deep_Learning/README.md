# Deep Learning

Atividades de redes neurais com TensorFlow do curso de Machine Learning (CIMATEC).

## Notebooks

- `notebooks/atividade_tensorflow.ipynb`: MLP para classificar dígitos (Aulas 12 e 13: Keras, `fit` e backpropagation com `GradientTape`). Usa o `load_digits` do scikit-learn, que já vem com o pacote.

## Como executar

```bash
uv sync
uv run jupyter lab notebooks/atividade_tensorflow.ipynb
```

O TensorFlow usa a GPU se houver CUDA disponível (`tensorflow[and-cuda]`). A célula de configuração no início do notebook pré-carrega as bibliotecas CUDA do pacote `nvidia`.

## Estrutura (Cookiecutter Data Science)

```
Deep_Learning/
├── data/            <- raw, interim, processed e external (não versionados)
├── docs/            <- documentação (Sphinx)
├── models/          <- modelos treinados (não versionados)
├── notebooks/       <- notebooks de atividades
├── references/      <- material de apoio
├── reports/         <- figuras e relatórios gerados
├── src/             <- código do projeto
├── Makefile         <- `make requirements` (uv sync), `make clean`, `make lint`
├── pyproject.toml   <- dependências, gerenciadas com uv
└── README.md
```
