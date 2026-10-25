# Deep Learning

Atividades de redes neurais com TensorFlow do curso de Machine Learning (CIMATEC).

## Estrutura

```
Deep_Learning/
├── notebooks/
│   └── atividade_tensorflow.ipynb   <- MLP para dígitos (Aulas 12 e 13: Keras, fit e backpropagation com GradientTape)
├── pyproject.toml                   <- dependências (tensorflow, scikit-learn, matplotlib, jupyterlab)
└── README.md
```

## Como executar

```bash
uv sync
uv run jupyter lab notebooks/atividade_tensorflow.ipynb
```

O notebook usa o `load_digits` do scikit-learn, que já vem com o pacote, então não há download de dados.
