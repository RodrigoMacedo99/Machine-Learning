# Visão Computacional

Atividades de visão computacional do curso de Machine Learning (CIMATEC).

## Estrutura

```
visao_computacional/
├── notebooks/
│   ├── Classificador_Imagens_v1.ipynb   <- primeira CNN (Aula 14), CIFAR-10
│   └── Classificador_Imagens_v2.ipynb   <- CNN com aumento de dados, batch norm e dropout
├── pyproject.toml                       <- dependências (tensorflow, matplotlib, scikit-learn, jupyterlab)
└── README.md
```

## Como executar

```bash
uv sync
uv run jupyter lab notebooks/
```

O CIFAR-10 é baixado pelo `tf.keras.datasets` na primeira execução.
