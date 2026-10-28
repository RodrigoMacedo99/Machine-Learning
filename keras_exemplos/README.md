## Structured data classification

Projeto baseado no exemplo oficial do Keras para classificacao binaria usando o dataset de doencas cardiacas.

### Executar

Na raiz deste projeto:

```bash
uv sync
uv run structured-data-classification-from-scratch --epochs 50
```

O modelo treinado sera salvo em `models/heart_disease_classifier.keras`.

Tambem e possivel executar o launcher legado:

```bash
uv run src/setup.py --epochs 1
```
