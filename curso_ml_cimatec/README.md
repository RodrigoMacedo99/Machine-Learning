# Machine Learning (CIMATEC, GRD-ADS-2209)

Projetos práticos da disciplina de Machine Learning. Cada pasta é um projeto independente, com seu próprio `pyproject.toml` e ambiente virtual.

| Projeto | O que é |
|---|---|
| [`olist/`](#olist-previsão-de-atraso-em-pedidos) | Previsão de atraso em pedidos do Olist (dataset público de e-commerce brasileiro) |
| [`Deep_Learning/`](#deep-learning-mlp-com-tensorflow) | Rede neural densa (MLP) para classificar dígitos, com TensorFlow |
| [`visao_computacional/`](#visão-computacional-cnn-para-imagens) | Redes convolucionais (CNN) para classificar imagens do CIFAR-10 |
| [`reconhecimento_facial/`](#reconhecimento-facial-controle-de-acesso) | Script de controle de acesso por reconhecimento facial |

## Pré-requisitos

- [uv](https://docs.astral.sh/uv/) para gerenciar os ambientes e as dependências.
- Python 3.12.
- Opcional: GPU NVIDIA com driver instalado. Os projetos de TensorFlow usam a GPU quando ela está disponível e rodam na CPU caso contrário.

Em todos os projetos, o fluxo básico é o mesmo:

```bash
cd <nome-do-projeto>
uv sync                      # cria o .venv e instala as dependências
uv run jupyter lab           # abre os notebooks (quando o projeto tem)
```

---

## Olist: previsão de atraso em pedidos

Dataset público de e-commerce brasileiro da Olist (99 mil pedidos). O objetivo é prever se um pedido será entregue com atraso (`is_late`), a partir da data da compra, dos itens e do cliente.

O projeto tem um pipeline completo: geração de features, treino com validação cruzada, explicação com SHAP e active learning.

**Como rodar**

```bash
cd olist
uv sync

# 1. Baixe o dataset do Kaggle para data/raw/
uv run --with kagglehub python - <<'EOF'
import shutil
from pathlib import Path
import kagglehub

src = Path(kagglehub.dataset_download("olistbr/brazilian-ecommerce"))
dst = Path("data/raw")
dst.mkdir(parents=True, exist_ok=True)
for csv in src.glob("*.csv"):
    shutil.copy(csv, dst / csv.name)
print(sorted(p.name for p in dst.iterdir()))
EOF

# 2. Rode o pipeline inteiro
uv run python main.py                          # ou: make pipeline
uv run python main.py --skip-active-learning   # pula a etapa mais lenta
```

Os resultados ficam em `models/` (modelo treinado) e `reports/` (métricas, curvas e gráficos SHAP).

Testes: `uv run pytest tests -q`.

---

## Deep Learning: MLP com TensorFlow

Atividade das Aulas 12 e 13: uma rede neural densa (MLP) para classificar dígitos manuscritos, com Keras e, em seguida, o passo de treino escrito à mão com `tf.GradientTape` (backpropagation).

**Como rodar**

```bash
cd Deep_Learning
uv sync
uv run jupyter lab notebooks/atividade_tensorflow.ipynb
```

O dataset (`load_digits`) já vem com o scikit-learn, então não há download.

---

## Visão computacional: CNN para imagens

Atividade das Aulas 14 e 21: duas redes convolucionais treinadas no CIFAR-10 (60 mil imagens de 32x32 em 10 classes).

- **v1:** uma CNN simples, usada como linha de base.
- **v2:** uma CNN mais profunda, com BatchNorm, aumento de dados, dropout, redução do learning rate e early stopping.

**Como rodar**

```bash
cd visao_computacional
uv sync
uv run jupyter lab notebooks/
```

O CIFAR-10 é baixado automaticamente na primeira execução.

---

## Reconhecimento facial: controle de acesso

Atividade da Aula 19: um script que verifica fotos contra uma lista de pessoas autorizadas, usando OpenCV e DeepFace (modelo ArcFace). Quem for reconhecido recebe um retângulo verde com **ACESSO LIBERADO**; os demais recebem um vermelho com **ACESSO NEGADO**.

**Como rodar**

```bash
cd reconhecimento_facial
uv sync
# uma foto por pessoa autorizada em fotos_autorizadas/
# as fotos a verificar em fotos_teste/
uv run python reconhecimento.py
```

Na primeira execução, o DeepFace baixa os pesos do ArcFace (cerca de 137 MB). As imagens anotadas saem em `resultados/`. As fotos de pessoas reais não são versionadas.

---

## Observações

- Os notebooks têm saídas da última execução, então é possível ver os resultados sem rodar de novo.
- Os resultados podem variar um pouco entre máquinas, principalmente nos treinos de GPU.
