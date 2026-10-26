# Reconhecimento facial

Controle de acesso por reconhecimento facial com OpenCV e DeepFace (Aula 19 do curso de Machine Learning, CIMATEC).

Cada rosto detectado em uma foto de teste é comparado com as pessoas autorizadas. Quem for reconhecido recebe um retângulo verde com **ACESSO LIBERADO**; os demais recebem um retângulo vermelho com **ACESSO NEGADO**.

## Estrutura

```
reconhecimento_facial/
├── fotos_autorizadas/   <- uma foto de cada pessoa autorizada (não versionada)
├── fotos_teste/         <- fotos a serem verificadas (não versionadas)
├── resultados/          <- imagens anotadas geradas pelo script (não versionadas)
├── preparar_fotos.py    <- baixa exemplos do LFW e monta as pastas acima
├── reconhecimento.py    <- script de verificação
├── pyproject.toml
└── README.md
```

As fotos de pessoas reais ficam fora do git pelo `.gitignore`. Só as pastas são versionadas.

## Como executar

Com as suas próprias fotos, coloque uma foto de cada pessoa autorizada em `fotos_autorizadas/` e as fotos a verificar em `fotos_teste/`, depois:

```bash
uv sync
uv run python reconhecimento.py
```

Sem fotos próprias, use exemplos públicos do LFW (Labeled Faces in the Wild):

```bash
uv sync
uv run python preparar_fotos.py    # baixa o LFW para data/lfw/ e monta as pastas
uv run python reconhecimento.py
```

Por padrão, `preparar_fotos.py` coloca uma foto de George W Bush em `fotos_autorizadas/` e duas em `fotos_teste/`: uma da mesma pessoa (deve dar LIBERADO) e uma de Colin Powell (deve dar NEGADO). Os nomes podem ser trocados com `--authorized-name` e `--denied-name`.

Opções do `reconhecimento.py`: `--authorized-dir`, `--test-dir` e `--output-dir`.

## Como funciona

- O RetinaFace detecta e alinha os rostos.
- O modelo `ArcFace` gera um embedding (vetor) de cada rosto.
- Um rosto é liberado se a distância cosseno para alguma foto autorizada for menor ou igual a 0,68 (limiar recomendado pelo DeepFace para o ArcFace).
- Os pesos dos modelos são baixados pelo DeepFace na primeira execução.

## Observações

- O detector padrão do OpenCV alinhava mal os rostos do LFW e dava distâncias altas para a mesma pessoa. Por isso o RetinaFace é usado no lugar dele.
- Rostos muito pequenos na foto podem não ser detectados. Nesse caso a foto é ignorada e um aviso é registrado.
