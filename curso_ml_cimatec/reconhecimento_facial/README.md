# Reconhecimento facial

Controle de acesso por reconhecimento facial com OpenCV e DeepFace (Aula 19 do curso de Machine Learning, CIMATEC).

Cada rosto detectado em uma foto de teste é comparado com as pessoas autorizadas. Quem for reconhecido recebe um retângulo verde com **ACESSO LIBERADO**; os demais recebem um retângulo vermelho com **ACESSO NEGADO**.

## Estrutura

```
reconhecimento_facial/
├── fotos_autorizadas/   <- uma foto de cada pessoa autorizada (não versionada)
├── fotos_teste/         <- fotos a serem verificadas (não versionadas)
├── resultados/          <- imagens anotadas geradas pelo script (não versionadas)
├── reconhecimento.py    <- script de verificação
├── pyproject.toml
└── README.md
```

As fotos de pessoas reais ficam fora do git pelo `.gitignore`. Só as pastas são versionadas.

## Como executar

```bash
uv sync
uv run python reconhecimento.py
```

Opções: `--authorized-dir`, `--test-dir` e `--output-dir`.

## Como funciona

- O modelo `ArcFace` gera um embedding (vetor) de cada rosto.
- Um rosto é liberado se a distância cosseno para alguma foto autorizada for menor ou igual a 0,68 (limiar recomendado pelo DeepFace para o ArcFace).
- Os pesos do modelo são baixados pelo DeepFace na primeira execução.

## Observações

- A detecção usa o OpenCV. Por isso o `opencv-python` está fixado em `<5`: a versão 5 não traz o `CascadeClassifier` usado pelo detector.
- Rostos muito pequenos na foto podem não ser detectados. Nesse caso a foto é ignorada e um aviso é registrado.
