"""Prepara fotos de exemplo para testar o reconhecimento facial, usando o LFW.

O LFW (Labeled Faces in the Wild) é um dataset público de rostos de figuras públicas. Por padrão:

- a primeira foto de PESSOA_AUTORIZADA vai para `fotos_autorizadas/` (pessoa liberada);
- a segunda foto de PESSOA_AUTORIZADA vai para `fotos_teste/` (deve dar ACESSO LIBERADO);
- a primeira foto de PESSOA_NEGADA vai para `fotos_teste/` (deve dar ACESSO NEGADO).

O download fica em `data/lfw/`, que não é versionado. As fotos são salvas com prefixos
`autorizado_` e `teste_`, e outros arquivos das pastas não são alterados.
"""

from pathlib import Path

import numpy as np
import typer
from loguru import logger
from PIL import Image
from sklearn.datasets import fetch_lfw_people

app = typer.Typer()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "lfw"
AUTHORIZED_DIR = BASE_DIR / "fotos_autorizadas"
TEST_DIR = BASE_DIR / "fotos_teste"


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def photos_of(dataset, name: str) -> list[np.ndarray]:
    if name not in dataset.target_names:
        raise typer.BadParameter(
            f"'{name}' não está no LFW com pelo menos o mínimo de fotos"
        )
    index = list(dataset.target_names).index(name)
    return [dataset.images[i] for i in np.where(dataset.target == index)[0]]


def save_photo(image: np.ndarray, path: Path) -> None:
    array = np.asarray(image)
    if array.max() <= 1.0:
        array = array * 255
    Image.fromarray(array.astype(np.uint8)).save(path)
    logger.info(f"salvo: {path.relative_to(BASE_DIR)}")


@app.command()
def main(
    authorized_name: str = typer.Option(
        "George W Bush", help="Pessoa que deve ser liberada."
    ),
    denied_name: str = typer.Option("Colin Powell", help="Pessoa que deve ser negada."),
    min_faces: int = typer.Option(15, help="Mínimo de fotos por pessoa no dataset."),
):
    """Baixa o LFW e monta as pastas de autorizados e de teste."""
    dataset = fetch_lfw_people(
        data_home=str(DATA_DIR),
        min_faces_per_person=min_faces,
        color=True,
        resize=1.0,
        slice_=None,
    )

    authorized = photos_of(dataset, authorized_name)
    denied = photos_of(dataset, denied_name)
    if len(authorized) < 2:
        raise typer.BadParameter(f"'{authorized_name}' precisa de pelo menos 2 fotos")

    AUTHORIZED_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    authorized_slug = slugify(authorized_name)
    denied_slug = slugify(denied_name)
    save_photo(authorized[0], AUTHORIZED_DIR / f"autorizado_{authorized_slug}.jpg")
    save_photo(authorized[1], TEST_DIR / f"teste_liberado_{authorized_slug}.jpg")
    save_photo(denied[0], TEST_DIR / f"teste_negado_{denied_slug}.jpg")

    logger.success("Fotos preparadas. Rode: uv run python reconhecimento.py")


if __name__ == "__main__":
    app()
