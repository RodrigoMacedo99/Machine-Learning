"""Controle de acesso por reconhecimento facial com OpenCV e DeepFace.

Cada foto de `fotos_teste/` é analisada. Os rostos reconhecidos como pessoas de
`fotos_autorizadas/` recebem um retângulo verde com "ACESSO LIBERADO"; os demais recebem um
retângulo vermelho com "ACESSO NEGADO". As imagens anotadas são salvas em `resultados/`.
"""

from pathlib import Path

import cv2
import numpy as np
import typer
from deepface import DeepFace
from loguru import logger

app = typer.Typer()

BASE_DIR = Path(__file__).resolve().parent
AUTHORIZED_DIR = BASE_DIR / "fotos_autorizadas"
TEST_DIR = BASE_DIR / "fotos_teste"
OUTPUT_DIR = BASE_DIR / "resultados"

MODEL_NAME = "ArcFace"
# RetinaFace alinha os rostos melhor que o detector padrão (OpenCV), o que muda muito a distância.
DETECTOR_BACKEND = "retinaface"
# Limiar de distância cosseno recomendado pelo DeepFace para o modelo ArcFace.
DISTANCE_THRESHOLD = 0.68

GREEN = (0, 255, 0)  # OpenCV usa BGR
RED = (0, 0, 255)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_authorized_embeddings(folder: Path) -> list[np.ndarray]:
    """Gera o embedding do rosto de cada foto autorizada."""
    embeddings = []
    for photo in list_images(folder):
        try:
            faces = DeepFace.represent(
                img_path=str(photo),
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=True,
            )
        except ValueError:
            logger.warning(f"Nenhum rosto encontrado em {photo.name}; foto ignorada")
            continue
        if len(faces) > 1:
            logger.warning(f"{photo.name} tem {len(faces)} rostos; usando o primeiro")
        embeddings.append(np.array(faces[0]["embedding"]))

    if not embeddings:
        raise FileNotFoundError(f"Nenhuma foto com rosto válido em {folder}")
    logger.info(f"{len(embeddings)} pessoa(s) autorizada(s) carregada(s)")
    return embeddings


def is_authorized(embedding: np.ndarray, authorized: list[np.ndarray]) -> bool:
    closest = min(cosine_distance(embedding, reference) for reference in authorized)
    return closest <= DISTANCE_THRESHOLD


def draw_access_box(image: np.ndarray, area: dict, granted: bool) -> None:
    x, y, w, h = area["x"], area["y"], area["w"], area["h"]
    color = GREEN if granted else RED
    text = "ACESSO LIBERADO" if granted else "ACESSO NEGADO"
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        image,
        text,
        (x, max(y - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def check_photo(photo: Path, authorized: list[np.ndarray]) -> np.ndarray | None:
    """Anota a foto com o resultado de cada rosto detectado. Devolve None se não houver rosto."""
    try:
        faces = DeepFace.represent(
            img_path=str(photo),
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError:
        logger.warning(f"Nenhum rosto encontrado em {photo.name}")
        return None

    image = cv2.imread(str(photo))
    for face in faces:
        granted = is_authorized(np.array(face["embedding"]), authorized)
        draw_access_box(image, face["facial_area"], granted)
        logger.info(f"{photo.name}: {'liberado' if granted else 'negado'}")
    return image


@app.command()
def main(
    authorized_dir: Path = AUTHORIZED_DIR,
    test_dir: Path = TEST_DIR,
    output_dir: Path = OUTPUT_DIR,
):
    """Verifica as fotos de teste contra as pessoas autorizadas e salva as imagens anotadas."""
    authorized = load_authorized_embeddings(authorized_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for photo in list_images(test_dir):
        annotated = check_photo(photo, authorized)
        if annotated is None:
            continue
        output_path = output_dir / f"resultado_{photo.name}"
        cv2.imwrite(str(output_path), annotated)

    logger.success(f"Resultados salvos em {output_dir}")


if __name__ == "__main__":
    app()
