"""Habilita a GPU no TensorFlow.

O pacote `tensorflow[and-cuda]` instala as bibliotecas CUDA dentro de `nvidia/`, mas o carregador
do TensorFlow nem sempre as encontra. Carregar essas bibliotecas antes de importar o TensorFlow
resolve o problema, sem precisar de variável de ambiente.
"""

import ctypes
import glob
import importlib.util
import os


def preload_cuda_libraries() -> None:
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    pkg = os.path.dirname(spec.submodule_search_locations[0])
    for lib in sorted(glob.glob(os.path.join(pkg, "nvidia", "*", "lib", "*.so*"))):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
