"""
Funzioni di utilità per I/O, versioni e formattazione.
"""
import ast
import inspect
import shutil
import sys
import re
import subprocess
from textwrap import dedent
from pathlib import Path
from typing import Any, Dict, Optional
from nltk.corpus import wordnet as wn


def ensure_dir(pathName: str) -> None:
    """
    Crea la directory se non esiste.

    Args:
        path (Path): percorso della directory.
    """
    path = Path(pathName)
    path.mkdir(parents=True, exist_ok=True)


def build_gloss_text(synset: wn.synset) -> str:
    """
    Costruisce un testo compatto per un synset (definizione, esempi, lemmi).

    Args:
        synset (Synset): synset WordNet.

    Returns:
        str: testo aggregato.
    """
    parts = [synset.definition()]
    parts += synset.examples()
    parts += [l.name().replace("_", " ") for l in synset.lemmas()[:3]]
    return " ".join([p for p in parts if p])


def safe_synset_name(name: str):
    """
    Recupera un synset in modo sicuro, restituendo None se non esiste.

    Args:
        name (str): nome del synset (es. 'dog.n.01').

    Returns:
        Synset or None: synset se trovato.
    """
    try:
        return wn.synset(name)
    except Exception:
        return None


def print_code(fn):
    """
    Stampa il codice sorgente di una funzione/Classe senza il docstring iniziale.

    Args:
        fn (str|callable): nome della funzione/Classe (es. "SimplifiedLesk") oppure
            l'oggetto funzione/classe stesso.
    """
    # Se riceviamo una stringa, proviamo a risolverla nel modulo `src.wsd`.
    target = fn
    if isinstance(fn, str):
        import importlib

        try:
            mod = importlib.import_module("src.wsd")
            target = getattr(mod, fn)
        except Exception:
            raise TypeError(
                "Nome fornito non risolve a un oggetto in 'src.wsd'. Passa l'oggetto o importa prima il simbolo."
            )

    # Ottieni il sorgente e rimuovi il docstring iniziale per leggibilità
    src = inspect.getsource(target)
    try:
        mod = ast.parse(dedent(src))
        node = mod.body[0]
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
        cleaned = ast.unparse(mod)
    except Exception:
        cleaned = src
    print(cleaned)
    
    

def detect_cuda_version() -> Optional[str]:
    """Try to detect CUDA version (returns e.g. '13.0' or None)."""
    # Try nvidia-smi
    nvs = shutil.which("nvidia-smi")
    if nvs:
        try:
            out = subprocess.check_output([nvs, "--query-gpu=driver_version,compute_cap", "--format=csv,noheader"], stderr=subprocess.STDOUT, text=True)
        except Exception:
            try:
                out = subprocess.check_output([nvs], stderr=subprocess.STDOUT, text=True)
            except Exception:
                out = ""
        # Look for common patterns
        m = re.search(r"CUDA\s*Version\s*:?\s*(\d+\.\d+)", out)
        if m:
            return m.group(1)
    # Try nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        try:
            out = subprocess.check_output([nvcc, "--version"], stderr=subprocess.STDOUT, text=True)
            m = re.search(r"release\s*(\d+\.\d+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass
    # As a last resort, check environment variable
    cuda_home = (sys.environ.get("CUDA_HOME") or sys.environ.get("CUDA_PATH"))
    if cuda_home:
        # Try to glean version from path
        m = re.search(r"(\d+\.\d+)", cuda_home)
        if m:
            return m.group(1)
    return None



def get_spacy_ita_modelname() -> str:
    """
    Restituisce il nome del modello spacy italiano da utilizzare, in base alla versione di spacy installata.

    Returns:
        str: nome del modello spacy italiano (es. "it_core_news_sm").
    """
    cuda_version = detect_cuda_version()
    if cuda_version and cuda_version.startswith("13"):
        return "it_core_news_lg"
    else:
        return "it_core_news_sm"
        
def get_spacy_en_modelname() -> str:
    """
    Restituisce il nome del modello spacy inglese da utilizzare, in base alla versione di spacy installata.

    Returns:
        str: nome del modello spacy inglese (es. "en_core_web_sm").
    """
    cuda_version = detect_cuda_version()
    if cuda_version and cuda_version.startswith("13"):
        return "en_core_web_trf"
    else:
        return "en_core_web_sm"
