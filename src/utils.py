"""
Funzioni di utilità per I/O, versioni e formattazione.
"""
import ast
import inspect
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


def safe_synset_name(name: str) -> Optional[wn.synset]:
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