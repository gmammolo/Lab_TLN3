"""
Helper WordNet semplificato per WSD con Lesk e embeddings.
"""
from typing import Any, Dict, List, Optional, Tuple
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.wsd import lesk


def penn_to_wn_pos(penn_tag: str) -> Optional[str]:
    """Converte tag Penn Treebank a tag WordNet (n, v, a, r)."""
    if penn_tag.startswith("NN"):
        return "n"
    elif penn_tag.startswith("VB"):
        return "v"
    elif penn_tag.startswith("JJ"):
        return "a"
    elif penn_tag.startswith("RB"):
        return "r"
    return None


def detect_pos_in_sentence(word: str, sentence: str) -> Optional[str]:
    """Estrae il POS WordNet della parola nella frase."""
    try:
        tokens = word_tokenize(sentence)
        tags = pos_tag(tokens)
        for tok, tag in tags:
            if tok.lower() == word.lower():
                return penn_to_wn_pos(tag)
    except:
        pass
    return None


def get_synset_structure(synset_name: str) -> Dict[str, Any]:
    """Estrae gloss, lemmi, esempi, iperonimi di un synset."""
    try:
        ss = wn.synset(synset_name)
        return {
            "gloss": ss.definition(),
            "lemmas": [l.name() for l in ss.lemmas()],
            "examples": ss.examples(),
            "hypernyms": [s.name() for s in ss.hypernyms()],
            "hyponyms": [s.name() for s in ss.hyponyms()],
        }
    except:
        return {}

