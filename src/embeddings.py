"""
Gestione embeddings semplificata e WSD con embeddings.
"""
from functools import lru_cache
from typing import List, Any
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache as _lru_cache
import numpy as np

@lru_cache(maxsize=1)
def get_embedding_model(model_name: str ) -> SentenceTransformer:
    """Carica il modello di embeddings (con caching)."""
    return SentenceTransformer(model_name)


def embed_disambiguate(word: str, sentence: str, pos=None, model_name: str = "all-MiniLM-L6-v2") -> List[tuple]:
    """
    Disambiguazione usando embeddings.
    
    Calcola l'embedding della frase e di ogni synset (gloss + esempi),
    seleziona il synset con massima similarità coseno.
    
    Args:
        word: parola target
        sentence: frase di contesto
        pos: POS filter (opzionale)
    
    Returns:
        list: lista di (synset_name, similarity_score) ordinata per score decrescente
    """
    model = get_embedding_model(model_name)

    try:
        candidates = wn.synsets(word, pos=pos)
        if not candidates:
             return []

        # Build extended text for each synset (definition, examples, lemmas, hypernyms, hyponyms)
        synset_texts = [_build_synset_text(c, include_examples=False, include_related=False) for c in candidates]
        synset_texts += [_build_synset_text(c, include_examples=True, include_related=False) for c in candidates]
        synset_texts += [_build_synset_text(c, include_examples=True, include_related=True) for c in candidates]

        # Encode in batch (more efficient than per-synset encode loop)
        gloss_embs = model.encode(synset_texts)
        sent_emb = model.encode(sentence)

        sims = cosine_similarity([sent_emb], gloss_embs)[0]
        # Compute similarities (vectorized)
        idx=np.argmax(sims)
        # Se idx < len(candidates), è il primo blocco (no examples, no related)
        # Se idx < 2*len(candidates), è il secondo blocco (with examples, no related)
        # Altrimenti è il terzo blocco (with examples and related)
        rnd = idx // len(candidates)
        idx_in_block = idx % len(candidates)
        return [candidates[idx_in_block].name(), float(sims[idx]), rnd]
    except Exception as e:
        import traceback
        print("embed_disambiguate exception occurred:", repr(e))
        traceback.print_exc()
        return []


@_lru_cache(maxsize=4096)
def _build_synset_text(syn: Any, include_examples: bool = True, include_related: bool = True, max_related: int = 10) -> str:
    """Build a richer textual representation for a synset.

    Includes: definition, examples, lemma names, and optionally related synsets
    (hypernyms and hyponyms up to `max_related` items).
    Cached by synset name to avoid recomputing the text.
    """
    parts = []
    # definition and examples
    defn = syn.definition()
    if defn:
        parts.append(defn)

    exs = syn.examples()
    if exs and include_examples:
        parts.append(' '.join(exs))

    # lemma names
    lemmas = [l.name().replace('_', ' ') for l in syn.lemmas()]
    if lemmas:
        parts.append(' '.join(lemmas))

    if include_related:
        # hypernyms
        try:
            for hyp in syn.hypernyms()[:max_related]:
                hdef = hyp.definition()
                if hdef:
                    parts.append(hdef)
                hexs = hyp.examples()
                if hexs:
                    parts.append(' '.join(hexs))
                hlem = [l.name().replace('_', ' ') for l in hyp.lemmas()]
                if hlem:
                    parts.append(' '.join(hlem))
        except Exception:
            pass

        # hyponyms
        try:
            for hy in syn.hyponyms()[:max_related]:
                hydef = hy.definition()
                if hydef:
                    parts.append(hydef)
                hyexs = hy.examples()
                if hyexs:
                    parts.append(' '.join(hyexs))
                hylem = [l.name().replace('_', ' ') for l in hy.lemmas()]
                if hylem:
                    parts.append(' '.join(hylem))
        except Exception:
            pass

    # join parts with space and return
    return ' '.join([p for p in parts if p])
