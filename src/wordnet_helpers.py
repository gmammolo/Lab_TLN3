"""
Helper WordNet semplificato per WSD con Lesk e embeddings.
"""
import math
from typing import Any, Counter, Dict, List, Optional, Set, Tuple
from functools import lru_cache
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
import src.utils as utils

lemmatizer = WordNetLemmatizer()

GLOSS_EXPANSIONS = {
    "amplify": ("make", "larger"),
    "enlarge": ("make", "larger"),
    "magnify": ("make", "larger"),
    "optical": ("look",),
    "perception": ("look",),
    "see": ("look",),
    "visual": ("look",),
    "instrument": ("device", "use"),
    "device": ("use",),
    "science": ("scientific", "scientifically", "study", "studied", "examine", "examined"),
    "scientific": ("scientifically", "study", "studied", "examine", "examined"),
    "research": ("study", "studied", "examine", "examined"),
}

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




def shortest_distance(syn1, syn2) -> Optional[int]:
    """
    Calcola la distanza minima (numero di passi) tra due synset in WordNet.
    Utilizza la funzione built-in di WordNet.

    Args:
        syn1: Primo synset
        syn2: Secondo synset

    Returns:
        Distanza minima come intero, None se non esiste un percorso
    """
    try:
        distance = syn1.shortest_path_distance(syn2)
        return distance
    except:
        return None


def shortest_path(syn1, syn2) -> Optional[List]:
    """
    Trova il percorso minimo tra due synset in WordNet.
    Utilizza BFS per esplorare tutte le relazioni semantiche.

    Args:
        syn1: Primo synset di partenza
        syn2: Secondo synset di destinazione

    Returns:
        Lista di synset rappresentanti il percorso minimo, None se non esiste un percorso
    """
    if syn1 == syn2:
        return [syn1]

    # BFS per trovare il percorso più breve
    from collections import deque

    queue = deque([(syn1, [syn1])])
    visited = {syn1}

    while queue:
        current, path = queue.popleft()

        # Esplora tutte le relazioni del synset corrente
        neighbors = set()
        neighbors.update(current.hypernyms())
        neighbors.update(current.hyponyms())
        # neighbors.update(current.member_holonyms())
        # neighbors.update(current.substance_holonyms())
        # neighbors.update(current.part_holonyms())
        # neighbors.update(current.member_meronyms())
        # neighbors.update(current.substance_meronyms())
        # neighbors.update(current.part_meronyms())
        # neighbors.update(current.similar_tos())

        for neighbor in neighbors:
            if neighbor == syn2:
                return path + [neighbor]

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return None




def normalize_text(text, stop_words):
    text = text.lower()
    text = re.sub(r'[_\-]+', ' ', text)
    tokens = [t for t in word_tokenize(text) if t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return tokens

def synset_to_tokens(syn, stop_words=set(), include_relations=True, rel_depth=1, include_rel_gloss=True):
    """
    Costruisce e normalizza una rappresentazione testuale espansa di un `synset`.

    La funzione concatena lemmi, gloss, esempi, domini e termini provenienti da
    relazioni (iperonimi, meronimi, similar_tos, ecc.) e applica espansioni
    lessicali definite in `GLOSS_EXPANSIONS`. Il testo risultante viene
    tokenizzato, lemmatizzato e filtrato rimuovendo le stopword fornite.

    Argomenti:
        syn: oggetto `wordnet.synset` da convertire.
        stop_words: insieme di stopword (stringhe) da rimuovere durante la
            normalizzazione.
        include_relations: se True include lemmi e gloss delle relazioni.
        rel_depth: profondità (int) per risalire gli iperonimi.
        include_rel_gloss: se True aggiunge anche i gloss dei synset correlati.

    Ritorna:
        Lista di token normalizzati (lemmi) pronta per confronto.

    Note:
        - `GLOSS_EXPANSIONS` fornisce termini aggiuntivi estratti dal gloss per
          aumentare la copertura lessicale.
        - Usare `include_rel_gloss=False` per ridurre la lunghezza del testo
          quando si confrontano molti synset.
    """
    def _synset_to_text(syn, include_relations=True, rel_depth=1, include_rel_gloss=True,) -> str:
        parts = []
        extra_terms = []
        extra_terms_set = set()
        def _add_part(text):
            if text:
                parts.append(text)
                _collect_expansions(text)

        def add_extra(term):
            if term and term not in extra_terms_set:
                extra_terms_set.add(term)
                extra_terms.append(term)

        def _collect_expansions(text):
            try:
                tokens = word_tokenize(text.lower())
            except Exception:
                return
            for tok in tokens:
                if not tok.isalpha():
                    continue
                lemma = lemmatizer.lemmatize(tok)
                expansions = GLOSS_EXPANSIONS.get(lemma)
                if expansions:
                    for e in expansions:
                        add_extra(e)
        _add_part(' '.join(syn.lemma_names()))
        _add_part(syn.definition() or '')
        if syn.examples():
            _add_part(' '.join(syn.examples()))

        if include_relations:
            related = set()

            def _collect_hypernyms(s, depth):
                if depth <= 0:
                    return
                for h in s.hypernyms():
                    related.add(h)
                    _collect_hypernyms(h, depth - 1)

            if rel_depth > 0:
                _collect_hypernyms(syn, rel_depth)

            related.update(syn.part_meronyms())
            related.update(syn.substance_meronyms())
            related.update(syn.member_meronyms())

            if syn.pos() in {"a", "s"}:
                related.update(syn.similar_tos())

            if related:
                _add_part(' '.join([l for r in related for l in r.lemma_names()]))
                if include_rel_gloss:
                    _add_part(' '.join([r.definition() for r in related if r.definition()]))

            domains = set(syn.topic_domains()) | set(syn.usage_domains())
            if domains:
                _add_part(' '.join([l for d in domains for l in d.lemma_names()]))
                if include_rel_gloss:
                    _add_part(' '.join([d.definition() for d in domains if d.definition()]))

        if extra_terms:
            parts.append(' '.join(extra_terms))

        return ' '.join(parts)
    
    txt = _synset_to_text(
        syn,
        include_relations=include_relations,
        rel_depth=rel_depth,
        include_rel_gloss=include_rel_gloss,
    )
    return normalize_text(txt, stop_words)


def old_extract_keywords(definition: str, stop_words: Set[str] = set(), lang: str = 'eng') -> List[str]:
    """
    Estrae parole chiave significative da una definizione.
    
    Args:
        definition: Testo della definizione
        stop_words: Set di stopwords da escludere (default: stopwords inglesi)
        lang: Lingua per WordNet (default: 'eng')
    
    Returns:
        Lista di parole chiave estratte e normalizzate
    """
    lemmatizer = WordNetLemmatizer()
    language = "italian" if lang == 'ita' else "english"

    def _has_italian_lemma(token: str) -> bool:
        try:
            return bool(wn.synsets(token, lang='ita')) or bool(wn.lemmas(token, lang='ita'))
        except Exception:
            return False

    def _normalize_italian_token(token: str) -> str:
        tok = token.lower()
        candidates = [tok]

        # Verbi: prova forme base (infinitivo)
        if tok.endswith("isce") and len(tok) > 5:
            candidates.append(tok[:-4] + "ire")
        if tok.endswith("ano") and len(tok) > 4:
            candidates.append(tok[:-3] + "are")
        if tok.endswith("ono") and len(tok) > 4:
            candidates.append(tok[:-3] + "ere")
            candidates.append(tok[:-3] + "ire")
        if tok.endswith("iamo") and len(tok) > 5:
            candidates.append(tok[:-4] + "are")
            candidates.append(tok[:-4] + "ere")
            candidates.append(tok[:-4] + "ire")
        if tok.endswith("a") or tok.endswith("e") or tok.endswith("i") or tok.endswith("o"):
            candidates.append(tok + "re")

        # Plurali: prova singolare
        if tok.endswith("i") and len(tok) > 3:
            candidates.append(tok[:-1] + "o")
            candidates.append(tok[:-1] + "e")
        if tok.endswith("e") and len(tok) > 3:
            candidates.append(tok[:-1] + "a")

        # femminile maschile
        if tok.endswith("a") and len(tok) > 3:
            candidates.append(tok[:-1] + "o")

        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)
            if _has_italian_lemma(cand):
                return cand

        if tok.endswith("i") and len(tok) > 3:
            return tok[:-1] + "e"
        if tok.endswith("e") and len(tok) > 3:
            return tok[:-1] + "a"
        return tok

    # Tokenizzazione e lowercasing
    tokens = word_tokenize(definition.lower(), language=language)

    # Rimozione punteggiatura e stopwords
    keywords: List[str] = []
    for token in tokens:
        if not token.isalnum() or token in stop_words or len(token) <= 2:
            continue
        if lang == 'ita':
            keywords.append(_normalize_italian_token(token))
        else:
            keywords.append(lemmatizer.lemmatize(token))

    return keywords





def extract_keywords(sentence: str, lang: str = 'eng') -> List[spacy.tokens.Token]:
    if lang == 'ita':
        nlp = spacy.load(utils.get_spacy_ita_modelname())
    else:
        nlp = spacy.load(utils.get_spacy_en_modelname())
    doc = nlp(sentence)
    filtred = [tok for tok in doc if tok.is_alpha and not tok.is_stop]
    return filtred



def translate_tokens(
    sentence: Optional[str] = None,
    lang: str = 'ita',
    DEBUG: int = 0
) -> List[str]:
    """
    Converte una lista di token in una lista di lemmi inglesi, scegliendo i synset
    piu coerenti con il contesto della frase.

    La funzione usa WordNet via NLTK e supporta il bilingua inglese/italiano
    (lang='eng' o 'ita'). Se viene fornita
    una frase o un contesto di token, seleziona per ogni token il synset con
    maggiore sovrapposizione lessicale con il contesto, e restituisce i lemmi
    inglesi del synset migliore.

    Nota:
                - I token in input sono gia normalizzati e filtrati con stopword italiane.
                - Se un token non ha synset, viene mantenuto cosi com'e.
    """    
    tokens = extract_keywords(sentence, lang)
    if DEBUG >= 2:
        print("Extracted tokens for translation:")
        for tok in tokens:
            print(tok.text, tok.lemma_ ,tok.pos_, tok.tag_)
    out: List[str] = []
    stop_words_eng = set(stopwords.words('english'))
    for tok in tokens:
        synsets =  wn.synsets(tok.lemma_, pos=penn_to_wn_pos(tok.pos_), lang=lang)
        
        # workaround missing token
        if len(synsets) == 0:
            synsets = wn.synsets(patch_tokens(tok.lemma_), pos=penn_to_wn_pos(tok.pos_), lang="eng")
        
        if len(synsets) > 0:
            l= get_best_synset(
                tok.lemma_,
                tokens,
                synsets,
            )
            if l not in stop_words_eng:
                out.append(l)
            else:
                continue
        else:
             out.append(tok.lemma_)
    
    if DEBUG >= 2:
        print(f"translate_tokens - Input Tokens: {tokens}")
        print(f"translate_tokens - Output: {out}")
    return out


def patch_tokens(token: str) -> str:
    """
    Applica un patch manuale a token specifici per migliorare la ricerca in WordNet.
    Questo è un workaround per token che non vengono riconosciuti correttamente.
    
    Args:
        token: Token da patchare
    Returns:
        token con patch applicata
    """ 
    dict = {}
    dict['ingrandito'] = 'enlarged'
    dict['invisibile'] = 'invisible'
    dict['rivelira'] = 'reveal'
    
    return dict.get(token, token)


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calcola la similarità di Jaccard tra due insiemi.
    
    Args:
        set1: Primo insieme di termini
        set2: Secondo insieme di termini
    
    Returns:
        Coefficiente di Jaccard (0-1)
    """
    set1 = set(set1)
    set2 = set(set2)

    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def cosine_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """
    Calcola la similarita del coseno tra due liste di token.

    Args:
        tokens1: Prima lista di token
        tokens2: Seconda lista di token

    Returns:
        Similarita del coseno (0-1)
    """
    if not tokens1 or not tokens2:
        return 0.0

    vector1 = Counter(tokens1)
    vector2 = Counter(tokens2)

    common_terms = set(vector1) & set(vector2)
    dot_product = sum(vector1[term] * vector2[term] for term in common_terms)

    norm1 = math.sqrt(sum(value * value for value in vector1.values()))
    norm2 = math.sqrt(sum(value * value for value in vector2.values()))

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def get_best_synset(
    token: str,
    context_tokens: List[spacy.tokens.Token],
    candidate_synsets: List[wn.synset],
    lang: str = "eng",
) -> Optional[str]:
    """
    Seleziona il synset migliore per `token` usando l'overlap con il contesto.

    Restituisce un lemma inglese del synset scelto (o il token originale
    se non e' possibile determinare un lemma in inglese).
    """
    def _split_lemma(lemma: str) -> List[str]:
        parts = re.split(r"[_\-]+", lemma.lower())
        return [p for p in parts if p]

    if not candidate_synsets:
        return token

    context_lang = {
        (t.lemma_ or t.text).lower()
        for t in context_tokens
        if t.is_alpha
    }
    context_lang.discard(token.lower())

    if lang == "eng":
        context_eng = set(context_lang)
    else:
        context_eng = set()
        for ctx in context_lang:
            for s in wn.synsets(ctx, lang=lang):
                for lemma in s.lemma_names("eng"):
                    for part in _split_lemma(lemma):
                        context_eng.add(part)
    best_syn = None
    best_score = -1.0
    best_freq = -1

    stop_words_eng = set(stopwords.words("english"))
    for syn in candidate_synsets:
        signature_eng = set(
            synset_to_tokens(
                syn,
                stop_words_eng,
                include_relations=True,
                rel_depth=1,
                include_rel_gloss=False,
            )
        )
        signature_lang = set()
        if lang != "eng":
            for lemma in syn.lemma_names(lang):
                for part in _split_lemma(lemma):
                    signature_lang.add(part)

        score_eng = jaccard_similarity(signature_eng, context_eng)
        score_lang = jaccard_similarity(signature_lang, context_lang)
        if lang == "eng":
            score = score_eng
        else:
            score = (0.7 * score_eng) + (0.3 * score_lang)

        freq = sum(l.count() for l in syn.lemmas())
        if score > best_score or (score == best_score and freq > best_freq):
            best_syn = syn
            best_score = score
            best_freq = freq

    if best_syn is None:
        return token

    lemma_names_eng = best_syn.lemma_names("eng")
    if lemma_names_eng:
        return lemma_names_eng[0]
    lemma_names_lang = best_syn.lemma_names(lang)
    return lemma_names_lang[0] if lemma_names_lang else token
