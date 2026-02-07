"""
Modulo per la ricerca onomasiologica (content-to-form) in WordNet.
Implementa funzioni per identificare synset a partire da definizioni testuali.
"""
from typing import List, Optional, Set, Tuple
from collections import Counter
import math
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import src.wordnet_helpers as wnh

# Download necessari
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)



def rank_synsets_by_similarity(
    definition_tokens: List[str],
    candidate_synsets: List[str],
    similarity_method: str = "cosine",
    stop_words: Set[str] = set(stopwords.words('english')),
    lang: str = 'eng',
    DEBUG: int = 0,
) -> List[Tuple]:
    """
    Ordina i synset candidati per similarità con la definizione.
    
    Args:
        definition_tokens: Lista di parole chiave estratte dalla definizione
        candidate_synsets: Lista di nomi di synset da valutare
        stop_words: Insieme di stopwords da escludere
        similarity_method: Metodo di similarita ("jaccard" o "cosine")
        DEBUG: Livello di debug (0=nessuno, 1=info, 2=dettagli)
    
    Returns:
        Lista di (synset, score) ordinata per similarità decrescente
    """
    if similarity_method not in {"jaccard", "cosine"}:
        raise ValueError("similarity_method deve essere 'jaccard' o 'cosine'")

    scored_synsets = []
    for synset_name in candidate_synsets:
        synset = wn.synset(synset_name)
        # Estrai token dalla definizione del synset
        synset_tokens = wnh.synset_to_tokens(synset, stop_words=stop_words, include_relations=True, rel_depth=1)
        if DEBUG >= 2:
            print(f"Token {synset_name}, tokens: {synset_tokens}")
        # Calcola similarità
        if similarity_method == "cosine":
            similarity = wnh.cosine_similarity(set(definition_tokens), set(synset_tokens))
        else:
            similarity = wnh.jaccard_similarity(set(definition_tokens), set(synset_tokens))
        scored_synsets.append((synset, similarity))
    
    # Filtra i synset con score 0.0 (non informativi)
    scored_synsets = [(s, sc) for (s, sc) in scored_synsets if sc != 0.0]
    # Ordina per score decrescente
    scored_synsets.sort(key=lambda x: x[1], reverse=True)
    
    return scored_synsets


def search_synset_from_definition(
    definition: str,
    max_results: int = 50,
    DEBUG: int = 0,
    deep_search: int = 20,
    similarity_method="jaccard",
    stop_words: Set[str] = set(stopwords.words('english')),
    extract_method: str = "new",
    lang: str = 'eng'
) -> List[Tuple]:
    """
    Data una definizione testuale, cerca i synset candidati in WordNet.
    
    L'algoritmo mantiene una lista globale di candidati (sempre i synset con maggiore
    similarità trovati finora) e itera cercando synset correlati. I nuovi synset
    vengono aggiunti ai candidati solo se hanno similarità >= miglior score precedente.
    
    Flusso:
    1. Estrai keyword dalla definizione
    2. Cerca synset per tutte le keyword, ranking, mantieni top max_results come candidati
    3. Itera:
       - Prendi synset correlati ai top max_results candidati
       - Classifica i nuovi synset
       - Aggiungi ai candidati solo quelli con score >= best_score
       - Re-rankigha tutti i candidati, mantieni top max_results
    4. Ferma quando: il best_score non migliora per deep_search iterazioni consecutive
    
    Args:
        definition: Testo della definizione
        max_results: Numero massimo di risultati da mantenere per iterazione
        DEBUG: Livello di debug (0=nessuno, 1=info iterazioni)
        deep_search: Numero di iterazioni massime prima di fermarsi (default: 20)
    
    Returns:
        Lista di tuple (synset, score) ordinata per rilevanza decrescente
    """
    if lang == 'ita':
        keywords = wnh.translate_tokens(
            sentence=definition,
            lang='ita',
            DEBUG=DEBUG
        )
    elif extract_method == "old":
        keywords = wnh.old_extract_keywords(definition, stop_words=stop_words, lang=lang)
    else:
        keywords = [t.lemma_ for t in wnh.extract_keywords(definition, lang=lang)]

    if DEBUG >= 1:
        print(f"Extracted keywords: {keywords}")
    
    # ===== GIRO 1: Cerca synset per tutte le keyword =====
    candidates: List = []
    seen = set()
    
    for keyword in keywords:
        # Cerca synset per questa keyword
        synsets = wn.synsets(keyword, lang="eng")
        
        for synset in synsets:
            if synset.name() not in seen:
                candidates.append(synset)
                seen.add(synset.name())
    
    if not candidates:
        return []


    if DEBUG >= 1:
        print(f"Iteration 0 (initial), candidates={len(candidates)}")

    iteration = 1
    v_related=set(candidates)
    # ===== ITERAZIONI: Cerca synset correlati ai top candidati =====
    while v_related and iteration <= deep_search:
        iteration += 1
        # Raccogli synset correlati
        p_related = []
        for synset in v_related:
            p_related.extend(synset.hypernyms())
            p_related.extend(synset.hyponyms())
            # p_related.extend(synset.part_meronyms())
            # p_related.extend(synset.substance_meronyms())
            # p_related.extend(synset.member_meronyms())
            # p_related.extend(synset.part_holonyms())
            # p_related.extend(synset.substance_holonyms())
            # p_related.extend(synset.member_holonyms())

        # Identifica nuovi synset
        v_related = set()
        for synset in p_related:
            if synset.name() not in seen:
                v_related.add(synset)
                seen.add(synset.name())
        
        if DEBUG >= 2:
            print(f"Iteration {iteration}: New v_related: {[s.name() for s in v_related]}")

        if not v_related:
            if DEBUG >= 1:
                print(f"Iteration {iteration}: No new related synsets found, stopping.")
            break
        candidates.extend(v_related)

    ranked = rank_synsets_by_similarity(
        keywords,
        [c.name() for c in candidates], 
        similarity_method=similarity_method,
        stop_words=stop_words,
        lang="eng",
        DEBUG=DEBUG,
        )
    return sorted(ranked, key=lambda x: x[1], reverse=True)[:max_results]
