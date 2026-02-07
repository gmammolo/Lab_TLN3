"""
Funzioni di utilità per il Lab2: preprocessing e calcolo similarità.
"""
import string
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import List
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_definition(text: str, nlp=None) -> List[str]:
    """
    Normalizza il testo di una definizione.
    
    Operazioni:
    - Lowercasing
    - Rimozione punteggiatura
    - Tokenizzazione
    - Stopword removal (italiano)
    - Lemmatizzazione
    
    Args:
        text (str): Testo della definizione
        nlp: Modello spacy (opzionale, se None usa word_tokenize)
    
    Returns:
        list: Lista di token lemmatizzati e filtrati
    """
    # Lowercasing
    text = text.lower()
    
    # Rimozione punteggiatura
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Stopwords italiano
    stop_words = set(stopwords.words('italian'))
    
    if nlp is not None:
        # Lemmatizzazione con spacy
        doc = nlp(text)
        tokens = [
            token.lemma_ 
            for token in doc 
            if token.lemma_ not in stop_words 
            and len(token.lemma_) >= 3
            and not token.is_space
        ]
    else:
        # Tokenizzazione base (fallback)
        tokens = word_tokenize(text, language='italian')
        tokens = [
            token 
            for token in tokens 
            if token not in stop_words 
            and len(token) >= 3
        ]
    
    return tokens


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calcola la cosine similarity tra due vettori."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def compute_lexical_similarity(tokens1: List[str], tokens2: List[str]) -> float:
    """
    Calcola la similarità lessicale tra due sequenze di token usando cosine similarity su frequenze termini.
    
    Args:
        tokens1 (list): Token della prima definizione
        tokens2 (list): Token della seconda definizione
    
    Returns:
        float: Valore di similarità tra 0 e 1
    """
    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0.0
    
    # Frequenze dei termini
    freq1 = Counter(tokens1)
    freq2 = Counter(tokens2)
    
    # Tutti i termini
    all_terms = set(freq1.keys()) | set(freq2.keys())
    
    # Vettori di frequenza
    vec1 = np.array([freq1.get(term, 0) for term in all_terms])
    vec2 = np.array([freq2.get(term, 0) for term in all_terms])
    
    # Cosine similarity
    return cosine_similarity(vec1, vec2)


def compute_semantic_similarity(
    tokens1: List[str], 
    tokens2: List[str], 
    embeddings_model
) -> float:
    """
    Calcola la similarità semantica tra due sequenze di token.
    
    Metodo: Media degli embedding dei token, poi cosine similarity
    
    Args:
        tokens1 (list): Token della prima definizione
        tokens2 (list): Token della seconda definizione
        embeddings_model: Modello di embedding (Word2Vec, FastText, etc.)
    
    Returns:
        float: Cosine similarity tra i vettori medi (0-1 dopo normalizzazione)
    """
    def get_sentence_vector(tokens, model):
        """Calcola la media degli embedding dei token."""
        vectors = []
        for token in tokens:
            try:
                if hasattr(model, 'get_vector'):
                    # FastText
                    vec = model.get_vector(token)
                elif hasattr(model, 'wv'):
                    # Gensim Word2Vec
                    vec = model.wv[token]
                else:
                    # Dizionario diretto
                    vec = model[token]
                vectors.append(vec)
            except (KeyError, ValueError):
                # Token non presente nel vocabolario
                continue
        
        if len(vectors) == 0:
            return None
        return np.mean(vectors, axis=0)
    
    vec1 = get_sentence_vector(tokens1, embeddings_model)
    vec2 = get_sentence_vector(tokens2, embeddings_model)
    #print (vec1, vec2)
    
    if vec1 is None or vec2 is None:
        return 0.0
    
    # Cosine similarity
    similarity = cosine_similarity(vec1, vec2)
    
    # Normalizza tra 0 e 1 (cosine può essere negativo)
    return max(0.0, min(1.0, (similarity + 1) / 2))


def build_similarity_matrix(
    definitions: List[List[str]], 
    similarity_func,
    **kwargs
) -> np.ndarray:
    """
    Costruisce una matrice di similarità tra tutte le coppie di definizioni.
    
    Args:
        definitions (list): Lista di definizioni tokenizzate
        similarity_func (callable): Funzione di calcolo similarità
        **kwargs: Argomenti aggiuntivi per similarity_func
    
    Returns:
        np.ndarray: Matrice simmetrica (n × n) di similarità
    """
    n = len(definitions)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                sim = 1.0
            else:
                sim = similarity_func(definitions[i], definitions[j], **kwargs)
            matrix[i, j] = sim
            matrix[j, i] = sim
    
    return matrix


def aggregate_similarities(
    df: pd.DataFrame,
    simlex_matrix: np.ndarray,
    simsem_matrix: np.ndarray,
    concept_col: str = 'concept',
    concreteness_col: str = 'concreteness',
    specificity_col: str = 'specificity'
) -> pd.DataFrame:
    """
    Aggrega i valori di simlex e simsem per categoria.
    
    Args:
        df (pd.DataFrame): DataFrame con metadati definizioni
        simlex_matrix (np.ndarray): Matrice similarità lessicale
        simsem_matrix (np.ndarray): Matrice similarità semantica
        concept_col (str): Nome colonna concetto
        concreteness_col (str): Nome colonna concretezza
        specificity_col (str): Nome colonna specificità
    
    Returns:
        pd.DataFrame: Risultati aggregati per categoria
    """
    results = []
    
    # Estrai le maschere di indice per ogni concetto
    for concept in df[concept_col].unique():
        mask = df[concept_col] == concept
        indices = df[mask].index.tolist()
        
        # Estrai sottomatrici
        simlex_values = []
        simsem_values = []
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx_i = indices[i]
                idx_j = indices[j]
                simlex_values.append(simlex_matrix[idx_i, idx_j])
                simsem_values.append(simsem_matrix[idx_i, idx_j])
        
        if len(simlex_values) > 0:
            # Metadati
            concreteness = df[mask][concreteness_col].iloc[0]
            specificity = df[mask][specificity_col].iloc[0]
            
            results.append({
                'concept': concept,
                'concreteness': concreteness,
                'specificity': specificity,
                'simlex_mean': np.mean(simlex_values),
                'simlex_std': np.std(simlex_values),
                'simlex_median': np.median(simlex_values),
                'simlex_min': np.min(simlex_values),
                'simlex_max': np.max(simlex_values),
                'simsem_mean': np.mean(simsem_values),
                'simsem_std': np.std(simsem_values),
                'simsem_median': np.median(simsem_values),
                'simsem_min': np.min(simsem_values),
                'simsem_max': np.max(simsem_values),
                'n_pairs': len(simlex_values)
            })
    
    return pd.DataFrame(results)


# Statistiche per concetto (media, mediana, moda)
def compute_concept_stats(matrix, df, concept_col='concept'):
    """Calcola media, mediana e moda delle similarità intra-concetto."""
    rows = []
    for concept in df[concept_col].unique():
        indices = df[df[concept_col] == concept].index.to_numpy()
        if len(indices) < 2:
            continue
        sub = matrix[np.ix_(indices, indices)]
        tri = sub[np.triu_indices_from(sub, k=1)]
        tri = tri[~np.isnan(tri)]
        if len(tri) == 0:
            continue
        # Work on rounded values to avoid floating-point fragmentation
        tri_rounded = np.round(tri, 3)

        # Prefer a non-zero mode if there are non-zero similarities;
        # fallback to full series mode if all values are zero or no non-zero mode exists.
        s = pd.Series(tri_rounded)
        nonzero = s[s != 0]
        if len(nonzero) > 0:
            mode_vals = nonzero.mode()
            # If mode of nonzero is empty for any reason, fallback to overall mode
            if len(mode_vals) == 0:
                mode_vals = s.mode()
        else:
            mode_vals = s.mode()

        mode_val = mode_vals.iloc[0] if len(mode_vals) > 0 else np.nan

        # Prefer median over non-zero values if available (reduces impact of OOV->0 similarities)
        tri_nonzero = tri[tri != 0]
        if len(tri_nonzero) > 0:
            median_val = float(np.median(tri_nonzero))
        else:
            median_val = float(np.median(tri))

        rows.append({
            'concetto': concept,
            'media': float(np.mean(tri)),
            'mediana': median_val,
            'moda': float(mode_val),
            'n_pairs': int(len(tri))
        })
    return pd.DataFrame(rows)


def lab2_plot_summary_df(summary_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Analisi Similarità Lessicale e Semantica per Concetto', fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Barplot confronto media simlex vs simsem
    ax1 = axes[0]
    x_pos = np.arange(len(summary_df))
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, summary_df['simlex_media'], width, label='Simlex', color='coral', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, summary_df['simsem_media'], width, label='Simsem', color='skyblue', alpha=0.8)
    ax1.set_xlabel('Concetto', fontweight='bold')
    ax1.set_ylabel('Similarità Media', fontweight='bold')
    ax1.set_title('Media di Similarità Lessicale vs Semantica', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(summary_df['concetto'], rotation=45, ha='right')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Plot 3: Confronto mediane (mediana è meno sensibile agli outlier)
    ax3 = axes[1]
    bars1 = ax3.bar(x_pos - width/2, summary_df['simlex_mediana'], width, label='Simlex', color='chocolate', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, summary_df['simsem_mediana'], width, label='Simsem', color='teal', alpha=0.8)
    ax3.set_xlabel('Concetto', fontweight='bold')
    ax3.set_ylabel('Similarità Mediana', fontweight='bold')
    ax3.set_title('Mediana di Similarità Lessicale vs Semantica', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(summary_df['concetto'], rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)
    
    return plt


def lab2_table_aggregate(summary_df):
    # Costruisco `agr_df` salvando in ogni cella un dizionario con word/simlex/simsem
    def _get_metrics(concept_name):
        row = summary_df[summary_df['concetto'].str.lower() == concept_name.lower()]
        if row.empty:
            return (None, None)
        return float(row['simlex_media'].values[0]), float(row['simsem_media'].values[0])

    # Recupera metriche per i concetti richiesti
    simlex_pericolo, simsem_pericolo = _get_metrics('Pericolo')
    simlex_pantalone, simsem_pantalone = _get_metrics('Pantalone')
    simlex_euristica, simsem_euristica = _get_metrics('Euristica')
    simlex_microscopio, simsem_microscopio = _get_metrics('Microscopio')


    agr_df = pd.DataFrame(index=['Generico', 'Specifico'], columns=['Astratto', 'Concreto'])

    agr_df.at['Generico','Astratto'] = {'word': 'Pericolo', 'simlex': simlex_pericolo, 'simsem': simsem_pericolo}
    agr_df.at['Generico','Concreto'] = {'word': 'Pantalone', 'simlex': simlex_pantalone, 'simsem': simsem_pantalone}
    agr_df.at['Specifico','Astratto'] = {'word': 'Euristica', 'simlex': simlex_euristica, 'simsem': simsem_euristica}
    agr_df.at['Specifico','Concreto'] = {'word': 'Microscopio', 'simlex': simlex_microscopio, 'simsem': simsem_microscopio}

    # Creo una versione per la visualizzazione usando un doppio loop (compatibile con tutte le versioni pandas)
    def _cell_to_html(d):
        if not isinstance(d, dict):
            return ''
        simlex = '' if d.get('simlex') is None else f"{d['simlex']:.3f}"
        simsem = '' if d.get('simsem') is None else f"{d['simsem']:.3f}"
        return f"{d.get('word','')}<br>simlex: {simlex}<br>simsem: {simsem}"

    display_df = pd.DataFrame(index=agr_df.index, columns=agr_df.columns)
    for r in agr_df.index:
        for c in agr_df.columns:
            display_df.at[r,c] = _cell_to_html(agr_df.at[r,c])

    return agr_df, display_df


def lab2_plot_heatmap(sns, agr_df):

    simlex_num = pd.DataFrame(index=agr_df.index, columns=agr_df.columns, dtype=float)
    simsem_num = pd.DataFrame(index=agr_df.index, columns=agr_df.columns, dtype=float)
    for r in agr_df.index:
        for c in agr_df.columns:
            cell = agr_df.at[r,c]
            if isinstance(cell, dict):
                simlex_num.at[r,c] = cell.get('simlex')
                simsem_num.at[r,c] = cell.get('simsem')
            else:
                simlex_num.at[r,c] = np.nan
                simsem_num.at[r,c] = np.nan

    # Impostazioni grafico: uso constrained_layout per evitare sovrapposizioni
    fig, axes = plt.subplots(1, 2, figsize=(12,5), constrained_layout=True)

    # Heatmap simlex
    ax = axes[0]
    im1 = sns.heatmap(simlex_num.astype(float), annot=True, fmt='.3f', cmap='YlOrRd',
                    vmin=0, vmax=1, cbar=True, linewidths=0.5, linecolor='gray', ax=ax,
                    annot_kws={"size":9})
    ax.set_title('Heatmap simlex (media)')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Heatmap simsem
    ax = axes[1]
    im2 = sns.heatmap(simsem_num.astype(float), annot=True, fmt='.3f', cmap='YlGnBu',
                    vmin=0, vmax=1, cbar=True, linewidths=0.5, linecolor='gray', ax=ax,
                    annot_kws={"size":9})
    ax.set_title('Heatmap simsem (media)')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    return fig