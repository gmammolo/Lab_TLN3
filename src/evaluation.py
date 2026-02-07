"""
Modulo per la valutazione dei sistemi di ricerca content-to-form.
Implementa metriche standard per valutare la qualità dell'identificazione di synset.
"""
from typing import Dict, List, Optional
import numpy as np


def calculate_accuracy_at_k(predictions: List[List], ground_truth: List, k: int = 1) -> float:
    """
    Calcola l'accuracy@k: percentuale di volte in cui il synset corretto
    appare nei primi k risultati.
    
    Args:
        predictions: Lista di liste di synset predetti (ordinati per rilevanza)
        ground_truth: Lista di synset corretti
        k: Numero di top risultati da considerare
    
    Returns:
        Accuracy@k (0-1)
    """
    if not predictions or not ground_truth:
        return 0.0
    
    correct = 0
    total = 0
    
    for pred_list, gold in zip(predictions, ground_truth):
        if gold is None or (isinstance(gold, str) and gold.strip() == ""):
            continue
        
        total += 1
        
        # Considera solo i primi k risultati
        top_k = pred_list[:k] if len(pred_list) >= k else pred_list
        
        # Verifica se il gold è nei top k
        # Gestisci sia il caso di synset objects che di stringhe
        gold_names = []
        if hasattr(gold, 'name'):
            gold_names = [gold.name()]
        elif isinstance(gold, str):
            gold_names = [gold]
        
        for pred in top_k:
            pred_name = pred.name() if hasattr(pred, 'name') else str(pred)
            if pred_name in gold_names:
                correct += 1
                break
    
    return correct / total if total > 0 else 0.0


def calculate_mrr(predictions: List[List], ground_truth: List) -> float:
    """
    Calcola il Mean Reciprocal Rank (MRR) per le predizioni.
    
    MRR = (1/N) * Σ(1/rank_i) dove rank_i è la posizione del primo
    risultato corretto per la query i-esima.
    
    Args:
        predictions: Lista di liste di synset ordinati per rilevanza
        ground_truth: Lista di synset corretti
    
    Returns:
        MRR score (0-1)
    """
    if not predictions or not ground_truth:
        return 0.0
    
    reciprocal_ranks = []
    
    for pred_list, gold in zip(predictions, ground_truth):
        if gold is None or (isinstance(gold, str) and gold.strip() == ""):
            continue
        
        # Gestisci sia synset objects che stringhe
        gold_names = []
        if hasattr(gold, 'name'):
            gold_names = [gold.name()]
        elif isinstance(gold, str):
            gold_names = [gold]
        
        # Trova la posizione del primo match
        for rank, pred in enumerate(pred_list, start=1):
            pred_name = pred.name() if hasattr(pred, 'name') else str(pred)
            if pred_name in gold_names:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # Nessun match trovato
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def find_rank_of_correct(predictions: List, ground_truth) -> Optional[int]:
    """
    Trova il rank (posizione 1-indexed) del synset corretto nella lista di predizioni.
    
    Args:
        predictions: Lista di synset predetti ordinati per rilevanza
        ground_truth: Synset corretto
    
    Returns:
        Rank del synset corretto (1-indexed) o None se non trovato
    """
    if not predictions or ground_truth is None:
        return None
    
    # Gestisci sia synset objects che stringhe
    gold_names = []
    if hasattr(ground_truth, 'name'):
        gold_names = [ground_truth.name()]
    elif isinstance(ground_truth, str):
        gold_names = [ground_truth]
    else:
        return None
    
    for rank, pred in enumerate(predictions, start=1):
        pred_name = pred.name() if hasattr(pred, 'name') else str(pred)
        if pred_name in gold_names:
            return rank
    
    return None


def evaluate_content_to_form(predictions: List[List], ground_truth: List) -> Dict[str, float]:
    """
    Valuta l'accuratezza della ricerca content-to-form con metriche multiple.
    
    Args:
        predictions: Lista di liste di synset predetti
        ground_truth: Lista di synset corretti di riferimento
    
    Returns:
        Dizionario con metriche: accuracy@1, accuracy@5, accuracy@10, MRR
    """
    metrics = {
        'accuracy@1': calculate_accuracy_at_k(predictions, ground_truth, k=1),
        'accuracy@5': calculate_accuracy_at_k(predictions, ground_truth, k=5),
        'accuracy@10': calculate_accuracy_at_k(predictions, ground_truth, k=10),
        'mrr': calculate_mrr(predictions, ground_truth)
    }
    
    return metrics


def calculate_precision_recall_at_k(predictions: List[List], ground_truth: List[List], k: int = 5) -> Dict[str, float]:
    """
    Calcola precision e recall@k quando ci sono multipli synset corretti per query.
    
    Args:
        predictions: Lista di liste di synset predetti
        ground_truth: Lista di liste di synset corretti (possono essere multipli)
        k: Numero di top risultati da considerare
    
    Returns:
        Dizionario con 'precision@k' e 'recall@k'
    """
    if not predictions or not ground_truth:
        return {'precision@k': 0.0, 'recall@k': 0.0}
    
    precision_scores = []
    recall_scores = []
    
    for pred_list, gold_list in zip(predictions, ground_truth):
        if not gold_list:
            continue
        
        # Top k predizioni
        top_k = pred_list[:k] if len(pred_list) >= k else pred_list
        
        # Converti in nomi per confronto
        pred_names = {p.name() if hasattr(p, 'name') else str(p) for p in top_k}
        gold_names = {g.name() if hasattr(g, 'name') else str(g) for g in gold_list}
        
        # Calcola intersezione
        correct = len(pred_names.intersection(gold_names))
        
        # Precision: quanti dei predetti sono corretti
        precision = correct / len(top_k) if top_k else 0.0
        precision_scores.append(precision)
        
        # Recall: quanti dei corretti sono stati predetti
        recall = correct / len(gold_list) if gold_list else 0.0
        recall_scores.append(recall)
    
    return {
        'precision@k': np.mean(precision_scores) if precision_scores else 0.0,
        'recall@k': np.mean(recall_scores) if recall_scores else 0.0
    }


def analyze_prediction_distribution(predictions: List[List], ground_truth: List, max_rank: int = 20) -> Dict[int, int]:
    """
    Analizza la distribuzione dei rank delle predizioni corrette.
    
    Args:
        predictions: Lista di liste di synset predetti
        ground_truth: Lista di synset corretti
        max_rank: Rank massimo da considerare
    
    Returns:
        Dizionario {rank: count} con la frequenza di ogni rank
    """
    rank_distribution = {i: 0 for i in range(1, max_rank + 1)}
    rank_distribution[0] = 0  # Non trovato
    
    for pred_list, gold in zip(predictions, ground_truth):
        rank = find_rank_of_correct(pred_list, gold)
        
        if rank is None:
            rank_distribution[0] += 1
        elif rank <= max_rank:
            rank_distribution[rank] += 1
        else:
            rank_distribution[0] += 1
    
    return rank_distribution
