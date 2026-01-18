"""
Evaluation Metrics Module

This module provides evaluation metrics for the ImpReSS implicit recommendation system.
Two metrics are measured: MRR@k and Recall@k, for k ∈ {1, 3, 5}. 
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_metrics(
    retrieved_products: List[str],
    ground_truth: List[str],
    k: int
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for a single query.
    
    Args:
        retrieved_products: List of retrieved SPC names.
        ground_truth: List of ground truth SPC names.
        k: Number of top results to consider.
        
    Returns:
        Dictionary containing calculated metrics:
            - recall@k: Fraction of relevant SPCs retrieved in top-k
            - mrr@k: Mean Reciprocal Rank (1/rank of first relevant SPC)
            
    """
    # Handle edge cases
    if not retrieved_products or not ground_truth:
        return {
            f'recall@{k}': 0.0,
            f'mrr@{k}': 0.0
        }
    
    # Take only top-k results
    retrieved_k = retrieved_products[:k]
    
    # Normalize for case-insensitive comparison
    retrieved_k_lower = [p.lower().strip() for p in retrieved_k]
    ground_truth_lower = [p.lower().strip() for p in ground_truth]
    
    # Calculate number of relevant items retrieved
    relevant_retrieved = len(set(retrieved_k_lower) & set(ground_truth_lower))
    
    # Recall@k: fraction of relevant that are retrieved
    recall_k = relevant_retrieved / len(ground_truth) if len(ground_truth) > 0 else 0.0
    
    # MRR@k: Mean Reciprocal Rank
    mrr_k = 0.0
    for i, product in enumerate(retrieved_k_lower):
        if product in ground_truth_lower:
            mrr_k = 1.0 / (i + 1)
            break
    
    return {
        f'recall@{k}': recall_k,
        f'mrr@{k}': mrr_k
    }


def compute_metrics_at_multiple_k(
    retrieved_products: List[str],
    ground_truth: List[str],
    k_values: List[int] = [1, 2, 3, 4, 5]
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for multiple k values.
    
    Args:
        retrieved_products: List of retrieved product names, ranked by relevance.
        ground_truth: List of ground truth product names.
        k_values: List of k values to compute metrics for.
        
    Returns:
        Dictionary containing all metrics for all k values.
        
    Example:
        >>> metrics = compute_metrics_at_multiple_k(
        ...     retrieved=['a', 'b', 'c', 'd', 'e'],
        ...     ground_truth=['b', 'e'],
        ...     k_values=[1, 3, 5]
        ... )
        >>> print(metrics['mrr@3'])
    """
    all_metrics = {}
    
    for k in k_values:
        metrics = compute_metrics(retrieved_products, ground_truth, k)
        all_metrics.update(metrics)
        
    return all_metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics from multiple queries.
    
    Computes the mean of each metric across all queries.
    
    Args:
        all_metrics: List of metric dictionaries, one per query.
        
    Returns:
        Dictionary containing mean values for each metric.
    """
    if not all_metrics:
        return {}
        
    aggregated = {}
    
    # Get all metric names from first entry
    metric_names = list(all_metrics[0].keys())
    
    for name in metric_names:
        values = [m.get(name, 0.0) for m in all_metrics]
        aggregated[name] = np.mean(values)
        
    return aggregated


def evaluate_on_test_set(
    test_data: pd.DataFrame,
    predict_fn: callable,
    ground_truth_column: str = 'ground_truth',
    query_column: str = 'diagnosis',
    k_values: List[int] = [1, 2, 3, 4, 5],
    parse_ground_truth_fn: Optional[callable] = None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate predictions on a test set.
    
    This function iterates through a test dataset, generates predictions
    for each sample, and computes aggregate metrics.
    
    Args:
        test_data: Test DataFrame containing queries and ground truth.
        predict_fn: Function that takes a query string and returns a list
            of predicted products ranked by relevance.
        ground_truth_column: Name of the ground truth column.
        query_column: Name of the query column.
        k_values: List of k values for metric computation.
        parse_ground_truth_fn: Optional function to parse ground truth strings.
            Should take a string and return a List[str].
            
    Returns:
        Tuple of:
            - Dictionary of aggregate metrics
            - DataFrame with detailed per-query results
            
    Example:
        >>> def my_predict(query):
        ...     # Returns list of product names
        ...     return model.predict(query, top_k=10)
        >>> 
        >>> agg_metrics, detailed = evaluate_on_test_set(
        ...     test_data=test_df,
        ...     predict_fn=my_predict,
        ...     query_column='diagnosis'
        ... )
        >>> print(f"MRR@5: {agg_metrics['mrr@5']:.4f}")
    """
    from ast import literal_eval
    
    # Default ground truth parser
    if parse_ground_truth_fn is None:
        def parse_ground_truth_fn(gt_str):
            if pd.isna(gt_str):
                return []
            try:
                gt_str = str(gt_str).strip()
                if gt_str.startswith('[') and gt_str.endswith(']'):
                    return literal_eval(gt_str)
                return [gt_str]
            except (ValueError, SyntaxError):
                return [str(gt_str).strip()]
    
    # Validate columns exist
    if query_column not in test_data.columns:
        raise ValueError(f"Query column '{query_column}' not found in test_data")
    if ground_truth_column not in test_data.columns:
        raise ValueError(f"Ground truth column '{ground_truth_column}' not found")
    
    # Initialize results storage
    all_query_metrics = []
    detailed_results = []
    
    # Process each test sample
    for idx, row in tqdm(test_data.iterrows(), 
                         total=len(test_data),
                         desc="Evaluating"):
        query = str(row[query_column]).strip()
        ground_truth = parse_ground_truth_fn(row[ground_truth_column])
        
        # Skip invalid samples
        if not query or not ground_truth:
            continue
            
        # Get predictions
        try:
            retrieved_products = predict_fn(query)
            if isinstance(retrieved_products, dict):
                # Handle multi-model output (take first)
                retrieved_products = list(retrieved_products.values())[0]
            if isinstance(retrieved_products[0], tuple):
                # Handle (product, score) tuples
                retrieved_products = [p for p, _ in retrieved_products]
        except Exception as e:
            logger.warning(f"Prediction failed for query {idx}: {e}")
            retrieved_products = []
        
        # Compute metrics
        query_metrics = compute_metrics_at_multiple_k(
            retrieved_products,
            ground_truth,
            k_values
        )
        all_query_metrics.append(query_metrics)
        
        # Store detailed results
        detailed_results.append({
            'query_id': idx,
            'query': query[:200] + '...' if len(query) > 200 else query,
            'ground_truth': ground_truth,
            'retrieved_products': retrieved_products[:max(k_values)],
            **query_metrics
        })
    
    # Aggregate metrics
    aggregate = aggregate_metrics(all_query_metrics)
    
    # Create detailed DataFrame
    detailed_df = pd.DataFrame(detailed_results)
    
    # Log summary
    logger.info("Evaluation complete:")
    logger.info(f"  Total samples evaluated: {len(all_query_metrics)}")
    for k in k_values:
        mrr = aggregate.get(f'mrr@{k}', 0.0)
        recall = aggregate.get(f'recall@{k}', 0.0)
        logger.info(f"  MRR@{k}: {mrr:.4f}, Recall@{k}: {recall:.4f}")
    
    return aggregate, detailed_df


def print_metrics_summary(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics",
    k_values: List[int] = [1, 3, 5]
) -> None:
    """
    Print a formatted summary of evaluation metrics.
    
    Args:
        metrics: Dictionary of computed metrics.
        title: Title for the summary.
        k_values: List of k values to display.
    """
    print(f"\n{'='*40}")
    print(f"{title}")
    print(f"{'='*40}")
    
    # Print header
    print(f"{'k':<4} {'Recall':<12} {'MRR':<12}")
    print(f"{'-'*28}")
    
    for k in k_values:
        recall = metrics.get(f'recall@{k}', 0.0)
        mrr = metrics.get(f'mrr@{k}', 0.0)
        
        print(f"{k:<4} {recall:<12.4f} {mrr:<12.4f}")
    
    print(f"{'='*40}")


def compare_models(
    model_metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'mrr@3'
) -> None:
    """
    Compare multiple models on a specific metric.
    
    Args:
        model_metrics: Dictionary mapping model names to their metrics.
        metric_name: The metric to compare on.
    """
    print(f"\n{'='*40}")
    print(f"Model Comparison: {metric_name}")
    print(f"{'='*40}")
    
    # Sort by metric value
    sorted_models = sorted(
        model_metrics.items(),
        key=lambda x: x[1].get(metric_name, 0.0),
        reverse=True
    )
    
    for rank, (model_name, metrics) in enumerate(sorted_models, 1):
        value = metrics.get(metric_name, 0.0)
        print(f"{rank}. {model_name}: {value:.4f}")
    
    print(f"{'='*40}")


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).
    
    RRF is a simple but effective method for combining rankings from
    multiple retrieval systems.
    
    Args:
        rankings: List of rankings, where each ranking is a list of items
            ordered by relevance (most relevant first).
        k: RRF parameter (typically 60).
        
    Returns:
        List of (item, score) tuples, sorted by combined score.
        
    Example:
        >>> ranking1 = ['a', 'b', 'c']
        >>> ranking2 = ['b', 'a', 'd']
        >>> combined = reciprocal_rank_fusion([ranking1, ranking2])
        >>> print(combined)
        [('b', 0.033), ('a', 0.032), ...]
    """
    scores: Dict[str, float] = {}
    
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            if item not in scores:
                scores[item] = 0.0
            scores[item] += 1.0 / (k + rank)
    
    # Sort by score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items
