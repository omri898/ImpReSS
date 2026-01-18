"""
BiEncoder ImpReSS Model Module

This module implements the core BiEncoderImpress class for implicit recommendation
in support conversations.
"""

import gc
import json
import logging
import os
import random
import time
import uuid
from ast import literal_eval
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import DatasetType, get_cleaning_function, normalize_product_name
from .evaluation import compute_metrics

logger = logging.getLogger(__name__)


class BiEncoderImpress:
    """
    BiEncoder ImpReSS (Implicit Recommender System for Conversational Support Agents)
    
    
    Attributes:
        base_embedding_model: HuggingFace model ID (default: multilingual-e5-large-instruct, 560M)
        device: Device for computation ('cuda' or 'cpu')
        cleaning_function: Dataset-specific function to clean conversation text
        num_negatives_per_positive: Number of negatives per positive
        
    Example:
        >>> from src.data import get_cleaning_function, DatasetType
        >>> 
        >>> # Create model with dataset-specific cleaning
        >>> cleaning_fn = get_cleaning_function(DatasetType.DS_CT)
        >>> model = BiEncoderImpress(
        ...     base_embedding_model='intfloat/multilingual-e5-large-instruct',
        ...     cleaning_function=cleaning_fn,
        ...     num_negatives_per_positive=4 
        ... )
        >>> 
        >>> # Load data and train with multi-view architecture
        >>> model.load_data(train_df, val_df, test_df, product_df)
        >>> model.prepare_training_data(
        ...     document_columns=['gpt_features', 'tavily_features']
        ... )
        >>> model.train(
        ...     document_columns=['gpt_features', 'tavily_features'],
        ...     epochs=5,
        ...     batch_size=16
        ... )
    """
    
    def __init__(
        self,
        base_embedding_model: str = 'intfloat/multilingual-e5-large-instruct',
        cleaning_function: Optional[Callable[[str], str]] = None,
        num_negatives_per_positive: int = 4,
        device: Optional[str] = None
    ):
        """
        Initialize the BiEncoderImpress system.
        
        Args:
            base_embedding_model: HuggingFace model ID for embeddings.
                Defaults to 'intfloat/multilingual-e5-large-instruct' (560M params).
            cleaning_function: Dataset-specific function to clean conversation text.
                Should have signature (text: str) -> str.
                If None, no cleaning is applied. Use get_cleaning_function() to
                obtain appropriate function for your dataset.
            num_negatives_per_positive: Number of negative samples per positive.
                Defaults to 4.
            device: Device for computation. Auto-detected if None.
        """
        self.base_embedding_model = base_embedding_model
        self.cleaning_function = cleaning_function
        self.num_negatives_per_positive = num_negatives_per_positive
        # Always use 'diagnosis' as query source (per paper methodology)
        self.query_source_column = 'diagnosis'
        
        # Device configuration
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"BiEncoderImpress initialized")
        logger.info(f"  Embedding model: {self.base_embedding_model}")
        logger.info(f"  Device: {self.device}")
        
        if self.device == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name()}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  GPU Memory: {memory_gb:.1f} GB")
        
        # Data containers
        self.train_data: Optional[pd.DataFrame] = None
        self.val_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.product_data: Optional[pd.DataFrame] = None
        self.available_products: Optional[List[str]] = None
        
        # Training containers
        self.train_triplets: Optional[Dict[str, List[Tuple[str, str, str]]]] = None
        self.trained_model_paths: Optional[Dict[str, str]] = None
        
    def load_data(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        product_data: pd.DataFrame,
        available_products: Optional[List[str]] = None
    ) -> None:
        """
        Load pre-processed data into the model.
        
        This method expects DataFrames that have already been loaded and
        normalized using the data module's load_dataset function.
        
        Args:
            train_data: Training DataFrame with 'conversation_norm' column.
            val_data: Validation DataFrame with 'conversation_norm' column.
            test_data: Test DataFrame with 'conversation_norm' column.
            product_data: Product catalog DataFrame.
            available_products: Optional list of normalized product names.
                If None, extracted from product_data.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.product_data = product_data
        
        if available_products is not None:
            self.available_products = available_products
        elif 'category_norm' in product_data.columns:
            self.available_products = product_data['category_norm'].tolist()
        elif 'category' in product_data.columns:
            self.available_products = [
                normalize_product_name(p) for p in product_data['category'].tolist()
            ]
        else:
            raise ValueError("Cannot determine available products from product_data")
            
        logger.info(f"Data loaded:")
        logger.info(f"  Train: {len(self.train_data)} samples")
        logger.info(f"  Val: {len(self.val_data)} samples")
        logger.info(f"  Test: {len(self.test_data)} samples")
        logger.info(f"  Products: {len(self.available_products)} items")
        
    def _is_valid_ground_truth(self, ground_truth_str: str) -> bool:
        """
        Validate that a ground truth string is properly formatted.
        
        Args:
            ground_truth_str: The ground truth string to validate.
            
        Returns:
            True if valid, False otherwise.
        """
        if pd.isna(ground_truth_str):
            return False
            
        gt_str = str(ground_truth_str).strip()
        
        if not gt_str or gt_str == '[]' or gt_str == 'nan':
            return False
            
        try:
            gt_list = literal_eval(gt_str)
            return isinstance(gt_list, list) and len(gt_list) > 0
        except (ValueError, SyntaxError):
            return False
            
    def _parse_ground_truth(self, ground_truth_str: str) -> List[str]:
        """
        Parse ground truth string into list of normalized product names.
        
        Args:
            ground_truth_str: Ground truth string.
            
        Returns:
            List of normalized product names.
        """
        if pd.isna(ground_truth_str):
            return []
            
        try:
            gt_str = str(ground_truth_str).strip()
            
            if gt_str.startswith('[') and gt_str.endswith(']'):
                gt_list = literal_eval(gt_str)
                return [normalize_product_name(item) for item in gt_list if item]
            else:
                return [normalize_product_name(gt_str)]
        except (ValueError, SyntaxError):
            return [normalize_product_name(ground_truth_str)]
            
    def prepare_training_data(
        self,
        document_columns: List[str],
        query_column: Optional[str] = None
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Prepare training triplets for contrastive learning.
        
        Creates (query, positive, negative) triplets where:
        - query: The search query (e.g., LLM-generated diagnosis)
        - positive: Document from the ground truth product
        - negative: Document from a non-ground-truth product
        
        Args:
            document_columns: List of document column names from product_data
                to create triplets for (e.g., ['gpt_description', 'gpt_features']).
            query_column: Column to use as query source. If None, uses
                self.query_source_column.
                
        Returns:
            Dictionary mapping document column names to lists of triplets.
            
        Raises:
            ValueError: If data is not loaded or required columns are missing.
        """
        if self.train_data is None or self.product_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        query_col = query_column or self.query_source_column
        
        if query_col not in self.train_data.columns:
            raise ValueError(f"Query column '{query_col}' not found in train_data")
            
        # Validate document columns exist in product data
        missing_cols = [c for c in document_columns if c not in self.product_data.columns]
        if missing_cols:
            raise ValueError(f"Document columns not found in product_data: {missing_cols}")
            
        logger.info(f"Preparing training triplets...")
        logger.info(f"  Query column: {query_col}")
        logger.info(f"  Document columns: {document_columns}")
        
        # Create product lookup dictionaries
        product_docs: Dict[str, Dict[str, str]] = {}
        for _, row in self.product_data.iterrows():
            product_name = normalize_product_name(row.get('category', ''))
            if product_name:
                product_docs[product_name] = {
                    col: str(row.get(col, '')) for col in document_columns
                }
                
        # Generate triplets for each document column
        self.train_triplets = {}
        
        for doc_col in document_columns:
            triplets = []
            
            for _, row in tqdm(self.train_data.iterrows(), 
                               total=len(self.train_data),
                               desc=f"Creating triplets for {doc_col}"):
                # Get query
                query = str(row.get(query_col, '')).strip()
                if not query:
                    continue
                    
                # Get ground truth products
                if not self._is_valid_ground_truth(row.get('ground_truth', '')):
                    continue
                    
                gt_products = self._parse_ground_truth(row['ground_truth'])
                if not gt_products:
                    continue
                    
                # Get positive documents
                for gt_product in gt_products:
                    if gt_product not in product_docs:
                        continue
                        
                    positive_doc = product_docs[gt_product].get(doc_col, '')
                    if not positive_doc:
                        continue
                        
                    # Sample negative products
                    negative_products = [
                        p for p in product_docs.keys()
                        if p not in gt_products
                    ]
                    
                    if not negative_products:
                        continue
                        
                    # Sample negatives
                    num_negatives = min(
                        self.num_negatives_per_positive,
                        len(negative_products)
                    )
                    sampled_negatives = random.sample(negative_products, num_negatives)
                    
                    for neg_product in sampled_negatives:
                        negative_doc = product_docs[neg_product].get(doc_col, '')
                        if negative_doc:
                            triplets.append((query, positive_doc, negative_doc))
                            
            self.train_triplets[doc_col] = triplets
            logger.info(f"  {doc_col}: {len(triplets)} triplets created")
            
        return self.train_triplets
        
    def train(
        self,
        document_columns: List[str],
        epochs: int = 5,
        batch_size: int = 16,
        warmup_steps: int = 100,
        learning_rate: float = 1e-6,
        model_save_path: str = "trained_models",
        validation_steps: int = 100,
        top_k_val: int = 5,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.001
    ) -> Dict[str, str]:
        """
        Train bi-encoder models for each document column.
        
        Each document column is trained independently with its own epoch loop,
        validation, and early stopping. Models are loaded/freed one at a time
        to minimize GPU memory usage.
        
        Args:
            document_columns: Document column names to train models for.
            epochs: Number of training epochs per column. Defaults to 5.
            batch_size: Training batch size. Defaults to 16.
            warmup_steps: Warmup steps for learning rate. Defaults to 100.
            learning_rate: Learning rate. Defaults to 1e-6.
            model_save_path: Directory to save trained models.
            validation_steps: Steps between validation. Defaults to 100.
            top_k_val: Top-k for validation metrics. Defaults to 5.
            early_stopping_patience: Epochs without improvement before stopping.
            early_stopping_min_delta: Minimum improvement threshold.
            
        Returns:
            Dictionary mapping document column names to saved model paths.
            
        Raises:
            ValueError: If training triplets are not prepared.
        """
        if self.train_triplets is None:
            raise ValueError(
                "Training triplets not prepared. Call prepare_training_data() first."
            )
            
        missing_cols = [c for c in document_columns if c not in self.train_triplets]
        if missing_cols:
            raise ValueError(f"Triplets not found for columns: {missing_cols}")
            
        # Create save directory
        os.makedirs(model_save_path, exist_ok=True)
        checkpoint_dir = os.path.join(model_save_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        logger.info(f"Starting training...")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Model save path: {model_save_path}")
        
        self.trained_model_paths = {}
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
        # Train each document column independently
        for doc_col in document_columns:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model for: {doc_col}")
            logger.info(f"{'='*60}")
            
            triplets = self.train_triplets[doc_col]
            if not triplets:
                logger.warning(f"No triplets for {doc_col}, skipping.")
                continue
                
            # Early stopping state
            best_score = float('-inf')
            best_epoch = -1
            epochs_without_improvement = 0
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            model = None
            
            try:
                # Load model
                model = SentenceTransformer(
                    self.base_embedding_model,
                    device=self.device,
                    trust_remote_code=True
                )
                
                # Create training examples
                train_examples = [
                    InputExample(texts=[anchor, positive, negative])
                    for anchor, positive, negative in triplets
                ]
                
                # Create dataloader
                train_dataloader = DataLoader(
                    train_examples,
                    shuffle=True,
                    batch_size=batch_size,
                    num_workers=0
                )
                
                # Create loss
                train_loss = losses.TripletLoss(
                    model,
                    triplet_margin=0.8,
                    distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE
                )
                
                logger.info(f"Training with {len(triplets)} triplets...")
                
                # Training loop with early stopping
                for epoch in range(epochs):
                    logger.info(f"Epoch {epoch + 1}/{epochs}")
                    
                    # Train for one epoch
                    model.fit(
                        train_objectives=[(train_dataloader, train_loss)],
                        epochs=1,
                        warmup_steps=warmup_steps if epoch == 0 else 0,
                        show_progress_bar=True,
                        evaluation_steps=0,
                        optimizer_params={'lr': learning_rate},
                    )
                    
                    # Save model
                    model_path = os.path.join(model_save_path, f"{doc_col}_model")
                    model.save(model_path)
                    self.trained_model_paths[doc_col] = model_path
                    
                    # Validation
                    val_metrics = self._run_validation(
                        model, doc_col, top_k_val
                    )
                    
                    if val_metrics:
                        # Calculate early stopping score
                        current_score = self._calculate_early_stopping_score(val_metrics)
                        
                        if current_score > best_score + early_stopping_min_delta:
                            best_score = current_score
                            best_epoch = epoch + 1
                            epochs_without_improvement = 0
                            
                            # Save checkpoint
                            checkpoint_path = os.path.join(
                                checkpoint_dir,
                                f"{doc_col}_best.pt"
                            )
                            torch.save({
                                'epoch': epoch + 1,
                                'best_score': best_score,
                                'model_path': model_path,
                            }, checkpoint_path)
                            
                            logger.info(
                                f"New best score: {current_score:.4f} at epoch {epoch + 1}"
                            )
                        else:
                            epochs_without_improvement += 1
                            logger.info(
                                f"No improvement for {epochs_without_improvement} epochs "
                                f"(best: {best_score:.4f} at epoch {best_epoch})"
                            )
                            
                        if epochs_without_improvement >= early_stopping_patience:
                            logger.info(
                                f"Early stopping triggered at epoch {epoch + 1}"
                            )
                            break
                            
                logger.info(
                    f"Training complete for {doc_col}. "
                    f"Best score: {best_score:.4f} at epoch {best_epoch}"
                )
                
            finally:
                # Cleanup
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
        logger.info(f"\n{'='*60}")
        logger.info(f"Training completed for all columns")
        logger.info(f"{'='*60}")
        
        for doc_col, path in self.trained_model_paths.items():
            logger.info(f"  {doc_col}: {path}")
            
        return self.trained_model_paths
        
    def _run_validation(
        self,
        model: SentenceTransformer,
        doc_col: str,
        top_k: int = 5,
        sample_size: int = 50
    ) -> Dict[str, float]:
        """
        Run validation and compute metrics.
        
        Args:
            model: The SentenceTransformer model to validate.
            doc_col: Document column being validated.
            top_k: Number of top results to evaluate.
            sample_size: Number of validation samples to use.
            
        Returns:
            Dictionary of validation metrics.
        """
        if self.val_data is None or self.product_data is None:
            return {}
            
        try:
            # Get validation queries
            query_col = self.query_source_column
            if query_col not in self.val_data.columns:
                return {}
                
            # Get product documents
            valid_products = []
            valid_docs = []
            
            for _, row in self.product_data.iterrows():
                product_name = normalize_product_name(row.get('category', ''))
                doc = str(row.get(doc_col, '')).strip()
                if product_name and doc:
                    valid_products.append(product_name)
                    valid_docs.append(doc)
                    
            if not valid_docs:
                return {}
                
            # Create embeddings and index
            product_embeddings = model.encode(valid_docs, convert_to_tensor=False)
            embedding_dim = product_embeddings.shape[1]
            
            index = faiss.IndexFlatL2(embedding_dim)
            faiss.normalize_L2(product_embeddings)
            index.add(product_embeddings.astype('float32'))
            
            # Sample validation queries
            valid_val_data = [
                (row[query_col], self._parse_ground_truth(row.get('ground_truth', '')))
                for _, row in self.val_data.iterrows()
                if row.get(query_col) and self._is_valid_ground_truth(row.get('ground_truth', ''))
            ]
            
            if not valid_val_data:
                return {}
                
            sample_size = min(sample_size, len(valid_val_data))
            sampled_data = random.sample(valid_val_data, sample_size)
            
            # Compute metrics
            all_metrics: Dict[str, List[float]] = {}
            
            for query, ground_truth in sampled_data:
                query_embedding = model.encode([query], convert_to_tensor=False)
                faiss.normalize_L2(query_embedding)
                
                _, indices = index.search(query_embedding.astype('float32'), top_k)
                retrieved = [valid_products[i] for i in indices[0]]
                
                for k in range(1, top_k + 1):
                    metrics = compute_metrics(retrieved, ground_truth, k)
                    for name, value in metrics.items():
                        if name not in all_metrics:
                            all_metrics[name] = []
                        all_metrics[name].append(value)
                        
            # Average metrics
            avg_metrics = {
                name: np.mean(values) for name, values in all_metrics.items()
            }
            
            # Calculate early stopping score
            early_stopping_score = self._calculate_early_stopping_score(avg_metrics)
            
            logger.info(
                f"Validation metrics for {doc_col}: "
                f"MRR@1={avg_metrics.get('mrr@1', 0):.4f}, "
                f"MRR@3={avg_metrics.get('mrr@3', 0):.4f}, "
                f"MRR@5={avg_metrics.get('mrr@5', 0):.4f}, "
                f"Recall@3={avg_metrics.get('recall@3', 0):.4f}, "
                f"ES_Score={early_stopping_score:.4f}"
            )
            
            return avg_metrics
            
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
            return {}
            
    def _calculate_early_stopping_score(
        self,
        metrics: Dict[str, float]
    ) -> float:
        """
        Calculate early stopping score from validation metrics.
        
        Score = 0.5 * MRR@1 + 0.3 * MRR@3 + 0.2 * MRR@5
        
        Args:
            metrics: Dictionary of validation metrics.
            
        Returns:
            Calculated early stopping score.
        """
        mrr_1 = metrics.get('mrr@1', 0.0)
        mrr_3 = metrics.get('mrr@3', 0.0)
        mrr_5 = metrics.get('mrr@5', 0.0)
        
        return 0.5 * mrr_1 + 0.3 * mrr_3 + 0.2 * mrr_5
        
    def load_trained_model(
        self,
        model_path: str
    ) -> SentenceTransformer:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model directory.
            
        Returns:
            Loaded SentenceTransformer model.
        """
        return SentenceTransformer(model_path, device=self.device)
        
    def evaluate(
        self,
        model_path: str,
        document_columns: List[str],
        split: str = "test",
        top_k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        Evaluate the model on a data split.
        
        Args:
            model_path: Path to directory containing trained models.
            document_columns: List of document columns to evaluate.
            split: Data split to evaluate on ('test', 'val', 'train').
            top_k_values: List of k values for metrics.
            
        Returns:
            Dictionary of aggregate metrics.
        """
        from .evaluation import evaluate_on_test_set
        
        # Get the appropriate data split
        if split == "test":
            eval_data = self.test_data
        elif split == "val":
            eval_data = self.val_data
        elif split == "train":
            eval_data = self.train_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if eval_data is None:
            raise ValueError(f"{split} data not loaded")
        
        # Check for diagnosis column
        query_column = self.query_source_column
        if query_column not in eval_data.columns:
            raise ValueError(f"Query column '{query_column}' not found in {split} data")
        
        # Load models for each document column
        doc_col = document_columns[0]  # Use first doc column for evaluation
        model_file = Path(model_path) / f"{doc_col}_model"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        logger.info(f"Loading model from: {model_file}")
        model = SentenceTransformer(str(model_file), device=self.device)
        
        # Define prediction function
        def predict_fn(query: str) -> List[str]:
            # Get product documents
            valid_products = []
            valid_docs = []
            
            for _, row in self.product_data.iterrows():
                product_name = normalize_product_name(row.get('category', ''))
                doc = str(row.get(doc_col, '')).strip()
                if product_name and doc:
                    valid_products.append(product_name)
                    valid_docs.append(doc)
            
            if not valid_docs:
                return []
            
            # Create index
            product_embeddings = model.encode(valid_docs, convert_to_tensor=False)
            embedding_dim = product_embeddings.shape[1]
            
            index = faiss.IndexFlatL2(embedding_dim)
            faiss.normalize_L2(product_embeddings)
            index.add(product_embeddings.astype('float32'))
            
            # Search
            query_embedding = model.encode([query], convert_to_tensor=False)
            faiss.normalize_L2(query_embedding)
            
            _, indices = index.search(query_embedding.astype('float32'), max(top_k_values))
            retrieved = [valid_products[i] for i in indices[0]]
            
            return retrieved
        
        # Run evaluation
        aggregate_metrics, _ = evaluate_on_test_set(
            test_data=eval_data,
            predict_fn=predict_fn,
            query_column=query_column,
            k_values=top_k_values
        )
        
        # Cleanup
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return aggregate_metrics
    
    def predict(
        self,
        query: str,
        model_paths: Dict[str, str],
        top_k: int = 5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get product recommendations for a query.
        
        Args:
            query: The search query (e.g., diagnosis).
            model_paths: Dictionary mapping doc columns to model paths.
            top_k: Number of top results to return.
            
        Returns:
            Dictionary mapping doc columns to lists of (product, score) tuples.
        """
        if self.product_data is None:
            raise ValueError("Product data not loaded.")
            
        results = {}
        
        for doc_col, model_path in model_paths.items():
            # Load model
            model = SentenceTransformer(model_path, device=self.device)
            
            # Get product documents
            valid_products = []
            valid_docs = []
            
            for _, row in self.product_data.iterrows():
                product_name = normalize_product_name(row.get('category', ''))
                doc = str(row.get(doc_col, '')).strip()
                if product_name and doc:
                    valid_products.append(product_name)
                    valid_docs.append(doc)
                    
            if not valid_docs:
                results[doc_col] = []
                continue
                
            # Create index
            product_embeddings = model.encode(valid_docs, convert_to_tensor=False)
            embedding_dim = product_embeddings.shape[1]
            
            index = faiss.IndexFlatL2(embedding_dim)
            faiss.normalize_L2(product_embeddings)
            index.add(product_embeddings.astype('float32'))
            
            # Search
            query_embedding = model.encode([query], convert_to_tensor=False)
            faiss.normalize_L2(query_embedding)
            
            distances, indices = index.search(query_embedding.astype('float32'), top_k)
            
            # Convert distances to similarity scores
            retrieved = [
                (valid_products[idx], 1.0 - dist)
                for idx, dist in zip(indices[0], distances[0])
            ]
            
            results[doc_col] = retrieved
            
            # Cleanup
            del model
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
