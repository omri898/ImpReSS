#!/usr/bin/env python3
"""
ImpReSS Main Entry Point

Unified command-line interface for the ImpReSS recommendation system.

Supported Datasets:
    - ds_ct: Cybersecurity Troubleshooting (user study)
    - ds_is: InfoSec Stack Exchange
    - ds_mu: Music Stack Exchange

Key Features:
    - Training with multi-view bi-encoders (intrinsic + extrinsic features)
    - Evaluation with MRR@k and Recall@k
    - Diagnosis caching for faster subsequent runs
    - Lightweight models (1B-8B params) for on-premise deployment

Usage Examples:
    # Train on InfoSec dataset with caching
    python main.py --dataset ds_is --mode train --use-cache --save-cache
    
    # Evaluate on Cybersecurity dataset
    python main.py --dataset ds_ct --mode evaluate
    
    # Train with custom hyperparameters
    python main.py --dataset ds_mu --mode train --epochs 5 --batch-size 16 \
        --learning-rate 1e-6 --llm-model llama1b
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path if running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))

from src.data import DatasetType, load_dataset, get_cleaning_function
from src.model import BiEncoderImpress
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Dataset name to type mapping
DATASET_TYPE_MAP: Dict[str, DatasetType] = {
    "ds_ct": DatasetType.DS_CT,
    "ds_is": DatasetType.DS_IS,
    "ds_mu": DatasetType.DS_MU,
}

# Multi-view document columns for each dataset
# Intrinsic view: gpt_features (LLM-generated, semantic alignment)
# Extrinsic view: tavily_features (web-sourced, real-world grounding)
DATASET_DOCUMENT_COLUMNS: Dict[str, List[str]] = {
    "ds_ct": ["gpt_features", "tavily_features"],
    "ds_is": ["gpt_features", "tavily_features"],
    "ds_mu": ["gpt_features", "tavily_features"],
}

# Default paths
DEFAULT_DATA_DIR = "data"
DEFAULT_MODEL_DIR = "trained_models"
DEFAULT_RESULTS_DIR = "results"


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure logging for the application."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def get_data_paths(dataset: str, data_dir: str) -> Dict[str, Path]:
    """
    Get the data file paths for a dataset.
    
    Args:
        dataset: Dataset name (ds_ct, ds_is, ds_mu).
        data_dir: Base data directory.
        
    Returns:
        Dictionary with paths to train, val, test, and spc_data files.
    """
    base_path = Path(data_dir) / dataset
    
    # Files are directly in the dataset folder (flattened structure)
    return {
        "train": base_path / "train_data.csv",
        "val": base_path / "validation_data.csv",
        "test": base_path / "test_data.csv",
        "spc_data": base_path / "spc_data.csv",
    }


def validate_paths(paths: Dict[str, Path]) -> None:
    """
    Validate that all required data paths exist.
    
    Args:
        paths: Dictionary of data paths.
        
    Raises:
        FileNotFoundError: If any required file is missing.
    """
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data files:\n" + "\n".join(f"  - {p}" for p in missing) +
            "\n\nRun 'python scripts/setup_repository_data.py' to set up the data."
        )


# =============================================================================
# Mode Handlers
# =============================================================================

def run_train(args: argparse.Namespace) -> None:
    """
    Run training mode.
    
    Trains a bi-encoder model on the specified dataset.
    """
    logger.info(f"Starting training on dataset: {args.dataset}")
    
    # Get data paths and validate
    paths = get_data_paths(args.dataset, args.data_dir)
    validate_paths(paths)
    
    # Get dataset type and document columns
    dataset_type = DATASET_TYPE_MAP[args.dataset]
    doc_columns = DATASET_DOCUMENT_COLUMNS[args.dataset]
    
    logger.info(f"Dataset type: {dataset_type.value}")
    logger.info(f"Document columns: {doc_columns}")
    
    # Load data
    logger.info("Loading data...")
    data = load_dataset(
        dataset_type=dataset_type,
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        product_catalog_path=paths["spc_data"],
    )
    
    # Initialize LLM for diagnosis generation
    logger.info("Initializing LLM for diagnosis generation...")
    logger.info(f"LLM model: {args.llm_model}")
    logger.info(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    from src.llm_handler import LLMHandler
    from src.query_generator import QueryGenerator
    
    llm_handler = LLMHandler(
        model_name=args.llm_model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        prompts_config_path="config/prompts.yaml"
    )
    
    # Initialize query generator
    query_generator = QueryGenerator(llm_handler)
    
    # Augment data with diagnoses (with caching support)
    # Only augment train/val for training to save time
    logger.info("Augmenting training and validation data with diagnoses...")
    data["train"] = query_generator.augment_data(
        df=data["train"],
        dataset_name=args.dataset,
        split_name="train",
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        save_cache=args.save_cache
    )
    
    data["val"] = query_generator.augment_data(
        df=data["val"],
        dataset_name=args.dataset,
        split_name="val",
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        save_cache=args.save_cache
    )
    
    logger.info("Data augmentation complete.")
    
    # Get cleaning function
    cleaning_function = get_cleaning_function(dataset_type)
    
    # Create model (always uses 'diagnosis' as query source)
    model = BiEncoderImpress(
        base_embedding_model=args.embedding_model,
        cleaning_function=cleaning_function,
        num_negatives_per_positive=args.num_negatives,
    )
    
    # Load data into model
    model.load_data(
        train_data=data["train"],
        val_data=data["val"],
        test_data=data["test"],
        product_data=data["product_catalog"],
        available_products=data["available_products"],
    )
    
    # Prepare training data
    model.prepare_training_data(document_columns=doc_columns)
    
    # Create output directory
    model_save_path = Path(args.model_dir) / args.dataset
    os.makedirs(model_save_path, exist_ok=True)
    
    # Train
    trained_paths = model.train(
        document_columns=doc_columns,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=str(model_save_path),
        early_stopping_patience=args.early_stopping_patience,
    )
    
    logger.info(f"Training complete. Models saved to: {model_save_path}")
    for col, path in trained_paths.items():
        logger.info(f"  - {col}: {path}")


def run_evaluate(args: argparse.Namespace) -> None:
    """
    Run evaluation mode.
    
    Evaluates a trained bi-encoder model on the test set.
    """
    logger.info(f"Starting evaluation on dataset: {args.dataset}")
    
    # Get data paths and validate
    paths = get_data_paths(args.dataset, args.data_dir)
    validate_paths(paths)
    
    # Get dataset type
    dataset_type = DATASET_TYPE_MAP[args.dataset]
    doc_columns = DATASET_DOCUMENT_COLUMNS[args.dataset]
    
    # Load data
    logger.info("Loading data...")
    data = load_dataset(
        dataset_type=dataset_type,
        train_path=paths["train"],
        val_path=paths["val"],
        test_path=paths["test"],
        product_catalog_path=paths["spc_data"],
    )
    
    # Initialize LLM for diagnosis generation
    logger.info("Initializing LLM for diagnosis generation...")
    logger.info(f"LLM model: {args.llm_model}")
    logger.info(f"Temperature: {args.temperature}, Max tokens: {args.max_tokens}")
    from src.llm_handler import LLMHandler
    from src.query_generator import QueryGenerator
    
    llm_handler = LLMHandler(
        model_name=args.llm_model,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
        prompts_config_path="config/prompts.yaml"
    )
    
    # Initialize query generator
    query_generator = QueryGenerator(llm_handler)
    
    # Augment test data with diagnoses (with caching support)
    logger.info("Augmenting test data with diagnoses...")
    data["test"] = query_generator.augment_data(
        df=data["test"],
        dataset_name=args.dataset,
        split_name="test",
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
        save_cache=args.save_cache
    )
    logger.info("Data augmentation complete.")
    
    # Get cleaning function and create model
    cleaning_function = get_cleaning_function(dataset_type)
    model = BiEncoderImpress(
        base_embedding_model=args.embedding_model,
        cleaning_function=cleaning_function,
    )
    
    # Load data into model
    model.load_data(
        train_data=data["train"],
        val_data=data["val"],
        test_data=data["test"],
        product_data=data["product_catalog"],
        available_products=data["available_products"],
    )
    
    # Load trained model and evaluate
    model_path = Path(args.model_path or args.model_dir) / args.dataset
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Train a model first with: python main.py --dataset {args.dataset} --mode train"
        )
    
    # Run evaluation on test set
    results = model.evaluate(
        model_path=str(model_path),
        document_columns=doc_columns,
        split="test",
        top_k_values=args.top_k,
    )
    
    # Save results
    results_path = Path(args.results_dir) / args.dataset
    os.makedirs(results_path, exist_ok=True)
    
    # Print and save results
    logger.info("Evaluation Results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save to file
    import json
    results_file = results_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ImpReSS: Implicit Recommendation System for Support Conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train on InfoSec dataset
    python main.py --dataset ds_is --mode train
    
    # Evaluate trained model
    python main.py --dataset ds_is --mode evaluate
    
    # Train with custom settings
    python main.py --dataset ds_mu --mode train --epochs 10 --batch-size 32
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["ds_ct", "ds_is", "ds_mu"],
        help="Dataset to use: ds_ct (customer support), ds_is (SE security), ds_mu (SE music)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate"],
        help="Operation mode: train or evaluate"
    )
    
    # Path arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Model save/load directory (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Specific model path for evaluation (overrides --model-dir)"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help=f"Results output directory (default: {DEFAULT_RESULTS_DIR})"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate (default: 1e-6)"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Early stopping patience in epochs (default: 5)"
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=4,
        help="Number of negatives per positive (default: 4)"
    )
    
    # Model configuration
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="intfloat/multilingual-e5-large-instruct",
        help="HuggingFace embedding model ID"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k values for evaluation metrics (default: 1 3 5)"
    )
    
    # LLM configuration (for diagnosis generation)
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama1b",
        help="LLM model for diagnosis generation (default: llama1b)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature for diagnosis generation (default: 0.3)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="LLM max tokens to generate for diagnosis (default: 512)"
    )
    
    # Diagnosis generation and caching
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Load diagnoses from cache if available (default: False)"
    )
    parser.add_argument(
        "--save-cache",
        action="store_true",
        help="Save generated diagnoses to cache for future use (default: False)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Directory for diagnosis cache files (default: cache)"
    )
    
    # General options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Log file path"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    
    logger.info("="*60)
    logger.info("ImpReSS: Implicit Recommendation System for Support Conversations")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Mode: {args.mode}")
    
    # Run the appropriate mode
    try:
        if args.mode == "train":
            run_train(args)
        elif args.mode == "evaluate":
            run_evaluate(args)
        else:
            parser.error(f"Unknown mode: {args.mode}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during {args.mode}: {e}")
        sys.exit(1)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
