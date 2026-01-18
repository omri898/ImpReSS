import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DatasetType, get_cleaning_function, load_dataset
from src.llm_handler import LLMHandler
from src.model import BiEncoderImpress
from src.evaluation import evaluate_on_test_set, print_metrics_summary


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the training script.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to log file.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )
    
    # Suppress noisy libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
    logging.getLogger('faiss').setLevel(logging.WARNING)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_cli_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge command-line arguments with configuration.
    
    Command-line arguments override config file values.
    
    Args:
        config: Configuration dictionary from YAML file.
        args: Parsed command-line arguments.
        
    Returns:
        Merged configuration dictionary.
    """
    # Override training parameters if provided
    if args.epochs is not None:
        config.setdefault('training', {})['epochs'] = args.epochs
        
    if args.batch_size is not None:
        config.setdefault('training', {})['batch_size'] = args.batch_size
        
    if args.learning_rate is not None:
        config.setdefault('training', {})['learning_rate'] = args.learning_rate
    
    # Override paths if provided
    if args.train_path is not None:
        config.setdefault('data', {})['train_path'] = args.train_path
        
    if args.val_path is not None:
        config.setdefault('data', {})['val_path'] = args.val_path
        
    if args.test_path is not None:
        config.setdefault('data', {})['test_path'] = args.test_path
        
    if args.output_path is not None:
        config.setdefault('output', {})['model_save_path'] = args.output_path
    
    # Override dataset type if provided
    if args.dataset_type is not None:
        config['dataset_type'] = args.dataset_type
    
    return config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train ImpReSS BiEncoder model for product recommendation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with config file
    python scripts/train.py --config_path config/config.yaml
    
    # Override training parameters
    python scripts/train.py --config_path config/config.yaml --epochs 5 --batch_size 16
    
    # Specify dataset type
    python scripts/train.py --config_path config/config.yaml --dataset_type ds_ct
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the configuration YAML file.'
    )
    
    # Optional overrides - Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config).'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Training batch size (overrides config).'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='Learning rate (overrides config).'
    )
    
    # Optional overrides - Data paths
    parser.add_argument(
        '--train_path',
        type=str,
        default=None,
        help='Path to training data CSV (overrides config).'
    )
    parser.add_argument(
        '--val_path',
        type=str,
        default=None,
        help='Path to validation data CSV (overrides config).'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default=None,
        help='Path to test data CSV (overrides config).'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save trained models (overrides config).'
    )
    
    # Optional overrides - Dataset configuration
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['ds_ct', 'ds_is', 'ds_mu'],
        default=None,
        help='Type of dataset: ds_ct (customer support), ds_is (InfoSec SE), ds_mu (Music SE) (overrides config).'
    )
    
    # Logging options
    parser.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level.'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Path to log file (optional).'
    )
    
    # Evaluation options
    parser.add_argument(
        '--skip_evaluation',
        action='store_true',
        help='Skip test set evaluation after training.'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("="*60)
    logger.info("ImpReSS Training Pipeline")
    logger.info("="*60)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config_path}")
        config = load_config(args.config_path)
        config = merge_cli_args(config, args)
        
        # Extract configuration sections
        data_config = config.get('data', {})
        training_config = config.get('training', {})
        output_config = config.get('output', {})
        
        # Get dataset type
        dataset_type_str = config.get('dataset_type', 'ds_ct')
        dataset_type = DatasetType(dataset_type_str)
        
        logger.info(f"Dataset type: {dataset_type.value}")
        
        # Get cleaning function based on dataset type
        cleaning_function = get_cleaning_function(dataset_type)
        
        # Load data
        logger.info("Loading datasets...")
        data = load_dataset(
            dataset_type=dataset_type,
            train_path=data_config.get('train_path'),
            val_path=data_config.get('val_path'),
            test_path=data_config.get('test_path'),
            product_catalog_path=data_config.get('product_catalog_path')
        )
        
        # Get document columns
        document_columns = config.get('document_columns', ['gpt_description', 'gpt_features'])
        
        # Initialize model
        logger.info("Initializing BiEncoderImpress model...")
        embedding_config = config.get('embedding', {})
        
        model = BiEncoderImpress(
            base_embedding_model=embedding_config.get(
                'model_name', 'intfloat/multilingual-e5-large-instruct'
            ),
            cleaning_function=cleaning_function,
            negative_sampling_strategy=training_config.get(
                'negative_sampling', {}
            ).get('strategy', 'random'),
            num_negatives_per_positive=training_config.get(
                'negative_sampling', {}
            ).get('num_negatives_per_positive', 4),
        )
        
        # Load data into model
        model.load_data(
            train_data=data['train'],
            val_data=data['val'],
            test_data=data['test'],
            product_data=data['product_catalog'],
            available_products=data.get('available_products')
        )
        
        # Prepare training data
        logger.info("Preparing training triplets...")
        model.prepare_training_data(document_columns=document_columns)
        
        # Get training parameters
        epochs = training_config.get('epochs', 5)
        batch_size = training_config.get('batch_size', 16)
        learning_rate = training_config.get('learning_rate', 1e-6)
        warmup_steps = training_config.get('warmup_steps', 100)
        validation_steps = training_config.get('validation_steps', 100)
        
        early_stopping_config = training_config.get('early_stopping', {})
        patience = early_stopping_config.get('patience', 5)
        min_delta = early_stopping_config.get('min_delta', 0.001)
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(
            output_config.get('model_save_path', 'trained_models'),
            f"run_{timestamp}"
        )
        
        # Train model
        logger.info("Starting training...")
        trained_paths = model.train(
            document_columns=document_columns,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            model_save_path=model_save_path,
            validation_steps=validation_steps,
            top_k_val=config.get('evaluation', {}).get('top_k', 5),
            early_stopping_patience=patience,
            early_stopping_min_delta=min_delta
        )
        
        logger.info("Training completed!")
        logger.info(f"Models saved to: {model_save_path}")
        
        # Run evaluation on test set if not skipped
        if not args.skip_evaluation:
            logger.info("\n" + "="*60)
            logger.info("Running test set evaluation...")
            logger.info("="*60)
            
            # Create prediction function
            def predict_fn(query: str) -> list:
                results = model.predict(
                    query=query,
                    model_paths=trained_paths,
                    top_k=10
                )
                # Combine results from all models
                all_products = []
                for doc_col, predictions in results.items():
                    for product, score in predictions:
                        all_products.append((product, score))
                # Sort by score and deduplicate
                seen = set()
                unique_products = []
                for p, s in sorted(all_products, key=lambda x: x[1], reverse=True):
                    if p not in seen:
                        seen.add(p)
                        unique_products.append(p)
                return unique_products
            
            # Evaluate (always use 'diagnosis' as query column)
            aggregate_metrics, detailed_df = evaluate_on_test_set(
                test_data=data['test'],
                predict_fn=predict_fn,
                query_column='diagnosis',
                k_values=[1, 2, 3, 4, 5]
            )
            
            # Print summary
            print_metrics_summary(aggregate_metrics, title="Test Set Results")
            
            # Save detailed results
            results_path = os.path.join(
                output_config.get('results_path', 'results'),
                f"evaluation_{timestamp}.csv"
            )
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            detailed_df.to_csv(results_path, index=False)
            logger.info(f"Detailed results saved to: {results_path}")
        
        logger.info("\n" + "="*60)
        logger.info("Training pipeline completed successfully!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
