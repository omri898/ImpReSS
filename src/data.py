"""
Data Loading and Text Normalization Module

This module provides utilities for loading and preprocessing datasets for ImpReSS.
It supports three real-world datasets where users seek problem resolution:
    - DS_CT: Customer Support/Chat (cybersecurity troubleshooting user study)
    - DS_IS: InfoSec Stack Exchange (information security Q&A)
    - DS_MU: Music Stack Exchange (music Q&A)

Key Features:
    - Unified data loading interface for multiple dataset types
    - Dataset-specific conversation text cleaning and normalization
    - Support for train/validation/test splits
    - Solution Product Category (SPC) catalog loading and normalization
    - Ground truth based on user acceptance signals (explicit ratings or accepted answers)
"""

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from tqdm import tqdm

# Enable progress bars for pandas operations
tqdm.pandas()

logger = logging.getLogger(__name__)


class DatasetType(Enum):
    """
    Enumeration of supported dataset types.
    
    Dataset naming:
        - DS_CT: Customer Support/Chat - Cybersecurity troubleshooting
        - DS_IS: InfoSec StackExchange - Information security Q&A
        - DS_MU: Music StackExchange - Music education/equipment Q&A
    """
    DS_CT = "ds_ct"      # Customer Support/Chat (formerly "human")
    DS_IS = "ds_is"      # InfoSec StackExchange (formerly "infosec")
    DS_MU = "ds_mu"      # Music StackExchange (formerly "music")
    
    # Legacy aliases for backward compatibility
    HUMAN = "ds_ct"
    STACKEXCHANGE = "stackexchange"  # Generic SE type


def clean_human_conversation(text: str) -> str:
    """
    Clean and normalize conversation text from DS_CT (Customer Support/Chat dataset).
    
    This function processes conversation text from the user study where participants
    engaged with a cybersecurity troubleshooting agent. The raw data contains
    structured sections separated by "---" delimiters and includes recommendation
    content in a block with an "Especially for you" text that should be removed.
    
    Args:
        text: The raw conversation text from DS_CT dataset.
    
    Returns:
        The cleaned conversation text suitable for diagnosis generation, or the original text if it's NaN/None.
    """
    if pd.isna(text):
        return text
    
    s = str(text)
    
    # Look for section separator
    sep_match = re.search(r'\n\s*---\s*\n', s)
    
    if not sep_match:
        # No separator found - return stripped text
        return s.strip()
    
    # Split at separator
    pre = s[:sep_match.start()]
    post = s[sep_match.end():]
    
    # Remove any "**Especially for you:**" promotional blocks from conversation history
    promo_idx = pre.rfind('**Especially for you:**')
    if promo_idx != -1:
        ai_turn = pre.rfind('\n - AI:', 0, promo_idx)
        pre = pre[:ai_turn if ai_turn != -1 else promo_idx].rstrip()
    
    # Extract first response block (include_response behavior)
    resp = re.split(r'\n\s*-\s*User:\s', post, maxsplit=1)[0].strip()
    if resp and not re.match(r'^\s*-\s*AI:', resp):
        resp = f" - AI: {resp.lstrip()}"
    
    combined = (pre.rstrip() + '\n' + resp).strip() if resp else pre.strip()
    return combined


def clean_stackexchange_conversation(text: str) -> str:
    """
    Clean and normalize conversation text from StackExchange data.
    
    StackExchange data typically doesn't require the same cleaning as human
    experiment data, so this function performs minimal normalization.
    
    Args:
        text: The raw conversation/question text. Can be NaN/None.
    
    Returns:
        The normalized conversation text.
    """
    if pd.isna(text):
        return text
    
    s = str(text).strip()
    return s


def get_cleaning_function(dataset_type: DatasetType) -> Callable[[str], str]:
    """
    Get the appropriate cleaning function for a dataset type.
    
    Args:
        dataset_type: The type of dataset being processed.
        
    Returns:
        A cleaning function that takes text and returns cleaned text.
    """
    # DS_CT (Customer Support) uses human conversation cleaning
    if dataset_type == DatasetType.DS_CT or dataset_type == DatasetType.HUMAN:
        return clean_human_conversation
    # DS_IS and DS_MU (StackExchange) use minimal cleaning
    elif dataset_type in (DatasetType.DS_IS, DatasetType.DS_MU, DatasetType.STACKEXCHANGE):
        return clean_stackexchange_conversation
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def normalize_product_name(product_name: str) -> str:
    """
    Normalize a product name for consistent matching.
    
    Args:
        product_name: The raw product name string.
        
    Returns:
        Normalized product name (lowercase, stripped, cleaned).
    """
    if pd.isna(product_name):
        return ""
    
    name = str(product_name).strip()
    # Remove quotes
    name = re.sub(r'^"+|"+$', '', name)
    # Remove common prefixes (e.g., "SecMate's")
    name = re.sub(r"^SecMate's\s+", '', name)
    # Normalize whitespace and convert to lowercase
    name = re.sub(r'\s+', ' ', name).lower()
    
    return name


def _load_and_normalize_dataset(
    data_path: Union[str, Path],
    data_name: str,
    cleaning_function: Callable[[str], str],
    conversation_column: str = "conversation"
) -> pd.DataFrame:
    """
    Load and normalize a single dataset file.
    
    Args:
        data_path: Path to the CSV data file.
        data_name: Name of the dataset (for logging).
        cleaning_function: Function to use for text cleaning.
        conversation_column: Name of the conversation column in the CSV.
        
    Returns:
        Loaded and normalized DataFrame.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        KeyError: If required columns are missing.
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading {data_name} data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Check for required columns
    if conversation_column not in df.columns:
        raise KeyError(f"Required column '{conversation_column}' not found in {data_path}")
    
    # Apply conversation normalization
    logger.info(f"Normalizing conversation text for {data_name} data...")
    df['conversation_norm'] = df[conversation_column].progress_apply(cleaning_function)
    
    logger.info(f"Loaded {len(df)} conversations for {data_name}.")
    return df


def load_product_catalog(
    catalog_path: Union[str, Path],
    category_column: str = "category"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load and normalize the product/service catalog.
    
    Args:
        catalog_path: Path to the product catalog CSV file.
        category_column: Name of the category column.
        
    Returns:
        Tuple of (DataFrame, list of normalized product names).
        
    Raises:
        FileNotFoundError: If the catalog file doesn't exist.
        KeyError: If required columns are missing.
    """
    catalog_path = Path(catalog_path)
    
    if not catalog_path.exists():
        raise FileNotFoundError(f"Product catalog not found: {catalog_path}")
    
    logger.info(f"Loading product catalog from {catalog_path}...")
    df = pd.read_csv(catalog_path)
    
    if category_column not in df.columns:
        raise KeyError(f"Required column '{category_column}' not found in {catalog_path}")
    
    # Normalize category names
    df['category_norm'] = df[category_column].apply(normalize_product_name)
    
    available_products = df['category_norm'].tolist()
    logger.info(f"Loaded {len(available_products)} products from catalog.")
    
    return df, available_products


def load_dataset(
    dataset_type: Union[str, DatasetType],
    train_path: Union[str, Path],
    val_path: Union[str, Path],
    test_path: Union[str, Path],
    product_catalog_path: Optional[Union[str, Path]] = None,
    conversation_column: str = "conversation"
) -> Dict[str, Union[pd.DataFrame, List[str]]]:
    """
    Load and preprocess train, validation, and test datasets.
    
    This is the main entry point for data loading. It handles both customer support
    data (DS_CT, with specialized conversation cleaning) and StackExchange data
    (DS_IS for InfoSec, DS_MU for Music).
    
    Args:
        dataset_type: Type of dataset ('ds_ct', 'ds_is', or 'ds_mu').
        train_path: Path to training data CSV.
        val_path: Path to validation data CSV.
        test_path: Path to test data CSV.
        product_catalog_path: Optional path to product catalog CSV.
        conversation_column: Name of the conversation column in CSV files.
        
    Returns:
        Dictionary containing:
            - 'train': Training DataFrame with 'conversation_norm' column
            - 'val': Validation DataFrame with 'conversation_norm' column
            - 'test': Test DataFrame with 'conversation_norm' column
            - 'product_catalog': Product catalog DataFrame (if path provided)
            - 'available_products': List of normalized product names (if path provided)
            
    Raises:
        ValueError: If invalid dataset_type is provided.
        FileNotFoundError: If any required file is not found.
        KeyError: If required columns are missing.
        
    Example:
        >>> data = load_dataset(
        ...     dataset_type='ds_ct',
        ...     train_path='data/train.csv',
        ...     val_path='data/val.csv', 
        ...     test_path='data/test.csv',
        ...     product_catalog_path='data/products.csv'
        ... )
        >>> train_df = data['train']
        >>> products = data['available_products']
    """
    # Convert string to enum if needed
    if isinstance(dataset_type, str):
        try:
            dataset_type = DatasetType(dataset_type.lower())
        except ValueError:
            raise ValueError(
                f"Invalid dataset_type: '{dataset_type}'. "
                f"Must be one of: {[t.value for t in DatasetType]}"
            )
    
    logger.info(f"Loading dataset with type: {dataset_type.value}")
    
    # Get the appropriate cleaning function
    cleaning_function = get_cleaning_function(dataset_type)
    
    # Load all datasets
    datasets = {
        'train': (train_path, "training"),
        'val': (val_path, "validation"),
        'test': (test_path, "test")
    }
    
    result: Dict[str, Union[pd.DataFrame, List[str]]] = {}
    
    for key, (path, name) in datasets.items():
        result[key] = _load_and_normalize_dataset(
            data_path=path,
            data_name=name,
            cleaning_function=cleaning_function,
            conversation_column=conversation_column
        )
    
    # Load product catalog if provided
    if product_catalog_path is not None:
        catalog_df, available_products = load_product_catalog(product_catalog_path)
        result['product_catalog'] = catalog_df
        result['available_products'] = available_products
    
    # Log summary
    logger.info(f"Dataset loading complete:")
    logger.info(f"  - Train: {len(result['train'])} samples")
    logger.info(f"  - Val: {len(result['val'])} samples")
    logger.info(f"  - Test: {len(result['test'])} samples")
    if 'available_products' in result:
        logger.info(f"  - Products: {len(result['available_products'])} items")
    
    return result


def validate_ground_truth(ground_truth_str: str) -> bool:
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
        from ast import literal_eval
        gt_list = literal_eval(gt_str)
        return isinstance(gt_list, list) and len(gt_list) > 0
    except (ValueError, SyntaxError):
        return False


def parse_ground_truth(ground_truth_str: str) -> List[str]:
    """
    Parse a ground truth string into a list of product names.
    
    Args:
        ground_truth_str: Ground truth string (could be JSON list or other format).
        
    Returns:
        List of ground truth product names.
    """
    if pd.isna(ground_truth_str):
        return []
    
    try:
        from ast import literal_eval
        
        gt_str = str(ground_truth_str).strip()
        
        # Try to parse as Python list
        if gt_str.startswith('[') and gt_str.endswith(']'):
            gt_list = literal_eval(gt_str)
            return [normalize_product_name(item) for item in gt_list if item]
        else:
            # Fallback: split by comma
            return [
                normalize_product_name(item.strip().strip("'\""))
                for item in gt_str.split(',')
                if item.strip()
            ]
    except (ValueError, SyntaxError):
        # Last resort: return as single item
        normalized = normalize_product_name(ground_truth_str)
        return [normalized] if normalized else []
