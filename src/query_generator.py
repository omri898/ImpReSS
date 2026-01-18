"""
Query Generator Module

This module provides on-the-fly diagnosis generation for ImpReSS training data.
It distills raw support conversations into concise diagnostic queries.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from .llm_handler import LLMHandler

logger = logging.getLogger(__name__)

# Enable progress bars for pandas operations
tqdm.pandas()


class QueryGenerator:
    """
    Generates diagnostic queries from support conversations using LLMs.
    
    Attributes:
        llm_handler: The LLM handler used for diagnosis generation.
        
    Example:
        >>> from src.llm_handler import LLMHandler
        >>> from src.query_generator import QueryGenerator
        >>> 
        >>> llm = LLMHandler(model_name="llama1b")
        >>> generator = QueryGenerator(llm)
        >>> augmented_df = generator.augment_data(
        ...     df=train_data,
        ...     dataset_name="ds_ct",
        ...     split_name="train",
        ...     cache_dir="cache",
        ...     use_cache=True,  # Load from cache if exists
        ...     save_cache=True  # Save to cache after generation
        ... )
    """
    
    def __init__(self, llm_handler: LLMHandler):
        """
        Initialize the QueryGenerator.
        
        Args:
            llm_handler: An initialized LLMHandler for diagnosis generation.
        """
        self.llm_handler = llm_handler
        
    def _generate_diagnosis(
        self,
        conversation: str,
        dataset_name: str,
        conversation_id: Optional[str] = None
    ) -> str:
        """
        Generate a single diagnostic query from a support conversation.
        
        Args:
            conversation: The raw conversation text (already cleaned)
            dataset_name: Dataset identifier (ds_ct, ds_is, ds_mu) for
                selecting appropriate prompt template
            conversation_id: Optional ID for logging
            
        Returns:
            Generated diagnostic query string, or empty string if generation fails (logged as error)
        """
        if pd.isna(conversation) or not conversation.strip():
            logger.warning(
                f"Empty conversation for {conversation_id or 'unknown'}, "
                "returning empty diagnosis"
            )
            return ""
        
        try:
            diagnosis = self.llm_handler.generate_with_template(
                prompt_name="diagnosis",
                variables={"conversation": conversation},
                dataset=dataset_name
            )
            return diagnosis.strip()
        
        except Exception as e:
            logger.error(
                f"Failed to generate diagnosis for {conversation_id or 'unknown'}: {e}"
            )
            return ""
    
    def augment_data(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        split_name: str,
        cache_dir: str,
        use_cache: bool = False,
        save_cache: bool = False
    ) -> pd.DataFrame:
        """
        Augment a DataFrame with diagnosis column, with optional caching.
        
        This method either loads cached diagnoses or generates new ones using
        the LLM. The cache file is named: {cache_dir}/{dataset_name}_{split_name}_queries.csv
        
        Args:
            df: DataFrame containing conversation data. Must have 'conversation_norm'
                and 'conversation_id' or 'question_id' columns.
            dataset_name: Dataset identifier (ds_ct, ds_is, ds_mu) used for
                          prompt selection and cache naming.
            split_name: Split identifier (train, val, test) for cache naming.
            cache_dir: Directory path for cache files.
            use_cache: If True, load from cache if available.
            save_cache: If True, save generated diagnoses to cache.
            
        Returns:
            DataFrame with added 'diagnosis' column.
            
        Raises:
            KeyError: If required columns are missing from the DataFrame.
        """
        # Determine conversation ID column
        if 'conversation_id' in df.columns:
            id_col = 'conversation_id'
        elif 'question_id' in df.columns:
            id_col = 'question_id'
        else:
            raise KeyError(
                "DataFrame must contain either 'conversation_id' or 'question_id' column"
            )
        
        # Check for conversation column
        if 'conversation_norm' not in df.columns:
            raise KeyError(
                "DataFrame must contain 'conversation_norm' column. "
                "Ensure data is loaded with proper text normalization."
            )
        
        # Construct cache path
        cache_path = Path(cache_dir) / f"{dataset_name}_{split_name}_queries.csv"
        
        # Try to load from cache
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached diagnoses from: {cache_path}")
            try:
                cache_df = pd.read_csv(cache_path)
                
                # Validate cache has required columns
                if id_col not in cache_df.columns or 'diagnosis' not in cache_df.columns:
                    logger.warning(
                        f"Cache file {cache_path} missing required columns, "
                        "regenerating diagnoses"
                    )
                else:
                    # Merge diagnosis column into original dataframe
                    df = df.merge(
                        cache_df[[id_col, 'diagnosis']],
                        on=id_col,
                        how='left'
                    )
                    
                    # Check if any diagnoses are missing
                    missing_count = df['diagnosis'].isna().sum()
                    if missing_count > 0:
                        logger.warning(
                            f"Cache loaded but {missing_count} diagnoses are missing. "
                            "Consider regenerating."
                        )
                    else:
                        logger.info(
                            f"Successfully loaded {len(df)} cached diagnoses for "
                            f"{dataset_name}/{split_name}"
                        )
                    
                    return df
                    
            except Exception as e:
                logger.error(f"Failed to load cache from {cache_path}: {e}")
                logger.info("Falling back to generation...")
        
        # Generate diagnoses
        logger.info(
            f"Generating diagnoses for {dataset_name}/{split_name} "
            f"({len(df)} conversations)..."
        )
        
        # Generate diagnosis for each row with progress bar
        diagnoses = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Generating {split_name}"):
            diagnosis = self._generate_diagnosis(
                conversation=row['conversation_norm'],
                dataset_name=dataset_name,
                conversation_id=row[id_col]
            )
            diagnoses.append(diagnosis)
        
        # Add diagnosis column to dataframe
        df['diagnosis'] = diagnoses
        
        logger.info(
            f"Successfully generated {len(df)} diagnoses for {dataset_name}/{split_name}"
        )
        
        # Count failures (empty diagnoses)
        failure_count = sum(1 for d in diagnoses if not d or not d.strip())
        if failure_count > 0:
            logger.warning(
                f"{failure_count}/{len(diagnoses)} diagnoses failed to generate "
                "and were set to empty strings"
            )
        
        # Save to cache if requested
        if save_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create cache dataframe with only ID and diagnosis
            cache_df = df[[id_col, 'diagnosis']].copy()
            
            try:
                cache_df.to_csv(cache_path, index=False)
                logger.info(f"Saved diagnoses cache to: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save cache to {cache_path}: {e}")
        
        return df
