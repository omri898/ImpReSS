"""
LLM Handler Module for Open-Source Models

This module provides a unified interface for loading and using, open-source
Large Language Models (LLMs) for diagnosis generation in ImpReSS.

Supported Models:
    - Llama 3.2 (1B, 3B) and Llama 3.1 (8B)
    - Mistral 7B v0.3
    - Qwen 2.5 (1.5B)

Note: This module intentionally excludes proprietary models (GPT-4, Claude, etc.)
to ensure the system can be deployed on-premise without external API dependencies.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


# Supported model mapping
SUPPORTED_MODELS: Dict[str, str] = {
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama1b": "meta-llama/Llama-3.2-1B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral7b-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


@dataclass
class TokenUsage:
    """Token usage statistics for LLM calls."""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class LLMHandler:
    """
    Handler for Open-Source LLMs for Diagnosis Generation.
    
    This class provides a unified interface for LLMs to distill support conversations into concise diagnostic queries.
    
    Attributes:
        model_name: The canonical model name (e.g., 'llama1b') or HuggingFace model ID
        device: The device to run inference on ('cuda' or 'cpu')
        temperature: Sampling temperature for generation (default: 0.3 for focused output)
        max_new_tokens: Maximum tokens to generate (default: 512 for concise diagnoses)
        
    Example:
        >>> handler = LLMHandler(
        ...     model_name="llama1b",
        ...     temperature=0.3,
        ...     max_new_tokens=512
        ... )
        >>> 
        >>> # Generate diagnostic query from support conversation
        >>> diagnosis = handler.generate_with_template(
        ...     prompt_name="diagnosis",
        ...     variables={"conversation": conversation_text},
        ...     dataset="ds_ct"
        ... )
        >>> print(diagnosis)
        "User experiencing slow PC performance. Suggested checking..."
    """
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        max_new_tokens: int = 512,
        device: Optional[str] = None,
        auth_token: Optional[str] = None,
        prompts_config_path: Optional[str] = None
    ):
        """
        Initialize the LLM Handler.
        
        Args:
            model_name: Model name key (e.g., 'llama8b') or full HuggingFace ID.
            temperature: Sampling temperature. Defaults to 0.3.
            max_new_tokens: Maximum tokens to generate. Defaults to 512.
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
            auth_token: HuggingFace auth token. Falls back to env var if None.
            prompts_config_path: Path to prompts.yaml configuration file.
            
        Raises:
            ValueError: If an unsupported model is specified.
        """
        # Resolve model ID
        self.model_key = model_name
        if model_name in SUPPORTED_MODELS:
            self.model_id = SUPPORTED_MODELS[model_name]
        else:
            # Assume it's a full HuggingFace model ID
            self.model_id = model_name
            
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Device configuration
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"LLMHandler initializing with model: {self.model_id}")
        logger.info(f"Device: {self.device}")
        
        # Get auth token
        self.auth_token = auth_token or self._get_auth_token()
        
        # Initialize components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.pipeline: Optional[Any] = None
        self._eos_token_ids: Optional[List[int]] = None
        
        # Load prompts configuration
        self.prompts: Dict[str, str] = {}
        if prompts_config_path:
            self._load_prompts(prompts_config_path)
        
        # Load the model
        self._load_model()
        
    def _get_auth_token(self) -> Optional[str]:
        """
        Get HuggingFace authentication token from environment.
        
        Returns:
            Auth token string or None if not found.
        """
        # Try TOKENS env var (JSON format)
        tokens_json = os.getenv("TOKENS")
        if tokens_json:
            try:
                tokens = json.loads(tokens_json)
                if self.model_id in tokens:
                    return tokens[self.model_id]
            except json.JSONDecodeError:
                pass
        
        # Try HF_TOKEN env var
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            return hf_token
            
        # logger.warning(
        #     f"No HuggingFace auth token found for {self.model_id}. "
        #     "Set TOKENS (JSON) or HF_TOKEN environment variable."
        # )
        return None
        
    def _load_prompts(self, prompts_path: str) -> None:
        """
        Load prompts from YAML configuration file.
        
        Args:
            prompts_path: Path to the prompts.yaml file.
        """
        prompts_path = Path(prompts_path)
        if not prompts_path.exists():
            logger.warning(f"Prompts config not found: {prompts_path}")
            return
            
        try:
            with open(prompts_path, 'r') as f:
                prompts_config = yaml.safe_load(f)
                
            # Extract prompt templates
            for key, value in prompts_config.items():
                if isinstance(value, dict) and 'template' in value:
                    self.prompts[key] = value['template']
                    
            logger.info(f"Loaded {len(self.prompts)} prompt templates")
        except Exception as e:
            logger.error(f"Failed to load prompts config: {e}")
            
    def _load_model(self) -> None:
        """
        Load the LLM model and tokenizer.
        
        Raises:
            RuntimeError: If model loading fails.
        """
        try:
            from langchain_huggingface import HuggingFacePipeline
            
            logger.info(f"Loading tokenizer for {self.model_id}...")
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    token=self.auth_token,
                    use_fast=True,
                    trust_remote_code=True,
                )
            except TypeError:
                # Fallback for older transformers versions
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    use_auth_token=self.auth_token,
                    use_fast=True,
                    trust_remote_code=True,
                )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Collect EOS token IDs
            self._eos_token_ids = self._collect_eos_token_ids()
            
            logger.info(f"Loading model {self.model_id}...")
            
            # Determine dtype
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # Pipeline kwargs
            pipeline_kwargs = {
                "temperature": self.temperature,
                "max_new_tokens": self.max_new_tokens,
                "clean_up_tokenization_spaces": True,
                "return_full_text": False,
                "token": self.auth_token,
            }
            
            if self._eos_token_ids:
                pipeline_kwargs["eos_token_id"] = self._eos_token_ids
                
            # Create pipeline
            self.pipeline = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                model_kwargs={
                    "device_map": "auto",
                    "torch_dtype": torch_dtype,
                    "low_cpu_mem_usage": True,
                },
                pipeline_kwargs=pipeline_kwargs,
            )
            
            logger.info(f"Model {self.model_id} loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_id}: {e}")
            
    def _collect_eos_token_ids(self) -> List[int]:
        """
        Collect all relevant EOS token IDs for the model.
        
        Returns:
            List of EOS token IDs.
        """
        eos_ids: List[int] = []
        
        if self.tokenizer is None:
            return eos_ids
            
        # Add main EOS token
        if self.tokenizer.eos_token_id is not None:
            eos_ids.append(self.tokenizer.eos_token_id)
            
        # Add model-specific special tokens
        special_eos_tokens = ["<|eot_id|>", "<|im_end|>", "<|end|>"]
        vocab = self.tokenizer.get_vocab()
        
        for token in special_eos_tokens:
            if token in vocab:
                token_id = self.tokenizer.convert_tokens_to_ids(token)
                if token_id is not None:
                    eos_ids.append(token_id)
                    
        # Remove duplicates while preserving order
        eos_ids = list(dict.fromkeys(eos_ids))
        
        return eos_ids
        
    def get_prompt(self, prompt_name: str, dataset: Optional[str] = None) -> str:
        """
        Retrieve a prompt template by name, with optional dataset-specific resolution.
        
        For diagnosis prompts, the dataset parameter enables automatic selection of
        the appropriate dataset-specific prompt (e.g., 'human_diagnosis', 
        'infosec_diagnosis', 'music_diagnosis').
        
        Args:
            prompt_name: The name of the prompt to retrieve (e.g., 'diagnosis').
            dataset: Optional dataset name to select dataset-specific prompts.
                     Supported values: 'ds_ct', 'ds_is', 'ds_mu'.
                     
        Returns:
            The prompt template string.
            
        Note:
            If the prompt is not found in the configuration, returns a
            placeholder message indicating it needs to be replaced.
            
        Example:
            >>> # Get dataset-specific diagnosis prompt
            >>> prompt = handler.get_prompt("diagnosis", dataset="ds_is")
            >>> # Returns the 'ds_is_diagnosis' template
        """
        # If dataset is provided and this is a diagnosis-type prompt,
        # try to get the dataset-specific version first
        if dataset:
            dataset_specific_key = f"{dataset}_{prompt_name}"
            if dataset_specific_key in self.prompts:
                return self.prompts[dataset_specific_key]
        
        # Fall back to the exact prompt name
        if prompt_name in self.prompts:
            return self.prompts[prompt_name]
            
        # Return placeholder if not found
        return (
            f"Temporary placeholder for {prompt_name}. "
            "This needs to be replaced with actual prompt from LangSmith."
        )
        
    def format_chat_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Format chat messages using the model's chat template.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            
        Returns:
            Formatted prompt string ready for the model.
            
        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> formatted = handler.format_chat_messages(messages)
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
            
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The user prompt/query.
            system_prompt: Optional system prompt to prepend.
            
        Returns:
            Generated text response.
            
        Raises:
            RuntimeError: If the model is not loaded.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
            
        # Build messages
        messages: List[Dict[str, str]] = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        messages.append({"role": "user", "content": prompt})
        
        # Format prompt
        formatted_prompt = self.format_chat_messages(messages)
        
        # Generate
        response = self.pipeline.invoke(formatted_prompt)
        
        # Extract text
        if isinstance(response, str):
            return response.strip()
        elif hasattr(response, 'content'):
            return str(response.content).strip()
        else:
            return str(response).strip()
            
    def generate_with_template(
        self,
        prompt_name: str,
        variables: Dict[str, Any],
        dataset: Optional[str] = None
    ) -> str:
        """
        Generate text using a named prompt template.
        
        Args:
            prompt_name: Name of the prompt template to use.
            variables: Dictionary of variables to substitute in the template.
            dataset: Optional dataset name for dataset-specific prompts.
                     Supported values: 'ds_ct', 'ds_is', 'ds_mu'.
            
        Returns:
            Generated text response.
            
        Example:
            >>> # Using dataset-specific diagnosis prompt
            >>> response = handler.generate_with_template(
            ...     "diagnosis",
            ...     {"conversation": "User: My computer won't start..."},
            ...     dataset="ds_ct"
            ... )
        """
        template = self.get_prompt(prompt_name, dataset=dataset)
        
        # Substitute variables
        try:
            prompt = template.format(**variables)
        except KeyError as e:
            logger.warning(f"Missing variable in template: {e}")
            prompt = template
            
        return self.generate(prompt)
        
    def calculate_token_usage(
        self,
        prompt_text: str,
        response_text: str
    ) -> TokenUsage:
        """
        Calculate token usage for a prompt-response pair.
        
        Args:
            prompt_text: The input prompt text.
            response_text: The generated response text.
            
        Returns:
            TokenUsage object with input, output, and total token counts.
        """
        if self.tokenizer is None:
            return TokenUsage(0, 0, 0)
            
        try:
            input_tokens = len(self.tokenizer.encode(
                prompt_text, add_special_tokens=True
            ))
            output_tokens = len(self.tokenizer.encode(
                response_text, add_special_tokens=True
            ))
            
            return TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens
            )
        except Exception as e:
            logger.warning(f"Failed to calculate token usage: {e}")
            return TokenUsage(0, 0, 0)
            
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready for inference.
        
        Returns:
            True if the model is loaded, False otherwise.
        """
        return self.pipeline is not None and self.tokenizer is not None
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information.
        """
        return {
            "model_key": self.model_key,
            "model_id": self.model_id,
            "device": self.device,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "is_loaded": self.is_loaded(),
            "eos_token_ids": self._eos_token_ids,
        }
        
    def unload(self) -> None:
        """
        Unload the model to free memory.
        """
        import gc
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        gc.collect()
        logger.info(f"Model {self.model_id} unloaded")
