"""
ImpReSS: Implicit Recommendation System for Support Conversations

A lightweight implicit recommender system for conversational support agents
that identifies relevant Solution Product Categories (SPCs) from problem-solving
dialogues without assuming user purchasing intent.

Published at IUI 2026 (31st International Conference on Intelligent User Interfaces)

Key Components:
    - Two-stage approach: LLM diagnosis generation + bi-encoder retrieval
    - Multi-view architecture: Intrinsic (LLM-generated) + Extrinsic (web-sourced) features
    - Lightweight models (1B-8B params) for on-premise deployment
    - Contrastive learning with triplet loss for semantic matching
    - RRF fusion for final ranking

Modules:
    - data: Dataset loading and text normalization (DS_CT, DS_IS, DS_MU)
    - model: BiEncoderImpress class with multi-view training and FAISS search
    - llm_handler: Lightweight open-source LLM inference (Llama, Mistral, Qwen)
    - query_generator: Diagnosis generation from raw conversations
    - evaluation: Metrics computation (MRR@k, Recall@k for k ∈ {1, 3, 5})
"""

__author__ = "Omri Haller"

from .data import load_dataset, DatasetType
from .model import BiEncoderImpress
from .llm_handler import LLMHandler
from .evaluation import compute_metrics, evaluate_on_test_set

__all__ = [
    "load_dataset",
    "DatasetType",
    "BiEncoderImpress",
    "LLMHandler",
    "compute_metrics",
    "evaluate_on_test_set",
]
