"""
XQuAD Data Preprocessing Pipeline

A Python-based data preprocessing pipeline for the XQuAD (Cross-lingual Question Answering Dataset)
that downloads bilingual JSON files from Google DeepMind's GitHub repository, flattens nested 
structures into aligned DataFrames, constructs zero-shot prompt templates, and outputs train/test 
CSV files for downstream model evaluation.

The pipeline ensures semantic equivalence between English and Spanish question-answer pairs through 
ID-based alignment and produces 1,190 aligned samples split 80/20 for training and testing.

Components:
    - DataDownloader: Downloads XQuAD JSON files from GitHub
    - JSONParser: Flattens nested JSON structures into tabular format
    - DataAligner: Aligns English and Spanish DataFrames using unique IDs
    - PromptConstructor: Wraps raw text into zero-shot prompt templates
    - TrainTestSplitter: Splits data and saves CSV files with proper encoding
    - DataValidator: Validates data quality and correctness
    - process_xquad_pipeline: Main pipeline orchestration function

Example:
    Basic usage of the pipeline:
    
    >>> from xquad_preprocessing import process_xquad_pipeline
    >>> train_path, test_path = process_xquad_pipeline(
    ...     languages=['en', 'es'],
    ...     test_size=0.20,
    ...     random_state=42
    ... )
    >>> print(f"Training data: {train_path}")
    >>> print(f"Test data: {test_path}")

"""

__version__ = "0.1.0"
__author__ = "XQuAD Preprocessing Team"
__license__ = "MIT"

from xquad_preprocessing.data_downloader import DataDownloader
from xquad_preprocessing.json_parser import JSONParser
from xquad_preprocessing.data_aligner import DataAligner
from xquad_preprocessing.prompt_constructor import PromptConstructor
from xquad_preprocessing.train_test_splitter import TrainTestSplitter
from xquad_preprocessing.data_validator import DataValidator
from xquad_preprocessing.pipeline import process_xquad_pipeline

__all__ = [
    "DataDownloader",
    "JSONParser",
    "DataAligner",
    "PromptConstructor",
    "TrainTestSplitter",
    "DataValidator",
    "process_xquad_pipeline",
]
