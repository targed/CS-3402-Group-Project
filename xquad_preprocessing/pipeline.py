"""
Main pipeline orchestration for XQuAD data preprocessing.

This module provides the main entry point for the XQuAD preprocessing pipeline,
orchestrating all components in the correct sequence and handling error propagation.

"""

from typing import List, Tuple
from xquad_preprocessing.data_downloader import DataDownloader
from xquad_preprocessing.json_parser import JSONParser
from xquad_preprocessing.data_aligner import DataAligner
from xquad_preprocessing.prompt_constructor import PromptConstructor
from xquad_preprocessing.train_test_splitter import TrainTestSplitter


def process_xquad_pipeline(
    languages: List[str] = ['en', 'es'],
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[str, str]:
    """
    Main pipeline for XQuAD data preprocessing.

    Executes components in sequence: download → parse → align → prompt → split.
    All exceptions are allowed to propagate to the caller with descriptive context.
    
    This function orchestrates the complete preprocessing workflow from downloading
    raw XQuAD JSON files to producing train/test CSV files ready for model evaluation.

    Preconditions:
        - languages contains exactly 2 language codes
        - 0 < test_size < 1
        - random_state >= 0
        - Internet connection is available
        
    Postconditions:
        - Returns paths to train and test CSV files
        - Train CSV contains ~(1-test_size) of aligned samples
        - Test CSV contains ~test_size of aligned samples
        - Both CSVs have identical column structure
        - All components executed in correct order

    Args:
        languages: List of language codes (default ['en', 'es'])
        test_size: Proportion of data for test set (default 0.20)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Tuple of (train_path, test_path) as absolute paths to generated CSV files

    Raises:
        urllib.error.URLError: If download fails due to network issues
        json.JSONDecodeError: If JSON parsing fails due to invalid format
        KeyError: If expected JSON keys are missing from downloaded data
        IndexError: If answers array is empty for any question
        ValueError: If DataFrames have no overlapping IDs or validation fails
        IOError: If file writing fails due to permissions or disk issues
        
    """
    # Step 1: Download raw data
    downloader = DataDownloader()
    raw_data = downloader.download_xquad_data(languages)

    # Step 2: Parse JSON into DataFrames
    parser = JSONParser()
    df_en = parser.extract_qas(raw_data['en'], 'en')
    df_es = parser.extract_qas(raw_data['es'], 'es')

    # Step 3: Align DataFrames
    aligner = DataAligner()
    df_merged = aligner.align_dataframes(df_en, df_es)

    # Step 4: Add prompts
    prompter = PromptConstructor()
    df_prompted = prompter.add_prompts(df_merged)

    # Step 5: Split and save
    splitter = TrainTestSplitter()
    train_path, test_path = splitter.split_and_save(
        df_prompted,
        test_size=test_size,
        random_state=random_state
    )

    return train_path, test_path
