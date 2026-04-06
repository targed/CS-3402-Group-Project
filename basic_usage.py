#!/usr/bin/env python3
"""
Example Usage Script for XQuAD Data Preprocessing Pipeline

This script demonstrates various ways to use the XQuAD preprocessing pipeline,
including basic pipeline execution, component-level usage, and error handling.
"""

import pandas as pd
import urllib.error
import json
from xquad_preprocessing.pipeline import process_xquad_pipeline
from xquad_preprocessing.data_downloader import DataDownloader
from xquad_preprocessing.json_parser import JSONParser
from xquad_preprocessing.data_aligner import DataAligner
from xquad_preprocessing.prompt_constructor import PromptConstructor
from xquad_preprocessing.train_test_splitter import TrainTestSplitter


def example_1_basic_pipeline():
    """
    Example 1: Basic Pipeline Execution
    
    Demonstrates the simplest way to use the pipeline with default parameters.
    """
    print("=" * 80)
    print("Example 1: Basic Pipeline Execution")
    print("=" * 80)
    
    try:
        # Run the complete pipeline with default parameters
        train_path, test_path = process_xquad_pipeline(
            languages=['en', 'es'],
            test_size=0.20,
            random_state=42
        )
        
        print(f"\n✓ Pipeline completed successfully!")
        print(f"  Training data saved to: {train_path}")
        print(f"  Test data saved to: {test_path}")
        
        return train_path, test_path
    
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise


def example_2_inspect_results(train_path: str, test_path: str):
    """
    Example 2: Loading and Inspecting Results
    
    Demonstrates how to load and examine the output CSV files.
    """
    print("\n" + "=" * 80)
    print("Example 2: Loading and Inspecting Results")
    print("=" * 80)
    
    # Load the CSV files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Total samples: {len(train_df) + len(test_df)}")
    print(f"  Split ratio: {len(test_df) / (len(train_df) + len(test_df)):.2%} test")
    
    print(f"\nColumns in output:")
    for col in train_df.columns:
        print(f"  - {col}")
    
    print(f"\nSample English Prompt:")
    print("-" * 80)
    print(train_df.iloc[0]['prompt_en'])
    print("-" * 80)
    print(f"Expected Answer: {train_df.iloc[0]['answer_en']}")
    
    print(f"\nSample Spanish Prompt:")
    print("-" * 80)
    print(train_df.iloc[0]['prompt_es'])
    print("-" * 80)
    print(f"Expected Answer: {train_df.iloc[0]['answer_es']}")


def example_3_component_level_usage():
    """
    Example 3: Component-Level Usage
    
    Demonstrates how to use individual components for more control.
    """
    print("\n" + "=" * 80)
    print("Example 3: Component-Level Usage")
    print("=" * 80)
    
    # Step 1: Download data
    print("\nStep 1: Downloading XQuAD data...")
    downloader = DataDownloader()
    raw_data = downloader.download_xquad_data(['en', 'es'])
    print(f"  ✓ Downloaded data for {len(raw_data)} languages")
    
    # Step 2: Parse JSON
    print("\nStep 2: Parsing JSON structures...")
    parser = JSONParser()
    df_en = parser.extract_qas(raw_data['en'], 'en')
    df_es = parser.extract_qas(raw_data['es'], 'es')
    print(f"  ✓ Extracted {len(df_en)} English Q&A pairs")
    print(f"  ✓ Extracted {len(df_es)} Spanish Q&A pairs")
    
    # Step 3: Align DataFrames
    print("\nStep 3: Aligning English and Spanish data...")
    aligner = DataAligner()
    df_merged = aligner.align_dataframes(df_en, df_es)
    print(f"  ✓ Aligned {len(df_merged)} question-answer pairs")
    
    # Step 4: Add prompts
    print("\nStep 4: Constructing zero-shot prompts...")
    prompter = PromptConstructor()
    df_prompted = prompter.add_prompts(df_merged)
    print(f"  ✓ Added prompt columns to {len(df_prompted)} rows")
    
    # Step 5: Split and save
    print("\nStep 5: Splitting and saving data...")
    splitter = TrainTestSplitter()
    train_path, test_path = splitter.split_and_save(
        df_prompted,
        test_size=0.20,
        random_state=42
    )
    print(f"  ✓ Saved training data to: {train_path}")
    print(f"  ✓ Saved test data to: {test_path}")


def example_4_error_handling():
    """
    Example 4: Error Handling Examples
    
    Demonstrates how to handle various error conditions.
    """
    print("\n" + "=" * 80)
    print("Example 4: Error Handling Examples")
    print("=" * 80)
    
    # Example 4a: Network error handling
    print("\nExample 4a: Handling Network Errors")
    try:
        downloader = DataDownloader()
        # This will fail if network is unavailable
        raw_data = downloader.download_xquad_data(['en'])
        print("  ✓ Download succeeded")
    except urllib.error.URLError as e:
        print(f"  ✗ Network error occurred: {e}")
        print("  → Check your internet connection and try again")
    
    # Example 4b: JSON parsing error handling
    print("\nExample 4b: Handling JSON Parsing Errors")
    try:
        parser = JSONParser()
        # Simulate malformed JSON
        malformed_json = {'wrong_key': []}
        df = parser.extract_qas(malformed_json, 'en')
    except KeyError as e:
        print(f"  ✗ JSON structure error: {e}")
        print("  → Verify the JSON file matches the expected XQuAD schema")
    
    # Example 4c: Alignment error handling
    print("\nExample 4c: Handling Alignment Errors")
    try:
        aligner = DataAligner()
        # Create DataFrames with no overlapping IDs
        df_en = pd.DataFrame({'id': ['1', '2'], 'context_en': ['a', 'b'], 
                              'question_en': ['q1', 'q2'], 'answer_en': ['a1', 'a2']})
        df_es = pd.DataFrame({'id': ['3', '4'], 'context_es': ['c', 'd'], 
                              'question_es': ['q3', 'q4'], 'answer_es': ['a3', 'a4']})
        df_merged = aligner.align_dataframes(df_en, df_es)
    except ValueError as e:
        print(f"  ✗ Alignment error: {e}")
        print("  → Ensure both JSON files are from the same XQuAD version")
    
    # Example 4d: File writing error handling
    print("\nExample 4d: Handling File Writing Errors")
    try:
        splitter = TrainTestSplitter()
        df = pd.DataFrame({'id': [1, 2, 3], 'data': ['a', 'b', 'c']})
        # Try to write to an invalid path
        train_path, test_path = splitter.split_and_save(
            df,
            train_path="/invalid/path/train.csv",
            test_path="/invalid/path/test.csv"
        )
    except (IOError, ValueError) as e:
        print(f"  ✗ File writing error: {e}")
        print("  → Check directory permissions and ensure output directory exists")


def example_5_custom_configuration():
    """
    Example 5: Custom Configuration
    
    Demonstrates how to customize pipeline parameters.
    """
    print("\n" + "=" * 80)
    print("Example 5: Custom Configuration")
    print("=" * 80)
    
    try:
        # Run pipeline with custom parameters
        train_path, test_path = process_xquad_pipeline(
            languages=['en', 'es'],
            test_size=0.30,  # 30% test split instead of default 20%
            random_state=123  # Different random seed
        )
        
        # Verify the custom split ratio
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        actual_test_ratio = len(test_df) / (len(train_df) + len(test_df))
        
        print(f"\n✓ Pipeline completed with custom configuration")
        print(f"  Requested test ratio: 30%")
        print(f"  Actual test ratio: {actual_test_ratio:.2%}")
        print(f"  Random seed: 123")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")


def main():
    """
    Main function to run all examples.
    """
    print("\n" + "=" * 80)
    print("XQuAD Data Preprocessing Pipeline - Example Usage")
    print("=" * 80)
    
    try:
        # Example 1: Basic pipeline execution
        train_path, test_path = example_1_basic_pipeline()
        
        # Example 2: Inspect results
        example_2_inspect_results(train_path, test_path)
        
        # Example 3: Component-level usage
        example_3_component_level_usage()
        
        # Example 4: Error handling
        example_4_error_handling()
        
        # Example 5: Custom configuration
        example_5_custom_configuration()
        
        print("\n" + "=" * 80)
        print("All examples completed!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
