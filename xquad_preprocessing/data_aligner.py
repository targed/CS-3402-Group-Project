"""
DataAligner component for aligning English and Spanish DataFrames.

This module provides functionality to merge bilingual question-answer pairs using
unique question IDs, ensuring semantic equivalence across languages and validating
data completeness.
"""

import pandas as pd
from typing import Set


class DataAligner:
    """
    Aligns English and Spanish DataFrames using unique question IDs.
    
    This class performs inner joins on bilingual DataFrames to create aligned
    question-answer pairs, validates data completeness, and reports alignment
    statistics. It ensures semantic equivalence through ID matching.
    
    """
    
    def align_dataframes(self, df_en: pd.DataFrame, df_es: pd.DataFrame) -> pd.DataFrame:
        """
        Merges English and Spanish DataFrames on question ID.
        
        Performs an inner join on the 'id' column to create bilingual question-answer
        pairs. Validates that all expected columns are present, all values are non-null,
        and reports alignment statistics including matched and unmatched pairs.

        Preconditions:
            - df_en contains columns: id, context_en, question_en, answer_en
            - df_es contains columns: id, context_es, question_es, answer_es
            - Both DataFrames have unique IDs (no duplicates)
            
        Postconditions:
            - Returns DataFrame with all columns from both inputs
            - Number of rows equals number of matching IDs
            - Each row has complete data for both languages
            - All values are non-null (Requirement 6.1)
            - Alignment count equals ID intersection (Requirement 3.5)

        Args:
            df_en: English DataFrame with columns [id, context_en, question_en, answer_en]
            df_es: Spanish DataFrame with columns [id, context_es, question_es, answer_es]

        Returns:
            Merged DataFrame with all columns from both languages
            Contains only rows where IDs exist in both input DataFrames

        Raises:
            ValueError: If DataFrames have no overlapping IDs, if merged DataFrame
                       is missing expected columns, if null values are present,
                       or if alignment count doesn't match ID intersection
                       
        """
        # Calculate alignment statistics before merge
        en_ids: Set[str] = set(df_en['id'])
        es_ids: Set[str] = set(df_es['id'])
        overlapping_ids: Set[str] = en_ids & es_ids
        
        # Requirement 3.4: Raise ValueError if no overlapping IDs exist
        if len(overlapping_ids) == 0:
            raise ValueError(
                f"No overlapping IDs found. "
                f"English DataFrame has {len(en_ids)} IDs, "
                f"Spanish DataFrame has {len(es_ids)} IDs, "
                f"but no IDs match between them."
            )
        
        # Requirement 3.1: Perform inner join on 'id' column
        df_merged = pd.merge(df_en, df_es, on='id', how='inner')
        
        # Requirement 3.2: Validate that merged DataFrame contains all columns from both inputs
        expected_columns = set(df_en.columns) | set(df_es.columns)
        actual_columns = set(df_merged.columns)
        if expected_columns != actual_columns:
            missing_columns = expected_columns - actual_columns
            raise ValueError(f"Merged DataFrame is missing columns: {missing_columns}")
        
        # Requirement 3.5: Report alignment statistics (matched vs unmatched pairs)
        matched_count = len(df_merged)
        en_unmatched = len(en_ids - es_ids)
        es_unmatched = len(es_ids - en_ids)
        
        print(f"Alignment Statistics:")
        print(f"  Matched pairs: {matched_count}")
        print(f"  English unmatched: {en_unmatched}")
        print(f"  Spanish unmatched: {es_unmatched}")
        
        # Requirement 3.3 & 6.1: Ensure all rows have non-null values in all columns
        null_counts = df_merged.isnull().sum()
        if null_counts.any():
            columns_with_nulls = null_counts[null_counts > 0].to_dict()
            raise ValueError(f"Aligned DataFrame contains null values: {columns_with_nulls}")
        
        # Verify alignment count equals ID intersection (Requirement 3.5)
        if len(df_merged) != len(overlapping_ids):
            raise ValueError(
                f"Alignment count mismatch: expected {len(overlapping_ids)} rows "
                f"but got {len(df_merged)} rows"
            )
        
        return df_merged
