"""
Data validation utilities for XQuAD preprocessing pipeline.

This module provides validation functions to ensure data quality and correctness
of the processed XQuAD dataset, including answer substring validation, null value
checks, and ID uniqueness verification.

"""

import pandas as pd
from typing import List, Tuple, Dict, Optional


class DataValidator:
    """
    Validates data quality for XQuAD preprocessing pipeline.
    
    This class provides comprehensive validation methods to ensure the integrity
    and correctness of processed XQuAD data, including structural validation,
    content validation, and semantic validation.
    
    """
    
    def validate_answer_substring(
        self, 
        df: pd.DataFrame, 
        languages: List[str] = ['en', 'es']
    ) -> Tuple[bool, List[str]]:
        """
        Validates that answers are substrings of their corresponding contexts.
        
        Checks each row to ensure that the answer text appears within the context
        text for all specified languages. This validates the fundamental property
        that answers must be extractable from their contexts.

        Preconditions:
            - df contains context_{lang} and answer_{lang} columns for each language
            - All values are non-null strings
            
        Postconditions:
            - Returns validation status and list of errors
            - If valid, error list is empty
        
        Args:
            df: DataFrame with context and answer columns for each language
            languages: List of language codes to validate (default: ['en', 'es'])
        
        Returns:
            Tuple of (is_valid, error_messages)
            - is_valid: True if all answers are substrings of contexts
            - error_messages: List of error messages for invalid rows
        
        Raises:
            KeyError: If required columns (context_{lang}, answer_{lang}) are missing
            
        """
        error_messages = []
        
        for lang in languages:
            context_col = f'context_{lang}'
            answer_col = f'answer_{lang}'
            
            # Check if required columns exist
            if context_col not in df.columns:
                raise KeyError(f"Missing required column: {context_col}")
            if answer_col not in df.columns:
                raise KeyError(f"Missing required column: {answer_col}")
            
            # Validate each row
            for idx, row in df.iterrows():
                context = str(row[context_col])
                answer = str(row[answer_col])
                
                if answer not in context:
                    error_messages.append(
                        f"Row {idx} ({lang}): Answer '{answer[:50]}...' "
                        f"is not a substring of context"
                    )
        
        is_valid = len(error_messages) == 0
        return is_valid, error_messages
    
    def validate_non_null_columns(
        self, 
        df: pd.DataFrame, 
        critical_columns: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verifies that all critical columns have non-null values.
        
        Checks specified columns (or all columns if none specified) to ensure
        data completeness. This is essential for downstream processing that
        expects complete data.

        Preconditions:
            - df is non-empty DataFrame
            - If critical_columns specified, all must exist in df
            
        Postconditions:
            - Returns validation status and list of errors
            - If valid, no columns have null values
        
        Args:
            df: DataFrame to validate
            critical_columns: List of column names that must have non-null values.
                            If None, validates all columns.
        
        Returns:
            Tuple of (is_valid, error_messages)
            - is_valid: True if all critical columns have no null values
            - error_messages: List of error messages for columns with nulls
        
        Raises:
            KeyError: If specified critical columns don't exist in DataFrame
            
        """
        if critical_columns is None:
            critical_columns = df.columns.tolist()
        
        # Check if all specified columns exist
        missing_columns = set(critical_columns) - set(df.columns)
        if missing_columns:
            raise KeyError(f"Columns not found in DataFrame: {missing_columns}")
        
        error_messages = []
        
        for col in critical_columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                error_messages.append(
                    f"Column '{col}' has {null_count} null value(s) "
                    f"out of {len(df)} rows"
                )
        
        is_valid = len(error_messages) == 0
        return is_valid, error_messages
    
    def validate_unique_ids(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Checks that question IDs are unique within the DataFrame.
        
        Verifies that each question has a unique identifier, which is essential
        for data alignment and preventing duplicate processing.

        Preconditions:
            - df contains 'id' column
            
        Postconditions:
            - Returns validation status and list of errors
            - If valid, all IDs are unique
        
        Args:
            df: DataFrame with 'id' column
        
        Returns:
            Tuple of (is_valid, error_messages)
            - is_valid: True if all IDs are unique
            - error_messages: List of error messages for duplicate IDs
        
        Raises:
            KeyError: If 'id' column is missing
            
        """
        if 'id' not in df.columns:
            raise KeyError("DataFrame must have an 'id' column")
        
        error_messages = []
        
        # Find duplicate IDs
        duplicate_ids = df[df.duplicated(subset=['id'], keep=False)]['id'].unique()
        
        if len(duplicate_ids) > 0:
            for dup_id in duplicate_ids:
                count = (df['id'] == dup_id).sum()
                error_messages.append(
                    f"ID '{dup_id}' appears {count} times (expected 1)"
                )
        
        is_valid = len(error_messages) == 0
        return is_valid, error_messages
    
    def validate_all(
        self, 
        df: pd.DataFrame, 
        languages: List[str] = ['en', 'es'],
        critical_columns: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Dict[str, any]]]:
        """
        Runs all validation checks on the DataFrame.
        
        Executes a comprehensive suite of validation checks including ID uniqueness,
        null value detection, and answer substring validation. Returns detailed
        results for each check to facilitate debugging and quality assurance.

        Preconditions:
            - df is non-empty DataFrame
            - df contains expected columns for specified languages
            
        Postconditions:
            - Returns overall validation status and detailed results
            - All validation checks have been executed
            - Results dictionary contains status for each check
        
        Args:
            df: DataFrame to validate
            languages: List of language codes to validate (default: ['en', 'es'])
            critical_columns: List of critical column names. If None, uses default
                            set based on languages.
        
        Returns:
            Tuple of (is_valid, results)
            - is_valid: True if all validations pass
            - results: Dictionary with validation results for each check
                      Format: {check_name: {'valid': bool, 'errors': List[str]}}
        
        Example:
            validator = DataValidator()
            is_valid, results = validator.validate_all(df)
            if not is_valid:
                for check, data in results.items():
                    if not data['valid']:
                        print(f"{check} failed:")
                        for error in data['errors']:
                            print(f"  - {error}")
                            
        """
        if critical_columns is None:
            # Default critical columns based on languages
            critical_columns = ['id']
            for lang in languages:
                critical_columns.extend([
                    f'context_{lang}',
                    f'question_{lang}',
                    f'answer_{lang}'
                ])
        
        results = {}
        
        # Run unique ID validation
        try:
            is_valid_ids, id_errors = self.validate_unique_ids(df)
            results['unique_ids'] = {
                'valid': is_valid_ids,
                'errors': id_errors
            }
        except Exception as e:
            results['unique_ids'] = {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
        
        # Run non-null validation
        try:
            is_valid_nulls, null_errors = self.validate_non_null_columns(
                df, critical_columns
            )
            results['non_null_columns'] = {
                'valid': is_valid_nulls,
                'errors': null_errors
            }
        except Exception as e:
            results['non_null_columns'] = {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
        
        # Run answer substring validation
        try:
            is_valid_substring, substring_errors = self.validate_answer_substring(
                df, languages
            )
            results['answer_substring'] = {
                'valid': is_valid_substring,
                'errors': substring_errors
            }
        except Exception as e:
            results['answer_substring'] = {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"]
            }
        
        # Overall validation status
        all_valid = all(result['valid'] for result in results.values())
        
        return all_valid, results
