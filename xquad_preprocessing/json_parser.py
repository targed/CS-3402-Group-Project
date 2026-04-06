"""
JSONParser component for flattening nested XQuAD JSON structures.

This module provides functionality to extract question-answer pairs from the nested
XQuAD JSON format and convert them into a flat tabular structure suitable for
data processing and machine learning workflows.

"""

import pandas as pd
from typing import Dict, Set


class JSONParser:
    """
    Flattens nested XQuAD JSON structure into tabular format.
    
    This class navigates the nested XQuAD JSON structure (data → paragraphs → qas)
    and extracts question-answer pairs into a pandas DataFrame with language-specific
    column names. It ensures data quality by validating structure and checking for
    duplicate IDs.
    
    """
    
    def extract_qas(self, json_data: Dict, lang: str) -> pd.DataFrame:
        """
        Extracts question-answer pairs from nested JSON structure.
        
        Navigates the XQuAD JSON hierarchy (data → paragraphs → qas) to extract
        question IDs, contexts, questions, and answers. Creates language-specific
        column names and validates data integrity.

        Preconditions:
            - json_data contains 'data' key with list structure
            - lang is non-empty string
            - JSON follows XQuAD schema (data → paragraphs → qas)
            - Each QA has at least one answer
            
        Postconditions:
            - Returns DataFrame with columns: id, context_{lang}, question_{lang}, answer_{lang}
            - Number of rows equals total number of QAs in JSON
            - All rows have non-null values
            - No duplicate IDs in result (Requirement 6.2)
            
        Loop Invariants:
            - All records in records list have exactly 4 keys
            - All record values are non-empty strings
            - No duplicate IDs exist in accumulated records

        Args:
            json_data: XQuAD JSON data with structure data->paragraphs->qas
            lang: Language code for column naming (e.g., 'en', 'es')

        Returns:
            DataFrame with columns: id, context_{lang}, question_{lang}, answer_{lang}
            Each row represents one question-answer pair

        Raises:
            KeyError: If expected JSON keys ('data', 'paragraphs', 'qas', etc.) are missing
            IndexError: If answers array is empty for any question
            ValueError: If duplicate IDs are found in the data
            
        """
        records = []
        seen_ids: Set[str] = set()

        try:
            # Navigate nested structure: data -> paragraphs -> qas
            for article in json_data['data']:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    
                    for qa in paragraph['qas']:
                        # Extract fields
                        q_id = qa['id']
                        question = qa['question']
                        
                        # Handle empty answers array
                        if not qa['answers']:
                            raise IndexError(f"Empty answers array for question ID: {q_id}")
                        
                        # Select first valid answer
                        answer = qa['answers'][0]['text']
                        
                        # Check for duplicate IDs
                        if q_id in seen_ids:
                            raise ValueError(f"Duplicate question ID found: {q_id}")
                        seen_ids.add(q_id)
                        
                        # Build record with language-specific column names
                        record = {
                            'id': q_id,
                            f'context_{lang}': context,
                            f'question_{lang}': question,
                            f'answer_{lang}': answer
                        }
                        records.append(record)
        
        except KeyError as e:
            raise KeyError(f"Missing expected JSON key: {e}")
        
        return pd.DataFrame(records)
