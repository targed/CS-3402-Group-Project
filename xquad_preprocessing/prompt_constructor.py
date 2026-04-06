"""
PromptConstructor component for wrapping raw text into zero-shot prompt templates.

This module provides functionality to format question-answer contexts into
standardized zero-shot prompts for both English and Spanish, following
consistent template structures for downstream language model evaluation.

"""

import pandas as pd


class PromptConstructor:
    """
    Wraps raw text into zero-shot prompt templates.
    
    This class constructs standardized zero-shot prompts by combining context
    and question text with language-specific templates. It ensures consistent
    formatting across all samples and preserves the original DataFrame.
    
    """
    
    def add_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds zero-shot prompt columns to DataFrame.
        
        Creates two new columns (prompt_en and prompt_es) containing formatted
        prompts that combine context and question text according to language-specific
        templates. Returns a copy of the DataFrame to preserve immutability.

        Preconditions:
            - df contains columns: context_en, question_en, context_es, question_es
            - All text columns are non-null
            
        Postconditions:
            - Returns DataFrame with additional columns: prompt_en, prompt_es
            - All prompts follow consistent template format
            - Original DataFrame is not modified (returns copy)
            
        Loop Invariants:
            - Each row's prompt correctly references its context and question
            - Prompt format remains consistent across all rows

        Args:
            df: DataFrame with context and question columns for both languages

        Returns:
            DataFrame with additional columns: prompt_en, prompt_es
            Original DataFrame is not modified

        """
        df = df.copy()
        
        df['prompt_en'] = df.apply(
            lambda row: self.build_en_prompt(row['context_en'], row['question_en']),
            axis=1
        )
        
        df['prompt_es'] = df.apply(
            lambda row: self.build_es_prompt(row['context_es'], row['question_es']),
            axis=1
        )
        
        return df

    def build_en_prompt(self, context: str, question: str) -> str:
        """
        Constructs English zero-shot prompt.
        
        Formats context and question into the standardized English template:
        "Context: {context}\nQuestion: {question}\nAnswer strictly based on the context:"

        Preconditions:
            - context is non-empty string
            - question is non-empty string
            
        Postconditions:
            - Returns prompt following exact template format
            - Prompt contains "Context:", "Question:", and "Answer strictly based on the context:"

        Args:
            context: The context paragraph containing the answer
            question: The question to be answered

        Returns:
            Formatted English prompt string

        """
        return f"Context: {context}\nQuestion: {question}\nAnswer strictly based on the context:"

    def build_es_prompt(self, context: str, question: str) -> str:
        """
        Constructs Spanish zero-shot prompt.
        
        Formats context and question into the standardized Spanish template:
        "Contexto: {context}\nPregunta: {question}\nResponde estrictamente basándote en el contexto:"

        Preconditions:
            - context is non-empty string
            - question is non-empty string
            
        Postconditions:
            - Returns prompt following exact template format
            - Prompt contains "Contexto:", "Pregunta:", and "Responde estrictamente basándote en el contexto:"

        Args:
            context: The context paragraph containing the answer
            question: The question to be answered

        Returns:
            Formatted Spanish prompt string

        """
        return f"Contexto: {context}\nPregunta: {question}\nResponde estrictamente basándote en el contexto:"
