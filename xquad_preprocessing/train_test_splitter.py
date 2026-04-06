"""
TrainTestSplitter component for splitting data into training and test sets.

This module provides functionality to split DataFrames into training and test sets
with configurable ratios, save them as CSV files with proper encoding, and validate
split quality. Includes security features for path validation and file permissions.

"""

import pandas as pd
import os
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split


class TrainTestSplitter:
    """
    Splits data into training and test sets and saves as CSV.
    
    This class handles data splitting with reproducible random seeds, validates
    split quality, writes CSV files with UTF-8 encoding, and implements security
    measures including path sanitization and file permission management.
    
    """
    
    def _sanitize_path(self, path: str) -> str:
        """
        Sanitizes and validates output paths to prevent directory traversal attacks.
        
        Checks for directory traversal patterns (e.g., "../") and converts paths
        to absolute paths for consistency and security.

        Preconditions:
            - path is non-empty string
            - path is a valid file path
        Postconditions:
            - Returns sanitized absolute path
            - Path does not contain directory traversal sequences
        
        Args:
            path: User-provided file path
            
        Returns:
            Sanitized absolute path
            
        Raises:
            ValueError: If path contains directory traversal sequences
            
        """
        # Convert to Path object for safe manipulation
        path_obj = Path(path)
        
        # Check for directory traversal patterns in the original path
        # This prevents attacks like "../../../etc/passwd"
        if ".." in path_obj.parts:
            raise ValueError(
                f"Invalid path: directory traversal detected in '{path}'. "
                "Paths containing '..' are not allowed."
            )
        
        # Convert to absolute path for consistency
        abs_path = path_obj.resolve()
        
        return str(abs_path)
    
    def _set_file_permissions(self, path: str) -> None:
        """
        Sets appropriate file permissions on output CSV files.
        
        Sets permissions to 0o640 (read/write for owner, read for group, no access for others)
        to ensure appropriate access control on generated files.

        Preconditions:
            - path points to an existing file
            - path is a valid file path
        Postconditions:
            - File has permissions set to 0o640 (rw-r-----)
        
        Args:
            path: Path to the file
            
        """
        # Set permissions: rw-r----- (owner: read/write, group: read, others: none)
        os.chmod(path, 0o640)
    
    def split_and_save(
        self,
        df: pd.DataFrame,
        test_size: float = 0.20,
        random_state: int = 42,
        train_path: str = "xquad_train.csv",
        test_path: str = "xquad_test.csv"
    ) -> Tuple[str, str]:
        """
        Splits DataFrame and saves to CSV files.
        
        Splits the input DataFrame into training and test sets using sklearn's
        train_test_split, validates split quality, writes CSV files with UTF-8
        encoding, and sets appropriate file permissions. All paths are sanitized
        to prevent directory traversal attacks.

        Preconditions:
            - df is non-empty DataFrame
            - 0 < test_size < 1
            - random_state >= 0
            - Output paths are writable
            
        Postconditions:
            - Creates two CSV files at specified paths
            - Train CSV contains approximately (1 - test_size) * len(df) rows
            - Test CSV contains approximately test_size * len(df) rows
            - Both CSVs have identical column structure
            - Files have appropriate permissions (0o640)
            - Returns tuple of (train_path, test_path)

        Args:
            df: Complete DataFrame to split
            test_size: Proportion of data for test set (default 0.20)
            random_state: Random seed for reproducibility (default 42)
            train_path: Output path for training CSV
            test_path: Output path for test CSV

        Returns:
            Tuple of (train_path, test_path) as absolute paths

        Raises:
            IOError: If file writing fails
            ValueError: If split quality validation fails or path validation fails
            
        """
        # Task 11.1: Sanitize output paths to prevent directory traversal attacks
        try:
            sanitized_train_path = self._sanitize_path(train_path)
            sanitized_test_path = self._sanitize_path(test_path)
        except ValueError as e:
            raise ValueError(f"Path validation failed: {e}")
        
        # Task 7.1: Split DataFrame using sklearn
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )
        
        # Task 7.3: Validate split quality
        self._validate_split_quality(df, train_df, test_df, test_size)
        
        # Task 7.2: Write CSV files with UTF-8 encoding
        try:
            train_df.to_csv(sanitized_train_path, index=False, encoding='utf-8')
            test_df.to_csv(sanitized_test_path, index=False, encoding='utf-8')
            
            # Task 11.1: Set appropriate file permissions on output CSV files
            self._set_file_permissions(sanitized_train_path)
            self._set_file_permissions(sanitized_test_path)
        except Exception as e:
            raise IOError(f"Failed to write CSV files: {e}")
        
        return (sanitized_train_path, sanitized_test_path)
    
    def _validate_split_quality(
        self,
        original_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        requested_test_size: float
    ) -> None:
        """
        Validates split quality. 
        
        Checks that the split ratio is accurate (within 1%), train and test sets
        are disjoint (no overlapping IDs), and the union of train and test IDs
        equals the complete dataset.

        Preconditions:
            - original_df is the source DataFrame before split
            - train_df and test_df are results of split operation
            - 0 < requested_test_size < 1
            
        Postconditions:
            - Split ratio is within 1% of requested_test_size
            - Train and test sets have no overlapping IDs
            - Union of train and test IDs equals original IDs
        
        Args:
            original_df: Original DataFrame before split
            train_df: Training DataFrame after split
            test_df: Test DataFrame after split
            requested_test_size: Requested test size ratio
            
        Raises:
            ValueError: If validation fails (ratio inaccurate, sets not disjoint,
                       or union incomplete)
                       
        """
        # Requirement 5.8: Verify split ratio accuracy (within 1% of requested test_size)
        total_size = len(original_df)
        actual_test_size = len(test_df) / total_size
        ratio_diff = abs(actual_test_size - requested_test_size)
        
        if ratio_diff > 0.01:
            raise ValueError(
                f"Split ratio accuracy failed: requested {requested_test_size:.2%}, "
                f"got {actual_test_size:.2%} (difference: {ratio_diff:.2%})"
            )
        
        # Requirement 6.5: Ensure train and test sets are disjoint (no overlapping IDs)
        if 'id' in original_df.columns:
            train_ids = set(train_df['id'])
            test_ids = set(test_df['id'])
            overlap = train_ids & test_ids
            
            if overlap:
                raise ValueError(
                    f"Train and test sets are not disjoint: {len(overlap)} overlapping IDs"
                )
            
            # Requirement 6.6: Ensure union of train and test IDs equals complete dataset
            original_ids = set(original_df['id'])
            union_ids = train_ids | test_ids
            
            if union_ids != original_ids:
                missing = original_ids - union_ids
                extra = union_ids - original_ids
                raise ValueError(
                    f"Union of train and test IDs does not equal complete dataset. "
                    f"Missing: {len(missing)}, Extra: {len(extra)}"
                )
