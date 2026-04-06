"""
DataDownloader component for fetching XQuAD JSON files from GitHub.

This module provides functionality to download XQuAD (Cross-lingual Question Answering Dataset)
JSON files from Google DeepMind's GitHub repository using secure HTTPS connections.

"""

import json
import urllib.request
import urllib.error
from typing import Dict, List


class DataDownloader:
    """
    Downloads XQuAD JSON files from Google DeepMind's GitHub repository.
    
    This class handles secure downloading of bilingual question-answering datasets
    from the official XQuAD repository, with comprehensive error handling for
    network and parsing failures.
    
    Attributes:
        BASE_URL: Base URL for XQuAD GitHub repository (uses HTTPS for security)
    
    """
    
    BASE_URL: str = "https://raw.githubusercontent.com/google-deepmind/xquad/master/"
    
    def download_xquad_data(self, languages: List[str]) -> Dict[str, dict]:
        """
        Downloads XQuAD JSON files for specified languages.
        
        Constructs GitHub raw content URLs for each language code, fetches the JSON
        data via HTTPS, and parses the responses into Python dictionaries. All network
        and parsing errors are propagated with descriptive context.

        Preconditions:
            - languages is a non-empty list of valid language codes
            - Internet connection is available
            - GitHub repository is accessible
            
        Postconditions:
            - Returns dictionary with one entry per language
            - Each entry contains valid parsed JSON data
            - JSON structure matches XQuAD schema
            - All URLs use HTTPS protocol (Requirement 10.1)

        Args:
            languages: List of language codes (e.g., ['en', 'es'])

        Returns:
            Dictionary mapping language codes to parsed JSON data
            Example: {'en': {...}, 'es': {...}}

        Raises:
            urllib.error.URLError: If download fails due to network issues
            json.JSONDecodeError: If JSON parsing fails due to invalid format
            
        """
        result = {}
        
        for lang in languages:
            # Construct GitHub raw content URL using HTTPS
            url = f"{self.BASE_URL}xquad.{lang}.json"
            
            try:
                # Fetch JSON data via HTTPS
                with urllib.request.urlopen(url) as response:
                    raw_data = response.read().decode('utf-8')
                
                # Parse JSON response into Python dictionary
                parsed_json = json.loads(raw_data)
                result[lang] = parsed_json
                
            except urllib.error.URLError as e:
                # Handle network failures with descriptive error message
                raise urllib.error.URLError(
                    f"Failed to download XQuAD data for language '{lang}' from {url}: {e.reason}"
                ) from e
            
            except json.JSONDecodeError as e:
                # Handle JSON parsing failures with descriptive error message
                raise json.JSONDecodeError(
                    f"Failed to parse JSON for language '{lang}' from {url}: {e.msg}",
                    e.doc,
                    e.pos
                ) from e
        
        return result
