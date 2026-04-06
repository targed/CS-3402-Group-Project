# XQuAD Data Preprocessing Pipeline

A Python-based data preprocessing pipeline for the XQuAD (Cross-lingual Question Answering Dataset) that downloads bilingual JSON files from Google DeepMind's GitHub repository, flattens nested structures, aligns English-Spanish pairs, constructs zero-shot prompts, and outputs train/test CSV files.

## Features

- **DataDownloader**: Downloads XQuAD JSON files from GitHub
- **JSONParser**: Flattens nested JSON structures into tabular format
- **DataAligner**: Aligns English and Spanish DataFrames using unique IDs
- **PromptConstructor**: Wraps raw text into zero-shot prompt templates
- **TrainTestSplitter**: Splits data into training and test sets and saves as CSV

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from xquad_preprocessing import process_xquad_pipeline

# Run the complete pipeline
train_path, test_path = process_xquad_pipeline(
    languages=['en', 'es'],
    test_size=0.20,
    random_state=42
)

print(f"Training data saved to: {train_path}")
print(f"Test data saved to: {test_path}")
```

## Project Structure

```
xquad_preprocessing/
├── __init__.py
├── data_downloader.py
├── json_parser.py
├── data_aligner.py
├── prompt_constructor.py
├── train_test_splitter.py
└── pipeline.py

```

## Requirements

- Python 3.8+
- pandas >= 1.3.0
- scikit-learn >= 0.24.0
- pytest >= 6.0.0
- hypothesis >= 6.0.0

## Available Examples

### basic_usage.py

Comprehensive example script covering all major use cases:

1. **Basic Pipeline Execution** - Simplest way to run the complete pipeline
2. **Loading and Inspecting Results** - How to load and examine output CSV files
3. **Component-Level Usage** - Using individual components for more control
4. **Error Handling** - Handling various error conditions gracefully
5. **Custom Configuration** - Customizing pipeline parameters

## Running the Examples

### Prerequisites

Make a Python virtual environment and ensure you have installed the package and its dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

### Run All Examples

```bash
python basic_usage.py
```

### Run Specific Examples

You can also import and run individual examples:

```python
from examples.basic_usage import example_1_basic_pipeline

train_path, test_path = example_1_basic_pipeline()
```

## Example Output

When you run the basic_usage.py script, you'll see:

- Progress messages for each pipeline stage
- Alignment statistics (matched/unmatched pairs)
- Dataset statistics (train/test split ratios)
- Sample prompts in both English and Spanish
- Error handling demonstrations

## Expected Results

After running the pipeline, you should have:

- `xquad_train.csv` - Training data (~952 samples, 80%)
- `xquad_test.csv` - Test data (~238 samples, 20%)
- Both files with UTF-8 encoding
- Columns: id, context_en, question_en, answer_en, context_es, question_es, answer_es, prompt_en, prompt_es

## License

See LICENSE file for details.
