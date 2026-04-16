import json
import os
import time
from pathlib import Path
from google import genai
from google.genai import types

# Initialize the client with your API key
API_KEY = os.getenv("GEMINI_API_KEY", "API-KEY") # Replace
client = genai.Client(api_key=API_KEY) # Prefer setting GEMINI_API_KEY in your environment
MODEL_ID = "gemini-3-flash-preview"
# MODEL_ID = "gemma-3-1b-it"
# MODEL_ID = "gemma-3-27b-it"

# Set to True if you want to run all 1190 questions. 
# Set to an integer (e.g., 100) if you only want a subset.
MAX_QUESTIONS = True

# Keep responses short but allow enough room to avoid empty MAX_TOKENS completions.
MAX_OUTPUT_TOKENS = 64


def _build_generation_config(enable_thinking=True):
    """Build config while staying compatible with different SDK versions."""
    config_kwargs = {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0.0,
    }

    # Some Gemini models spend output tokens on internal thinking.
    # Disabling thinking helps ensure we get final answer text.
    thinking_cls = getattr(types, "ThinkingConfig", None)
    if enable_thinking and thinking_cls is not None:
        config_kwargs["thinking_config"] = thinking_cls(thinking_budget=0)

    return types.GenerateContentConfig(**config_kwargs)


def _extract_answer_text(response):
    """Safely extract text from Gemini response objects."""
    if response is None:
        return ""

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text.strip()

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                return part_text.strip()

    return ""


def _save_predictions(predictions, output_filename):
    """Atomically persist predictions to avoid partial/corrupt files."""
    temp_filename = f"{output_filename}.tmp"
    with open(temp_filename, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    os.replace(temp_filename, output_filename)

def generate_gemini_predictions(language_file):
    print(f"Loading {language_file}...")
    with open(language_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Save to the exact format evaluate.py expects: {"id": "prediction"}
    # Works for paths like ../xquad/xquad.en.json -> en
    lang_code = Path(language_file).stem.split('.')[-1]
    output_filename = f"gemma-3-1b-it_{lang_code}_predictions.json"

    predictions = {}
    question_count = 0
    allow_thinking_config = True

    # Create/clear the output file at the beginning of a run.
    _save_predictions(predictions, output_filename)

    print(f"Starting inference for {language_file} using {MODEL_ID}...")
    
    # Iterate through the SQuAD/XQuAD JSON structure
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            
            for qa in paragraph["qas"]:
                # True means "no limit"; int means "process up to that many questions"
                if isinstance(MAX_QUESTIONS, int) and not isinstance(MAX_QUESTIONS, bool) and question_count >= MAX_QUESTIONS:
                    break
                
                q_id = qa["id"]
                question = qa["question"]
                
                # MATCHING QWEN'S PROMPT EXACTLY FOR A FAIR TEST
                prompt = f"{context} Answer in a few words with no explanation, {question}"
                
                try:
                    # System instructions help force Gemini to be brief and not chatty
                    try:
                        response = client.models.generate_content(
                            model=MODEL_ID,
                            contents=prompt,
                            config=_build_generation_config(enable_thinking=allow_thinking_config)
                        )
                    except Exception as inner_e:
                        error_text = str(inner_e)
                        if allow_thinking_config and "Thinking is not enabled" in error_text:
                            print("Info: model does not support thinking_config; retrying without it.")
                            response = client.models.generate_content(
                                model=MODEL_ID,
                                contents=prompt,
                                config=_build_generation_config(enable_thinking=False)
                            )
                            allow_thinking_config = False
                        else:
                            raise

                    answer = _extract_answer_text(response)
                    if not answer:
                        first_candidate = (getattr(response, "candidates", None) or [None])[0]
                        finish_reason = getattr(first_candidate, "finish_reason", None)
                        print(
                            f"Warning on question {q_id}: model returned no text "
                            f"(finish_reason={finish_reason}, max_output_tokens={MAX_OUTPUT_TOKENS})."
                        )
                except Exception as e:
                    print(f"API Error on question {q_id}: {e}")
                    answer = "" # Fallback on error
                
                predictions[q_id] = answer
                question_count += 1

                # Persist each answer immediately so progress survives failures.
                _save_predictions(predictions, output_filename)
                
                print(f"[{question_count}] ID: {q_id} | Ans: {answer}")
                
                # RATE LIMIT HANDLING: Free tier varies
                time.sleep(2.1)

            if isinstance(MAX_QUESTIONS, int) and not isinstance(MAX_QUESTIONS, bool) and question_count >= MAX_QUESTIONS:
                break

    print(f"Done! Saved {question_count} predictions to {output_filename}")

# Run for both English and Spanish
if __name__ == '__main__':
    # Make sure xquad.en.json and xquad.es.json are in the same directory
    generate_gemini_predictions("../xquad/xquad.en.json")
    generate_gemini_predictions("../xquad/xquad.es.json")