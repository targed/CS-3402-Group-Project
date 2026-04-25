import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_DIR = "./gemma-3-270m-xquad-finetuned"
# BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
# ADAPTER_DIR = "./qwen-0.5b-xquad-finetuned"

print("Loading Base Model and Fine-Tuned Adapter...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

USE_CUDA = torch.cuda.is_available()
MODEL_DTYPE = torch.bfloat16 if USE_CUDA and torch.cuda.is_bf16_supported() else (torch.float16 if USE_CUDA else torch.float32)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, device_map="auto", dtype=MODEL_DTYPE)

# This merges your custom training onto the base model!
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval() # Set to evaluation mode

try:
    MODEL_DEVICE = next(model.parameters()).device
except StopIteration:
    MODEL_DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def generate_answers(language_file):
    print(f"\nProcessing {language_file}...")
    with open(language_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    predictions = {}
    
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                q_id = qa["id"]
                question = qa["question"]
                
                # The exact prompt format it was trained on
                prompt = f"{context} Answer in a few words with no explanation, {question} "

                tokenized = tokenizer(prompt, return_tensors="pt")
                input_ids = tokenized["input_ids"].to(MODEL_DEVICE)
                attention_mask = tokenized.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(MODEL_DEVICE)
                
                # Generate response
                with torch.no_grad():
                    output = model.generate(
                        input_ids, 
                        attention_mask=attention_mask,
                        max_new_tokens=20, 
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False # Strict, deterministic answers
                    )
                
                # Decode and grab ONLY the new text it generated
                full_text = tokenizer.decode(output[0], skip_special_tokens=True)
                answer = full_text[len(prompt):].strip()
                
                predictions[q_id] = answer
                print(f"ID: {q_id} | Ans: {answer}")

    # Save to JSON
    out_name = f"finetuned_gemma_3_270m_{Path(language_file).stem.split('.')[-1]}.json"
    with open(out_name, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)
    print(f"Saved to {out_name}")

if __name__ == "__main__":
    # generate_answers("../xquad/xquad.en.json")
    generate_answers("../xquad/xquad.es.json")
    