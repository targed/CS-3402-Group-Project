import os
import subprocess
import sys


def _ensure_utf8_mode():
    """Relaunch with UTF-8 mode so third-party templates decode reliably on Windows."""
    if sys.flags.utf8_mode:
        return

    if os.environ.get("FINETUNE_UTF8_REEXEC") == "1":
        return

    env = os.environ.copy()
    env["FINETUNE_UTF8_REEXEC"] = "1"
    env["PYTHONUTF8"] = "1"

    # os.execve can be unstable on some Windows/Python builds. Use a safe handoff.
    try:
        result = subprocess.run([sys.executable, "-X", "utf8", *sys.argv], env=env)
    except KeyboardInterrupt:
        raise SystemExit(130)
    raise SystemExit(result.returncode)


_ensure_utf8_mode()

import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# BASE_MODEL_ID = "Qwen/Qwen1.5-0.5B"
# ADAPTER_DIR = "./qwen-0.5b-xquad-finetuned"
MODEL_ID = "google/gemma-3-270m-it"
OUTPUT_DIR = "./gemma-3-270m-xquad-finetuned"

USE_CUDA = torch.cuda.is_available()
USE_BF16 = USE_CUDA and torch.cuda.is_bf16_supported()
USE_FP16 = USE_CUDA and not USE_BF16
MODEL_DTYPE = torch.bfloat16 if USE_BF16 else (torch.float16 if USE_CUDA else torch.float32)

print("1. Loading Tokenizer and Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Load model in standard format (16GB VRAM is plenty for this)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto", 
    dtype=MODEL_DTYPE
)

print("2. Preparing the Training Data...")
# Load the CSV you made in Phase 1
df = pd.read_csv("../xquad_train.csv")

# We need to train the model on BOTH English and Spanish so it learns the format
training_texts =[]
for _, row in df.iterrows():
    # English prompt + answer
    en_text = f"{row['context_en']} Answer in a few words with no explanation, {row['question_en']} {row['answer_en']}{tokenizer.eos_token}"
    # Spanish prompt + answer
    es_text = f"{row['context_es']} Answer in a few words with no explanation, {row['question_es']} {row['answer_es']}{tokenizer.eos_token}"
    
    training_texts.extend([en_text, es_text])

# Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_dict({"text": training_texts})

print("3. Setting up LoRA (Efficient Fine-Tuning)...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

print("4. Starting the Training Process...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,           # 3 passes through the data
    logging_steps=10,
    save_strategy="no",
    optim="adamw_torch",
    fp16=USE_FP16,
    bf16=USE_BF16,
    use_cpu=not USE_CUDA,
    dataset_text_field="text",
    max_length=512,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
)

trainer.train()

print(f"5. Saving Fine-Tuned Model to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("FINE-TUNING COMPLETE!")