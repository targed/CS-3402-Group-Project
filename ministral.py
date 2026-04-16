#!/share/ceph/scratch/bw4d4/venv/bin/python
import os, json, sys
from tqdm import tqdm

# ./ministral.py mistralai/Ministral-3-14B-Base-2512 Ministral-3-14B-Base-2512
# ./ministral.py mistralai/Ministral-3-8B-Base-2512 Ministral-3-8B-Base-2512
# ./ministral.py mistralai/Ministral-3-3B-Base-2512 Ministral-3-3B-Base-2512

print("[USAGE] ministral.py model output_folder")
model_name = sys.argv[1]
output_folder = sys.argv[2]

os.environ["HF_HOME"] = "/share/ceph/scratch/bw4d4/huggingface"
os.environ["HF_TOKEN"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print("Importing")
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    FineGrainedFP8Config,
)
import json

print("Loading model")
model = Mistral3ForConditionalGeneration.from_pretrained(
    model_name, device_map="auto", trust_remote_code=True
)
print("Loading tokenizer")
tokenizer = MistralCommonBackend.from_pretrained(model_name, trust_remote_code=True)
for language_file in ["en", "es"]:
    prompt = (
        " Answer in a few words with no explanation, "
        if language_file == "en"
        else " Responde con pocas palabras, sin explicaciones, "
    )
    print("Loading data")
    with open(f"xquad.{language_file}.json", "r") as f:
        language = json.load(f)
    print("Generating predictions")
    total_questions = sum(
        (
            sum((len(paragraph["qas"]) for paragraph in data["paragraphs"]))
            for data in language["data"]
        )
    )
    progress = tqdm(
        desc=language_file, total=total_questions, unit="q", unit_scale=True
    )
    output_i = 0
    for data in language["data"]:
        progress.set_description(f"{output_folder}/{language_file}{output_i}.json")
        output_json = {}
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                input_ids = tokenizer.encode(
                    context + prompt + question["question"], return_tensors="pt"
                ).cuda()
                output = model.generate(
                    input_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    max_length=input_ids.shape[1] + 20,
                )
                output_json[question["id"]] = tokenizer.decode(
                    output[0][input_ids.shape[1] :], skip_special_tokens=True
                )
                progress.update(1)
        with open(f"{output_folder}/{language_file}{output_i}.json", "w") as f:
            json.dump(output_json, f)
        output_i += 1
    progress.close()
