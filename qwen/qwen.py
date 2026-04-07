#!/share/ceph/scratch/bw4d4/qwen/venv/bin/python
import os, json, time
os.environ["HF_HOME"] = "/share/ceph/scratch/bw4d4/huggingface"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
print("Importing")
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
print("Loading model")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-72B", device_map="auto", trust_remote_code=True)
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-72B", trust_remote_code=True)
print("Loading data")
input_tokens = {}
question_i = 0
for language_file in ["en", "es"]:
    input_tokens[language_file] = {}
    with open(f"xquad.{language_file}.json", "r") as f:
        language = json.load(f)
    output_i = 0
    for data in language["data"]:
        output_json = {}
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            for question in paragraph["qas"]:
                input_ids = tokenizer(context + " Answer in a few words with no explanation, " + question["question"], return_tensors="pt").input_ids.cuda()
                output = model.generate(input_ids, pad_token_id=tokenizer.pad_token_id, max_new_tokens=20)
                output_json[question["id"]] = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
                print(f"[{time.strftime("%Y%m%dT%H%M%S")}] {output_i}.json question {question_i} {question["id"]}")
                question_i += 1
        with open(f"{language_file}{output_i}.json", "w") as f:
            json.dump(output_json, f)
        output_i += 1
