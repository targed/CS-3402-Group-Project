import subprocess
import csv
import os

# *** Requires the GitHub Copilot CLI to be installed and authenticated *** #
# 1190 total questions
# gpt-5 model used

# use subprocess to call the copilot CLI
def ask_copilot(prompt):
    exe = "EXE_PATH"  # *** replace with the actual path to your copilot CLI executable *** #
    cmd = [exe, "-p", f"Answer in as few words as possible: {prompt}"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.stdout

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'copilot_test.csv'), encoding="utf-8") as file:
        with open(os.path.join(script_dir, 'copilot_results.txt'), 'w') as output_file:
            reader = csv.reader(file, delimiter=',')
            header = next(reader) # skip the header row

            for row in reader:
                print("Processing ID: " + row[0])
                id = row[0]
                context_en = row[1]
                context_es = row[4]
                prompt_en = row[7]
                prompt_es = row[8]

                answer_en = ask_copilot(prompt_en + context_en)
                answer_es = ask_copilot(prompt_es + context_es)

                output_file.write("ID: " + id + "\n")
                output_file.write("Answer (EN): " + answer_en + "\n")
                output_file.write("Answer (ES): " + answer_es + "\n")
                print("Done\n")

    print("### ALL DONE ###")

main()