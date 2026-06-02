from pathlib import Path
from openai import OpenAI
import time

# ======================
# Paths
# ======================

input_path = Path("datasets/corpus_AHL/filtered/ALH_filtered_semantic.txt")

output_dir = Path("datasets/corpus_AHL/final")
output_dir.mkdir(parents=True, exist_ok=True)

cr_output_path = output_dir / "ahl_llm_cr.txt"
en_output_path = output_dir / "ahl_llm_en.txt"

# ======================
# Load Creole sentences
# ======================

with open(input_path, "r", encoding="utf-8") as f:
    creole_sentences = [line.strip() for line in f if line.strip()]

print("Total sentences:", len(creole_sentences))

# Test first on 100 sentences
# creole_sentences = creole_sentences[:100]

# ======================
# LLM client
# ======================

client = OpenAI()

MODEL_NAME = "gpt-5.4-mini"

def translate_creole_to_english(sentence):
    prompt = f"""
Translate the following Haitian Creole sentence into English.

Rules:
- Preserve the meaning as closely as possible.
- Do not add information.
- Do not explain.
- Return only the English translation.

Haitian Creole:
{sentence}

English:
""".strip()

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()

# ======================
# Translate
# ======================

with open(cr_output_path, "w", encoding="utf-8") as cr_file, \
     open(en_output_path, "w", encoding="utf-8") as en_file:

    for i, cr_text in enumerate(creole_sentences, start=1):
        try:
            en_text = translate_creole_to_english(cr_text)
        except Exception as e:
            print(f"Error at line {i}: {e}")
            en_text = ""
            time.sleep(2)

        if cr_text and en_text:
            cr_file.write(cr_text + "\n")
            en_file.write(en_text + "\n")

        if i % 10 == 0:
            print(f"Translated {i}/{len(creole_sentences)}")

print("Saved:", cr_output_path)
print("Saved:", en_output_path)