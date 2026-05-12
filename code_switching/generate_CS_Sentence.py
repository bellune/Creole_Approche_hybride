import os
import time
import pandas as pd
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI
from datasets import load_dataset

# =========================
# FICHIERS
# =========================

TEST_FILE = "datasets/corpus_culturel/code-switching/prcs_cr_en.jsonl"
OUTPUT_FILE = "datasets/corpus_culturel/code-switching/CS_Sentence_cr_en.jsonl"

# =========================
# API
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# LOAD DATASET
# =========================

test_data = load_dataset(
    "json",
    data_files={"test": TEST_FILE}
)["test"]

print(test_data)

# =========================
# RESULTATS
# =========================

results = []

# =========================
# BOUCLE
# =========================

for idx, item in enumerate(test_data):

    src_text = item["translation"]["src_text"]

    prompt = f"""
You are a Haitian Creole-English code-switching generator.

Task:
Generate a natural Haitian Creole-English code-switched sentence using SENTENCE-LEVEL ANNOTATION.

Instructions:
- Keep the original Haitian Creole sentence unchanged.
- Add a natural English annotation in parentheses.
- Preserve the cultural meaning.
- Prefer natural English expressions over literal translation.
- Output only the final annotated text.

Haitian Creole proverb or expression:
"{src_text}"

Annotated version:

"""

    try:
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        prediction = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Erreur ligne {idx}: {e}")
        prediction = ""

    results.append({
    **item,
    "src_text": prediction
   })

    print(f"{idx + 1}/{len(test_data)} terminé")

    time.sleep(0.5)

# =========================
# SAVE JSON
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nFichier sauvegardé :", OUTPUT_FILE)