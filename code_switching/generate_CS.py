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

FILES = ["datasets/corpus_culturel/train/cr_en.jsonl", 
         "datasets/corpus_culturel/dev/cr_en.jsonl",
           "datasets/corpus_culturel/test/cr_en.jsonl" ]

OUTPUT_FILES = ["datasets/corpus_culturel/code-switching/train/cr_cs_en.jsonl",
                 "datasets/corpus_culturel/code-switching/dev/cr_cs_en.jsonl",
                   "datasets/corpus_culturel/code-switching/test/cr_cs_en.jsonl"]


# =========================
# API
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

for FILE, OUTPUT_FILE in zip(FILES, OUTPUT_FILES):  



    # =========================
    # LOAD DATASET
    # =========================

    test_data = load_dataset(
        "json",
        data_files={"test": FILE}
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
    Generate a natural Haitian Creole-English code-switched sentence using TOKEN-LEVEL REPLACEMENT.

    Instructions:
    - Keep the sentence mostly in Haitian Creole.
    - Replace only one to three words depending on the length of the sentence with English.
    - Do NOT translate the whole sentence.
    - Keep the sentence fluent and natural.
    - Preserve the original meaning.
    - Output only the code-switched sentence.

    Haitian Creole sentence:
    "{src_text}"

    Code-switched sentence:
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