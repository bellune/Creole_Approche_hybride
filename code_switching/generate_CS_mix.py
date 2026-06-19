import os
import time
from pathlib import Path
import pandas as pd
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI
from datasets import load_dataset

# =========================
# FICHIERS
# =========================

OUTPUT_DIR =  Path("datasets/mix_corpus/code-switching")

FILES = ["datasets/mix_corpus/json/train_mix.jsonl", 
         "datasets/mix_corpus/json/valid_mix.jsonl"]

OUTPUT_FILES = [f"{OUTPUT_DIR}/train/cr_CSS_en.jsonl",
                 f"{OUTPUT_DIR}/dev/cr_CSS_en.jsonl"]


# =========================
# API
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

for split in ["train", "dev"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


for FILE, OUTPUT_FILE in zip(FILES, OUTPUT_FILES):  



    # =========================
    # LOAD DATASET
    # =========================

    test_data = load_dataset(
        "json",
        data_files={"test": FILE}
    )["test"]

    print(test_data)

#    selectionner 50% des données pour accélérer le processus
    test_data = test_data.shuffle(seed=42).select(range(len(test_data) // 2))
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