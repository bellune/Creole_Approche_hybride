import os
import time
import pandas as pd

from openai import OpenAI
from datasets import load_dataset

# =========================
# FICHIERS
# =========================

TEST_FILE = "datasets/corpus_culturel/test/cr_en.jsonl"
OUTPUT_FILE = "result/test_cr_en_with_GPT_predictions.csv"

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
    reference = item["translation"]["tgt_text"]

    prompt = f"""
Translate the following Haitian Creole text into English.

Haitian Creole text:
{src_text}

English:
"""

    try:

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            temperature=0
        )

        prediction = response.output_text.strip()

    except Exception as e:

        print(f"Erreur ligne {idx}: {e}")
        prediction = ""

    results.append({
        "src_text": src_text,
        "reference": reference,
        "prompt": prompt,
        "prediction": prediction
    })

    print(f"{idx + 1}/{len(test_data)} terminé")

    time.sleep(0.5)

# =========================
# SAVE CSV
# =========================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\nFichier sauvegardé :", OUTPUT_FILE)