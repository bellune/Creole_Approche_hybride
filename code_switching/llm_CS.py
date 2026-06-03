import os
import time
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI
from datasets import load_dataset

# =========================
# FICHIERS
# =========================

TEST_FILE = "datasets/corpus_culturel/code-switching/test/cr_CSS_en.jsonl"
OUTPUT_FILE = "result/Code_switching_cultural_with_GPT_predictions.csv"


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
 You are a Haitian Creole translator familiar with Haitian culture.

Translate the following Haitian Creole expression into natural English according to its Haitian cultural context.

Only provide the English translation.

Haitian Creole:
{src_text}

English:

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
    encoding="utf-8-sig"
)

print("\nFichier sauvegardé :", OUTPUT_FILE)