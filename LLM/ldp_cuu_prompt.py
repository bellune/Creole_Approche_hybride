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

TEST_FILE = "datasets/corpus_culturel/cr_en_with_explain.csv"
OUTPUT_FILE = "result/ldp_cult_with_GPT_predictions.csv"

# =========================
# API
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# LOAD CSV
# =========================

test_data = pd.read_csv(TEST_FILE)

print(test_data.head())
print(test_data.columns)


# =========================
# RESULTATS
# =========================

results = []

# =========================
# BOUCLE
# =========================

for idx, item in enumerate(test_data):

    src_text = item["src_text"]
    reference = item["reference"]
    cultural_context = item["explanation"]

    prompt = f"""
 French: On ne souffre pas de ce qu’on ignore.
English: Far from the eyes, far from the heart.

Spanish: Buenos dias
English: Good morning

 Igbo: Ịmụ igwe
 English: Machine learning
 
 Cultural context: {cultural_context}

 Haitian Creole: {src_text}
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