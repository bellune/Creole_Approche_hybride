import os
import time
from pathlib import Path
import pandas as pd
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

from openai import OpenAI
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk

# =========================
# FICHIERS
# =========================

OUTPUT_DIR =  Path("datasets/code-switching/FR_MT_CS")

# FILES = ["datasets/corpus_culturel/train/cr_en.jsonl", 
#          "datasets/corpus_culturel/dev/cr_en.jsonl",
#            "datasets/corpus_culturel/test/cr_en.jsonl" ]

path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-fra"

OUTPUT_FILES = [
    f"{OUTPUT_DIR}/train/cr_codeS_fra.jsonl"
                #  f"{OUTPUT_DIR}/dev/cr_codeS_fra.jsonl",
                #    f"{OUTPUT_DIR}/test/cr_codeS_fra.jsonl" 
                   ]



# -------------------------------
# Chargement des données
# -------------------------------

ds = load_from_disk(save_path)
print(ds)

train_ds = ds["train"]
val_ds   = ds["validation"]
test_ds  = ds["test"]

print(train_ds[0])

# --------------------------------
# Get 15% of the data for each split
# --------------------------------

train_ds = train_ds.shuffle(seed=42).select(range(int(0.10 * len(train_ds))))
# val_ds   = val_ds.shuffle(seed=42).select(range(int(0.15 * len(val_ds))))
# test_ds  = test_ds.shuffle(seed=42).select(range(int(0.15 * len(test_ds))))

# =========================
# API
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# for split in ["train", "dev", "test"]:
#     (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


for data, OUTPUT_FILE in zip([train_ds], OUTPUT_FILES):  

    # =========================
    # RESULTATS
    # =========================

    results = []

    # =========================
    # BOUCLE
    # =========================
    
    for idx, item in enumerate(data):

        src_text = item["translation"]["src_text"]

        prompt = f"""
    You are a Haitian Creole-French code-switching generator.

    Task:
    Generate a natural Haitian Creole-French code-switched sentence using TOKEN-LEVEL REPLACEMENT.

    Instructions:
    - Keep the sentence mostly in Haitian Creole.
    - Replace only few words depending on the length of the sentence with French.
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

  
             
        item["translation"]["src_text"] = prediction
        results.append(item)

        print(f"{idx + 1}/{len(data)} terminé")

        time.sleep(0.5)

    #--------------------------------
    # combine results with original dataset
    #--------------------------------
    # results_dataset = Dataset.from_list(results)
    
  

    # =========================
    # SAVE JSON
    # =========================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nFichier sauvegardé :", OUTPUT_FILE)