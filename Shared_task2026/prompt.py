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

# TEST_FILES = ["uqam_eval_srcs/hat_eng.jsonl", "uqam_eval_srcs/hat_fra.jsonl"]
# OUTPUT_FILES = ["uqam_eval_srcs/hat_eng_explain.csv", "uqam_eval_srcs/hat_fra_explain.csv"]
TEST_FILES = [ "uqam_eval_srcs/hat_fra.jsonl"]
OUTPUT_FILES = [ "uqam_eval_srcs/hat_fra_explain.csv"]



# =========================
# API
# =========================

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


for TEST_FILE, OUTPUT_FILE in zip(TEST_FILES, OUTPUT_FILES):
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
        id = idx

        prompt = f"""

        Haitian Creole expression:
    {src_text}

    Reference French translation:
        {reference}

    Provide a short cultural french explanation of this expression in one sentence based on its Reference French translation.
    don't provide a literal translation, but rather an explanation of the cultural meaning behind the expression. 
    ONLY provide the cultural explanation, without any additional information or context.

    """

        prompt2 = f"""

            Haitian Creole expression:
        {src_text}

        Reference French translation:
            {reference}

        Provide two short cultural expected French translation of this expression based on its Reference French translation.
        don't provide a literal translation, but rather an expected translation of the expression based on its cultural meaning. 
        ONLY provide the expected translation, without any additional information or context.
        separate the two expected translations with a separator "|" .

        """


        try:
            response = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            explanation = response.choices[0].message.content.strip()

            response = client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=[
                    {"role": "user", "content": prompt2}
                ],
                temperature=0
            )

            expected_translation = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Erreur ligne {idx}: {e}")
            explanation = ""
            expected_translation = ""

        results.append({
            "id" : id,
            "src_text": src_text,
            "reference": reference,
            "explanation": explanation,
            "expected_translation": expected_translation
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