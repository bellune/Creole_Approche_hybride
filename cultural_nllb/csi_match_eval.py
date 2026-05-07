import pandas as pd

TEST_FILE = "datasets/corpus_culturel/test/cr_en.jsonl"
OUTPUT_FILE = "datasets/corpus_culturel/cultural_annotation_test.csv"

df = pd.read_json(TEST_FILE, lines=True, encoding="utf-8")

df["cr"] = df["translation"].apply(lambda x: x["src_text"])
df["reference_en"] = df["translation"].apply(lambda x: x["tgt_text"])
df["source_type"] = df["source"].fillna("").apply(
    lambda x:
        "poem"
        if any(word in x.lower() for word in ["ife", "fanm"])
        else (
            "proverb"
            if "proverb" in x.lower()
            else "recit"
        )
)


annotation_df = df[["cr", "reference_en", "source_type"]].copy()

annotation_df["cultural_item"] = annotation_df["cr"]
annotation_df["expected_translations"] = ""
annotation_df["explanation"] = ""
annotation_df["is_cultural"] = "yes"


annotation_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Fichier créé :", OUTPUT_FILE)
print("Nombre total de phrases :", len(annotation_df))