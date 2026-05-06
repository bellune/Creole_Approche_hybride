import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

INPUT_FILE = "datasets/corpus_culturel/cr_fr.jsonl"

OUTPUT_DIR = Path("datasets/corpus_culturel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================
# Charger le dataset
# ============================

df = pd.read_json(INPUT_FILE, lines=True)

print("Total :", len(df))

# ============================
# Split train / dev
# ============================

train_df, dev_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42,
    shuffle=True
)

print("Train :", len(train_df))
print("Dev   :", len(dev_df))

# ============================
# Sauvegarde
# ============================

train_df.to_json(
    OUTPUT_DIR / "train/cr_fr.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

dev_df.to_json(
    OUTPUT_DIR / "dev/cr_fr.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)

print("\nFichiers sauvegardés.")