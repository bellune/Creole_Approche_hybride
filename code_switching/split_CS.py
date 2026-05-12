from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
BASE_FILE = Path("datasets/corpus_culturel/code-switching/final_CS_cr_en.jsonl")
OUTPUT_DIR = Path("datasets/corpus_culturel/code-switching")

def split_train_dev_test(df, name):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    print(f"Taille totale : {len(df)}")
    print(f"Train : {len(train_df)}")
    print(f"Dev   : {len(dev_df)}")
    print(f"Test  : {len(test_df)}")

    for split in ["train", "dev", "test"]:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)

    train_df.to_json(OUTPUT_DIR / "train" / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    dev_df.to_json(OUTPUT_DIR / "dev" / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
    test_df.to_json(OUTPUT_DIR / "test" / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)

datas = load_dataset(
    "json",
    data_files={"data": str(BASE_FILE)}
)["data"]

df = datas.to_pandas()

split_train_dev_test(df, "cr_en")
