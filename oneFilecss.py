from datasets import concatenate_datasets, load_dataset

TRAIN_FILE_CULT_CS = "datasets/corpus_culturel/code-switching/train/cr_CSS_en.jsonl"
TRAIN_FILE_CS = "datasets/corpus_culturel/code-switching/train/cr_codeS_en.jsonl"


# -------------------------------
# Chargement des données
# -------------------------------

dataset_CULT = load_dataset(
    "json",
    data_files={

        "train_cult_CS": TRAIN_FILE_CULT_CS,
        "train_code_switching": TRAIN_FILE_CS
    }
)




print("Train CULT CS:", len(dataset_CULT["train_cult_CS"]))
print("Train CULT CS (50%%):", len(dataset_CULT["train_cult_CS"].shuffle(seed=42).select(range(int(0.50 * len(dataset_CULT["train_cult_CS"]))))))
print("Train Code Switching:", len(dataset_CULT["train_code_switching"]))


train_ds = concatenate_datasets([
    dataset_CULT["train_cult_CS"].shuffle(seed=42).select(range(int(0.50 * len(dataset_CULT["train_cult_CS"])))),
    dataset_CULT["train_code_switching"]
])
    
train_ds = train_ds.shuffle(seed=42)
print("Train:", len(train_ds))

train_ds.to_json(
    "train_CS_full.jsonl",
    orient="records",
    lines=True,
    force_ascii=False
)