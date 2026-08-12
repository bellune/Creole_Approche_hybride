from pathlib import Path
from datasets import concatenate_datasets, load_from_disk, load_dataset


path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"
data_fairseq = save_path + "/mt-fairseq"

TEST_FILE  = "datasets/corpus_culturel/test/cr_en.jsonl"

# -------------------------------
# Chargement des données
# -------------------------------


ds = load_from_disk(save_path)
print(ds)

train_ds = ds["train"]
val_ds   = ds["validation"]
test_ds  = ds["test"]


dataset = load_dataset(
    "json",
    data_files={
        "test": TEST_FILE
    }
)

print(dataset["test"][0])

# concatenate the test dataset with the existing test split

test_ds = concatenate_datasets([
    dataset["test"],
    ds["test"]
])


SRC_LANG = "ht"
TGT_LANG = "en"



OUT_DIR = Path(data_fairseq)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_translation_split_for_fairseq(dataset, output_name):
    src_path = OUT_DIR / f"{output_name}.{SRC_LANG}"
    tgt_path = OUT_DIR / f"{output_name}.{TGT_LANG}"

    seen = set()
    count = 0

    with open(src_path, "w", encoding="utf-8") as src_file, \
         open(tgt_path, "w", encoding="utf-8") as tgt_file:

        for example in dataset :
        #   print("Example:", example["translation"])
           
          src = str(example["translation"]["src_text"]).strip()
          tgt = str(example["translation"]["tgt_text"]).strip()
        

          if not src or not tgt:
                continue

          pair = (src, tgt)
          if pair in seen:
                continue
          seen.add(pair)

          src_file.write(src.replace("\n", " ") + "\n")
          tgt_file.write(tgt.replace("\n", " ") + "\n")

          count += 1

          if count % 1000 == 0:
            print(f"{output_name}: {count} paires déjà traitées")

    print(f"{output_name}: terminé avec {count} paires enregistrées")

save_translation_split_for_fairseq(train_ds, "train")
save_translation_split_for_fairseq(val_ds, "valid")
save_translation_split_for_fairseq(test_ds, "test")