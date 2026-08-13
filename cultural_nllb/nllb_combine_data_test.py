

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from datasets import load_from_disk


TEST_FILE = "datasets/corpus_culturel/test/cr_en.jsonl"


path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

ds = load_from_disk(save_path)
print(ds)

datasetest = load_dataset(
    "json",
    data_files={
        "test": TEST_FILE    }
)




# Dans les tests general ajoute un id aleatoire, exemple : id="gen_00001"
ds["test"] = ds["test"].map(
    lambda x, idx: {**x, "id": f"gen_{idx+1:05d}"},
    with_indices=True
)

# concatenate the test dataset with the existing test split

test_ds = concatenate_datasets([
    ds["test"],
    datasetest["test"],
])
    
test_ds = test_ds.shuffle(seed=42)
print("Test:", len(test_ds))

# ecrire le test_ds dans un fichier jsonl
test_ds.to_json("datasets/corpus_culturel/test/cr_en_all.jsonl", orient="records", lines=True, force_ascii=False)