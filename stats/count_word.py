from datasets import load_from_disk
from datasets import concatenate_datasets
import re

# -------------------------------
# Chargement des données
# -------------------------------
path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

ds = load_from_disk(save_path)
print(ds)

test_ds = ds["test"]
train_ds = ds["train"]
val_ds   = ds["validation"]


total_pairs = len(train_ds) + len(val_ds) + len(test_ds)
print("Nombre total de paires :", total_pairs)

# Fusion des splits
all_ds = concatenate_datasets([train_ds, val_ds, test_ds])

print(ds)

def count_words_translation(dataset):
    hat_words = 0
    eng_words = 0

    for ex in dataset:
        hat_words += len(re.findall(r"[a-zA-ZÀ-ÿ]+", ex["translation"]["src_text"]))
        eng_words += len(re.findall(r"[a-zA-ZÀ-ÿ]+", ex["translation"]["tgt_text"]))

        # for i in range(5):
        #  if i == 5:
        #   break 
        #  print(ex["translation"]["src_text"].split())

    return hat_words, eng_words

hat_train, eng_train = count_words_translation(ds["train"])
hat_val, eng_val     = count_words_translation(ds["validation"])
hat_test, eng_test   = count_words_translation(ds["test"])


total_train_ht = hat_train 
total_train_en = eng_train
total_val_ht   = hat_val
total_val_en   =  eng_val
total_test_ht  = hat_test
total_test_en  = eng_test

total_src_words = hat_train + hat_val + hat_test
total_tgt_words = eng_train + eng_val + eng_test

print("GLOBAL")
print("Total Train HT :", total_train_ht)
print("Total Val  HT :", total_val_ht)
print("Total Test HT :", total_test_ht)

print("-----------------------------------")

print("Total Train EN:", total_train_en)
print("Total Val  EN :", total_val_en)
print("Total Test EN :", total_test_en)

print("-----------------------------------")

print("Mots créole (hat) :", total_src_words)
print("Mots anglais (eng) :", total_tgt_words)



