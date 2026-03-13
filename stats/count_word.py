import reload as r 
total_pairs = len(r.train_ds) + len(r.val_ds) + len(r.test_ds)
print("Nombre total de paires :", total_pairs)

from datasets import concatenate_datasets

# Fusion des splits
ds = concatenate_datasets([r.train_ds, r.val_ds, r.test_ds])

print(ds)

def count_words_translation(dataset):
    hat_words = 0
    eng_words = 0

    for ex in dataset:
        hat_words += len(ex["translation"]["src_text"].split())
        eng_words += len(ex["translation"]["tgt_text"].split())

    return hat_words, eng_words

hat_train, eng_train = count_words_translation(ds["train"])
hat_val, eng_val     = count_words_translation(ds["validation"])
hat_test, eng_test   = count_words_translation(ds["test"])

total_src_words = hat_train + hat_val + hat_test
total_tgt_words = eng_train + eng_val + eng_test

print("GLOBAL")
print("Mots créole (hat) :", total_src_words)
print("Mots anglais (eng) :", total_tgt_words)