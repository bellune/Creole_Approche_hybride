from transformers import AutoTokenizer
from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

def count_tokens(dataset, tokenizer):
    total_src = 0
    total_tgt = 0

    for example in dataset:
        src = example["src_text"]
        tgt = example["tgt_text"]

        total_src += len(tokenizer.tokenize(src))
        total_tgt += len(tokenizer.tokenize(tgt))

    return total_src, total_tgt


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

hat_train_tokens, eng_train_tokens = count_tokens(train_ds, tokenizer)
hat_val_tokens, eng_val_tokens     = count_tokens(val_ds, tokenizer)
hat_test_tokens, eng_test_tokens   = count_tokens(test_ds, tokenizer)

print("Tokens Train HT :", hat_train_tokens)
print("Tokens Val  HT :", hat_val_tokens)
print("Tokens Test HT :", hat_test_tokens)

print("----------------------------------")

print("Tokens Train EN :", eng_train_tokens)
print("Tokens Val  EN :", eng_val_tokens)
print("Tokens Test EN :", eng_test_tokens)