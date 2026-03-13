from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import evaluate

import torch, numpy as np, random
from transformers import MT5ForConditionalGeneration, AutoTokenizer

from datasets import load_from_disk


path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

# -------------------------------
# Chargement des données
# -------------------------------


ds = load_from_disk(save_path)
print(ds)

train_ds = ds["train"]
val_ds   = ds["validation"]
test_ds  = ds["test"]

print(train_ds[0])


# -------------------------------
# Chargement du modèle et du tokenizer
# -------------------------------

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Device:", device)


# --------------------------------
# Traitement des données        
# --------------------------------

import reload as r
def keep_hat_en(example):
    t = example["translation"]
    return (t["src_lang"] == "hat") and (t["tgt_lang"] == "eng")

train_f = train_ds.filter(keep_hat_en)
val_f   = val_ds.filter(keep_hat_en)
test_f  = test_ds.filter(keep_hat_en)

print("Train:", len(train_f), "Val:", len(val_f), "Test:", len(test_f))
print(train_f)


MAX_LEN = 128

def preprocess(examples):
    src_texts = [f"translate Haitian Creole to English: {item['src_text']}" for item in examples["translation"]]
    tgt_texts = [item["tgt_text"] for item in examples["translation"]]

    # Pad à longueur fixe pour stabilité
    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        tgt_texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    # Ignorer le padding dans la loss
    pad = tokenizer.pad_token_id
    labels = [[(tok if tok != pad else -100) for tok in seq] for seq in labels]

    model_inputs["labels"] = labels
    return model_inputs

tok_train = train_f.map(preprocess, batched=True, remove_columns=["translation"])
tok_val   = val_f.map(preprocess, batched=True, remove_columns=["translation"])
tok_test  = test_f.map(preprocess, batched=True, remove_columns=["translation"])





# -------------------------------
# entrainement du modele MT5 et evaluattion
# -------------------------------   

# Collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training args
args = Seq2SeqTrainingArguments(
    output_dir="mt5_baseline",
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    learning_rate=1e-4,
    max_grad_norm=1.0,
    warmup_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    predict_with_generate=False,
    fp16=False,
    report_to="none",
    seed=42
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# -------------------------
# 1) Sélection test
# -------------------------
k = 100
test_sample = test_f.select(range(min(k, len(test_f))))
translations = test_sample["translation"]

src_texts = [f"translate Haitian Creole to English: {ex['src_text']}" for ex in translations]
refs = [ex["tgt_text"] for ex in translations]

# -------------------------
# 2) Génération
# -------------------------
preds = []

for text in src_texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4
        )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    preds.append(prediction)

# -------------------------
# 3) BLEU
# -------------------------
bleu = evaluate.load("sacrebleu")
bleu_score = bleu.compute(
    predictions=preds,
    references=[[r] for r in refs]
)

print("BLEU score:", bleu_score["score"])

# -------------------------
# 4) chrF
# -------------------------
chrf = evaluate.load("chrf")
chrf_score = chrf.compute(
    predictions=preds,
    references=refs
)

print("chrF score:", chrf_score["score"])

# -------------------------
# 5) Exemples
# -------------------------
for i in range(min(10, len(test_sample))):
    print("\n---", i, "---")
    print("SRC :", test_sample[i]["translation"]["src_text"])
    print("PRED:", preds[i])
    print("REF :", test_sample[i]["translation"]["tgt_text"])