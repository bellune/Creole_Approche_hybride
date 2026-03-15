from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
import torch
import evaluate
import numpy as np
import random

# -------------------------------
# Chargement des données
# -------------------------------
path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

ds = load_from_disk(save_path)
print(ds)

train_ds = ds["train"]
val_ds = ds["validation"]
test_ds = ds["test"]

print(train_ds[0])

# -------------------------------
# Filtrage hat -> eng
# -------------------------------
def keep_hat_en(example):
    t = example["translation"]
    return (t["src_lang"] == "hat") and (t["tgt_lang"] == "eng")

train_f = train_ds.filter(keep_hat_en)
val_f = val_ds.filter(keep_hat_en)
test_f = test_ds.filter(keep_hat_en)

print("Train:", len(train_f), "Val:", len(val_f), "Test:", len(test_f))
print(train_f)

# -------------------------------
# Seed
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------------------
# Modèle et tokenizer
# -------------------------------
model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("Device:", device)

# -------------------------------
# Prétraitement
# -------------------------------
MAX_SOURCE_LEN = 128
MAX_TARGET_LEN = 128
PREFIX = "translate Haitian Creole to English: "

def preprocess(examples):
    src_texts = [PREFIX + item["src_text"] for item in examples["translation"]]
    tgt_texts = [item["tgt_text"] for item in examples["translation"]]

    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_SOURCE_LEN,
        truncation=True,
    )

    labels = tokenizer(
        text_target=tgt_texts,
        max_length=MAX_TARGET_LEN,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tok_train = train_f.map(preprocess, batched=True, remove_columns=["translation"])
tok_val = val_f.map(preprocess, batched=True, remove_columns=["translation"])
tok_test = test_f.map(preprocess, batched=True, remove_columns=["translation"])

# -------------------------------
# Métriques
# -------------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    bleu_score = bleu.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels]
    )["score"]

    chrf_score = chrf.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )["score"]

    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
    }

# -------------------------------
# Collator
# -------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# -------------------------------
# Arguments d'entraînement
# -------------------------------
args = Seq2SeqTrainingArguments(
    output_dir="mt5_baseline",
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    logging_steps=200,
    learning_rate=3e-5,
    max_grad_norm=1.0,
    warmup_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,
    save_total_limit=2,
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# -------------------------------
# Entraînement
# -------------------------------
trainer.train()

# -------------------------------
# Évaluation finale sur test
# -------------------------------
test_results = trainer.evaluate(eval_dataset=tok_test)
print("Test results:", test_results)

# -------------------------------
# Exemples de génération
# -------------------------------
k = 10
test_sample = test_f.select(range(min(k, len(test_f))))
translations = test_sample["translation"]

src_texts = [PREFIX + ex["src_text"] for ex in translations]
refs = [ex["tgt_text"] for ex in translations]

preds = []
for text in src_texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SOURCE_LEN
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4
        )

    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    preds.append(pred)

for i in range(len(preds)):
    print(f"\n--- {i} ---")
    print("SRC :", test_sample[i]["translation"]["src_text"])
    print("PRED:", preds[i])
    print("REF :", test_sample[i]["translation"]["tgt_text"])