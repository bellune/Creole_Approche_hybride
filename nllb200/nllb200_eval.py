import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# -------------------------------
# Chargement des données
# -------------------------------
path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

ds = load_from_disk(save_path)
print(ds)

test_ds = ds["test"]

# -------------------------------
# Chargement du modèle et tokenizer
# -------------------------------
model_path = "nllb200_baseline/checkpoint-41565"
base_model = "facebook/nllb-200-distilled-600M"

SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    src_lang=SRC_LANG,
    tgt_lang=TGT_LANG,
    use_fast=False
)

# Très important pour NLLB pendant generate()
model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

# --------------------------------
# Filtrage hat -> eng
# --------------------------------
def keep_hat_en(example):
    t = example["translation"]
    return (t["src_lang"] == "hat") and (t["tgt_lang"] == "eng")

test_f = test_ds.filter(keep_hat_en)
print("Test:", len(test_f))

# --------------------------------
# Prétraitement
# --------------------------------
MAX_LEN = 128

def preprocess(examples):
    src_texts = [item["src_text"] for item in examples["translation"]]
    tgt_texts = [item["tgt_text"] for item in examples["translation"]]

    model_inputs = tokenizer(
        src_texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=tgt_texts,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length"
    )["input_ids"]

    pad = tokenizer.pad_token_id
    labels = [[tok if tok != pad else -100 for tok in seq] for seq in labels]

    model_inputs["labels"] = labels
    return model_inputs

tok_test = test_f.map(preprocess, batched=True, remove_columns=["translation"])

print("src_lang:", SRC_LANG)
print("tgt_lang:", TGT_LANG)
print("hat_Latn:", tokenizer.convert_tokens_to_ids(SRC_LANG))
print("eng_Latn:", tokenizer.convert_tokens_to_ids(TGT_LANG))

# -------------------------------
# Métriques
# -------------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")
bleurt = evaluate.load("bleurt", config_name="bleurt-base-128")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # SacreBLEU attend liste de listes
    bleu_labels = [[label] for label in decoded_labels]

    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=bleu_labels
    )

    # chrF
    chrf_result = chrf.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    # TER
    ter_result = ter.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    # BLEURT
    bleurt_result = bleurt.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"],
        "ter": ter_result["score"],
        "bleurt": float(np.mean(bleurt_result["scores"]))
    }

# -------------------------------
# Evaluation
# -------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="eval_results",
    predict_with_generate=True,
    per_device_eval_batch_size=8,
    report_to="none",
    generation_max_length=128
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

results = trainer.evaluate(eval_dataset=tok_test)
print(results)