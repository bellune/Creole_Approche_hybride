from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate
import numpy as np

# ============================
# 1. Chemins
# ============================

BASE_MODEL = "facebook/nllb-200-distilled-600M"

CHECKPOINT_PATH = "backup_model/nllb_cultural_cr_en/checkpoint-7980"  # à modifier

TRAIN_FILE = "datasets/mix_corpus/json/train_mix.jsonl"
DEV_FILE = "datasets/mix_corpus/json/valid_mix.jsonl"

OUTPUT_DIR = "/root/model/nllb_cultural_cr_en"

# /root/model/nllb200Baseline


# ============================
# 2. Charger dataset
# ============================

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": DEV_FILE
    }
)

print(dataset)


# ============================
# 3. Charger tokenizer + modèle
# ============================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT_PATH)

SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"

tokenizer.src_lang = SRC_LANG
forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

model.generation_config.forced_bos_token_id = forced_bos_token_id


# ============================
# 4. Prétraitement
# ============================

MAX_LEN = 128

def preprocess(batch):
    src_texts = [item["src_text"] for item in batch["translation"]]
    tgt_texts = [item["tgt_text"] for item in batch["translation"]]

    tokenizer.src_lang = SRC_LANG

    inputs = tokenizer(
        src_texts,
        max_length=MAX_LEN,
        truncation=True
    )

    labels = tokenizer(
        tgt_texts,
        max_length=MAX_LEN,
        truncation=True
    )

    labels_ids = labels["input_ids"]

    labels_ids = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in labels_ids
    ]

    inputs["labels"] = labels_ids

    return inputs


tokenized_dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=dataset["train"].column_names
)


# ============================
# 5. Data collator
# ============================

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)


# ============================
# 6. Arguments d'entraînement
# ============================

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

   eval_strategy="steps",
     eval_steps=1000,
     save_strategy="steps",
     save_steps=1000,
     logging_steps=200,
 
     learning_rate=4e-5,
    # A100 80 GB : exploiter davantage le GPU
     per_device_train_batch_size=32,
     per_device_eval_batch_size=32,
     gradient_accumulation_steps=1,
     weight_decay=0.01,
 
     num_train_epochs=20,
 
     predict_with_generate=True,
     generation_max_length=128,
     generation_num_beams=4,
 
     # accélération A100
     bf16=True,
     fp16=False,
     tf32=True,
     save_total_limit=2,
 
     load_best_model_at_end=True,
     metric_for_best_model="bleu",
     greater_is_better=True,
 
     report_to="none"
)


# -------------------------------
# Training et evaluation
# -------------------------------

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Pour SacreBLEU : liste de listes
    bleu_labels = [[label] for label in decoded_labels]

    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=bleu_labels
    )

    # Pour chrF : liste simple
    chrf_result = chrf.compute(
        predictions=decoded_preds,
        references=decoded_labels
    )

    
    ter_result = ter.compute(
    predictions=decoded_preds,
    references=decoded_labels
   )
    

    return {
        "bleu": bleu_result["score"],
        "chrf": chrf_result["score"],
        "ter": ter_result["score"]  
    }



# ============================
# 7. Trainer
# ============================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],

    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=5,
                    early_stopping_threshold=0.05
                )
            ]
)


# ============================
# 8. Entraîner
# ============================

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Modèle culturel sauvegardé dans :", OUTPUT_DIR)