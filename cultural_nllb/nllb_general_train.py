from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import evaluate
import numpy as np

# ============================
# 1. Chemins
# ============================

BASE_MODEL = "facebook/nllb-200-distilled-600M"

CHECKPOINT_PATH = "nllb200_baseline4/checkpoint-166260"  # à modifier


TRAIN_FILE = "datasets/corpus_culturel/train/cr_en.jsonl"
DEV_FILE = "datasets/corpus_culturel/dev/cr_en.jsonl"
TRAIN_FILE_fr = "datasets/corpus_culturel/train/cr_fr.jsonl"
DEV_FILE_fr = "datasets/corpus_culturel/dev/cr_fr.jsonl"


OUTPUT_DIR = "model/nllb_cultural"
path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

# ============================
# 1. Charger CR-EN
# ============================

cr_en_train = load_dataset(
    "json",
    data_files=TRAIN_FILE
)["train"]

cr_en_dev = load_dataset(
    "json",
    data_files=DEV_FILE
)["train"]


# ============================
# 2. Charger CR-FR
# ============================

cr_fr_train = load_dataset(
    "json",
    data_files=TRAIN_FILE_fr
)["train"]

cr_fr_dev = load_dataset(
    "json",
    data_files=DEV_FILE_fr
)["train"]



# -------------------------------
# Chargement des données
# -------------------------------


ds = load_from_disk(save_path)
print(ds)

train_ds = ds["train"]
val_ds   = ds["validation"]

print(train_ds[0])


# ============================
# 3. Combiner
# ============================

train = concatenate_datasets([
    cr_en_train,
    cr_fr_train,
    train_ds
])

validation = concatenate_datasets([
    cr_en_dev,
    cr_fr_dev,
    val_ds
])

dataset = DatasetDict({
    "train": train,
    "validation": validation
})

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
# 6. Metrics
# ============================

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
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,

    num_train_epochs=4,

    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,

    fp16=True,
    save_total_limit=2,

    load_best_model_at_end=True,
    metric_for_best_model="blue",
    greater_is_better=True,

    report_to="none"
   
)


# ============================
# 7. Trainer
# ============================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,

    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],

    data_collator=data_collator,
    compute_metrics=compute_metrics
)


# ============================
# 8. Entraîner
# ============================

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Modèle culturel sauvegardé dans :", OUTPUT_DIR)