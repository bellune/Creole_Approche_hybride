from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

# ============================
# 1. Chemins
# ============================

BASE_MODEL = "facebook/nllb-200-distilled-600M"

CHECKPOINT_PATH = "nllb200_baseline4/checkpoint-166260"  # à modifier

TRAIN_FILE = "corpus_culturel/train/cr_en.jsonl"
DEV_FILE = "corpus_culturel/dev/cr_en.jsonl"

OUTPUT_DIR = "models/nllb_cultural_cr_en"


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
    src_texts = [
        item["src_text"] for item in batch["translation"]
    ]

    tgt_texts = [
        item["tgt_text"] for item in batch["translation"]
    ]

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

    inputs["labels"] = labels["input_ids"]

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

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    learning_rate=3e-5,
    num_train_epochs=5,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=50,

    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,

    fp16=True,

    save_total_limit=2
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
    tokenizer=tokenizer
)


# ============================
# 8. Entraîner
# ============================

trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Modèle culturel sauvegardé dans :", OUTPUT_DIR)