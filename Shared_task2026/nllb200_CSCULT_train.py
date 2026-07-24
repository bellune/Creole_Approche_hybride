from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, NllbTokenizer, EarlyStoppingCallback
from datasets import concatenate_datasets, load_dataset


from datasets import load_from_disk


path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

TRAIN_FILE = "datasets/corpus_culturel/code-switching/train/cr_codeS_en.jsonl"
TRAIN_FILE_CULT = "datasets/corpus_culturel/train/cr_en.jsonl"

DEV_FILE_CULT = "datasets/corpus_culturel/dev/cr_en.jsonl"
OUTPUT_DIR = "model/nllb_CSCULT"

# -------------------------------
# Chargement des données
# -------------------------------

dataset_CS = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE
    }
)

dataset_CULT = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE_CULT,
        "validation": DEV_FILE_CULT
    }
)

ds = load_from_disk(save_path)
print(ds)


train_ds = concatenate_datasets([
    ds["train"],
    dataset_CS["train"],
    dataset_CULT["train"]
])
    
train_ds = train_ds.shuffle(seed=42)
print("Train:", len(train_ds))


val_ds = concatenate_datasets([ds["validation"], dataset_CULT["validation"]])
val_ds = val_ds.shuffle(seed=42)

print("Val:", len(val_ds))

test_ds  = ds["test"]




# -------------------------------
# Chargement du modèle et du tokenizer
# -------------------------------

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = NllbTokenizer.from_pretrained(
    model_name,
    src_lang="hat_Latn",
    tgt_lang="eng_Latn",
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.generation_config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")


# --------------------------------
# Traitement des données
# --------------------------------

def keep_hat_en(example):
    t = example["translation"]
    return (t["src_lang"] == "hat") and (t["tgt_lang"] == "eng") or (t["src_lang"] == "hat_Latn") and (t["tgt_lang"] == "eng_Latn")

train_f = train_ds.filter(keep_hat_en)
val_f   = val_ds.filter(keep_hat_en)
test_f  = test_ds.filter(keep_hat_en)

print("Train:", len(train_f), "Val:", len(val_f), "Test:", len(test_f))
print(train_f)


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

tok_train = train_f.map(preprocess, batched=True, remove_columns=["translation"])
tok_val   = val_f.map(preprocess, batched=True, remove_columns=["translation"])
tok_test  = test_f.map(preprocess, batched=True, remove_columns=["translation"])


print("src_lang:", tokenizer.src_lang)
print("tgt_lang:", tokenizer.tgt_lang)
print("hat_Latn:", tokenizer.convert_tokens_to_ids("hat_Latn"))
print("eng_Latn:", tokenizer.convert_tokens_to_ids("eng_Latn"))


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


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="steps",
    eval_steps=5000,
    save_strategy="steps",
    save_steps=10000,
    logging_steps=200,

    learning_rate=4e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,

    num_train_epochs=3,

    predict_with_generate=True,
    generation_max_length=128,
    generation_num_beams=4,

    fp16=True,
    save_total_limit=1,
    save_only_model=True,

    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True,

    report_to="none"
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tok_train,
    eval_dataset=tok_val,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
       callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.05
        )
    ]
)

trainer.train()