import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
model_path = "nllb200_baseline2/checkpoint-41565"
base_model = "facebook/nllb-200-distilled-600M"

SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    src_lang=SRC_LANG,
    tgt_lang=TGT_LANG,
    use_fast=False
)

model.eval()

# -------------------------------
# Filtrage hat -> eng
# -------------------------------
def keep_hat_en(example):
    t = example["translation"]
    return (t["src_lang"] == "hat") and (t["tgt_lang"] == "eng")

test_f = test_ds.filter(keep_hat_en)
print("Test:", len(test_f))

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

predictions = []
references_bleu = []
references_plain = []

# -------------------------------
# Génération
# -------------------------------
for ex in test_f:
    src_text = ex["translation"]["src_text"]
    tgt_text = ex["translation"]["tgt_text"]

    inputs = tokenizer(
        src_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LEN
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            max_length=MAX_LEN,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG)
        )

    pred_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    predictions.append(pred_text)
    references_bleu.append([tgt_text])   # pour sacrebleu
    references_plain.append(tgt_text)    # pour chrf, ter, bleurt

# -------------------------------
# Calcul des scores
# -------------------------------
bleu_result = bleu.compute(
    predictions=predictions,
    references=references_bleu
)

chrf_result = chrf.compute(
    predictions=predictions,
    references=references_plain
)

ter_result = ter.compute(
    predictions=predictions,
    references=references_plain
)

bleurt_result = bleurt.compute(
    predictions=predictions,
    references=references_plain
)

print("BLEU   :", bleu_result["score"])
print("chrF   :", chrf_result["score"])
print("TER    :", ter_result["score"])
print("BLEURT :", float(np.mean(bleurt_result["scores"])))