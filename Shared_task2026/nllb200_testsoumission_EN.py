from datasets import load_dataset, load_from_disk
from matplotlib.pylab import save

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd

BASE_MODEL = "facebook/nllb-200-distilled-600M"

BASELINE_MODEL = "nllb200_baseline4/checkpoint-166260"
# CULTURAL_MODEL_TRI = "model/nllb_cultural_tri"
# MODELCS = "model/nllb_CS_MT"

TEST = "uqam_eval_srcs/hat-eng.hat"

# -------------------------------
# Les fichiers A SOUMETTRE HAT-EN
# ------------------------------

# NLLB + MT-Kreyòl +Code-switching + Culture
FILE_CS_CULT = "Shared_task2026/soumission/EN/uqam.1a.primary.hat-eng.txt" 

# NLLB + MT-Kreyòl +Code-switching
FILE_CS = "Shared_task2026/soumission/EN/uqam.1a.contrastive1.hat-eng.txt"

# NLLB + MT-Kreyòl
FILE_BS = "Shared_task2026/soumission/EN/uqam.1a.contrastive2.hat-eng.txt"


# -------------------------------
# Chargement du modèle et tokenizer
# -------------------------------

with open(TEST, encoding="utf-8") as f:
    test_sentences = [x.strip() for x in f]

device = "cuda" if torch.cuda.is_available() else "cpu"

SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.src_lang = SRC_LANG
forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")


def load_model(path):
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    model.to(device)
    model.eval()
    return model


def translate_dataset(model, test_sentences):
    predictions = []
    references = []
    sources = []

    for example in test_sentences:
        src = example

        inputs = tokenizer(
            src,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=4
            )


        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        sources.append(src)
        predictions.append(pred)



    return sources, predictions, references


print("Testing baseline...")
baseline_model = load_model(BASELINE_MODEL)
sources, baseline_preds, refs = translate_dataset(baseline_model, test_sentences)

# print("Testing  model...")
# model = load_model(MODEL)
# _, n_preds, _, = translate_dataset(model, test_f)

# print("Testing tri-lingual model...")
# tri_model = load_model(CULTURAL_MODEL_TRI)
# _, tri_preds, _, = translate_dataset(tri_model, test_f)

# print("Testing tri-lingual model CSS...")
# model_cs = load_model(MODELCS)
# _, predscs, _, = translate_dataset(model_cs, test_sentences)

# -----------------------------------
# save
# -----------------------------------

for filename, translations in [ (FILE_BS, baseline_preds) ]:
    with open(filename,"w",encoding="utf-8") as f:

        for t in translations:
            f.write(t+"\n")


df_results = pd.DataFrame({
    "cr": sources,
    "baseline_prediction": baseline_preds
})

df_results.to_csv(
    "Shared_task2026/result/task2026_HAT_EN_prediction_comparison_all.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nComparaison sauvegardée : Shared_task2026/result/task2026_HAT_EN_prediction_comparison_all.csv")