from pathlib import Path

from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd

BASE_MODEL = "facebook/nllb-200-distilled-600M"

BASELINE_MODEL = "nllb200_baseline4/checkpoint-166260"
# CULTURAL_MODEL_TRI = "model/nllb_cultural_tri"
# MODELCS = "model/nllb_CS_MT"

TESTSRC = "uqam_eval_srcs/hat-eng.hat"
TESTREF = "uqam_eval_srcs/eval.generale.uqam.googletranslate.hat-eng.hat.txt"



# -------------------------------
# Les fichiers A SOUMETTRE HAT-EN
# ------------------------------

output_dir = Path("Shared_task2026/soumission/EN")
output_dir.mkdir(parents=True, exist_ok=True)

# NLLB + MT-Kreyòl +Code-switching + Culture
FILE_CS_CULT = output_dir/"uqam.1a.primary.hat-eng.txt" 

# NLLB + MT-Kreyòl +Code-switching
FILE_CS = output_dir/"uqam.1a.contrastive1.hat-eng.txt"

# NLLB + MT-Kreyòl
FILE_BS = output_dir/"uqam.1a.contrastive2.hat-eng.txt"
# -------------------------------
# Chargement du modèle et tokenizer
# -------------------------------
with open(TESTSRC, encoding="utf-8") as f:
    test_sentences = [x.strip() for x in f]

with open(TESTREF, encoding="utf-8") as f:
    refs = [x.strip() for x in f]

SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"


device = "cuda" if torch.cuda.is_available() else "cpu"

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


def translate_dataset(model, test_sentences, refs):
    predictions = []
    references = []
    sources = []

    for example, ref in zip(test_sentences, refs):
        src = example
        ref = ref

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
        references.append(ref)



    return sources, predictions, references


print("Testing baseline...")
baseline_model = load_model(BASELINE_MODEL)
sources, baseline_preds, refs = translate_dataset(baseline_model, test_sentences, refs)

# print("Testing  model...")
# model = load_model(MODEL)
# _, n_preds, _, = translate_dataset(model, test_sentences, refs)

# print("Testing tri-lingual model...")
# tri_model = load_model(CULTURAL_MODEL_TRI)
# _, tri_preds, _, = translate_dataset(tri_model, test_sentences, refs)

# print("Testing tri-lingual model CSS...")
# model_cs = load_model(MODELCS)
# _, predscs, _, = translate_dataset(model_cs, test_sentences, refs)


# -------------------------------
# Métriques
# -------------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")
bleurt = evaluate.load("bleurt", config_name="bleurt-base-128")


baseline_bleu = bleu.compute(
    predictions=baseline_preds,
    references=[[r] for r in refs]
)

# cultural_bleu = bleu.compute(
#     predictions=n_preds,
#     references=[[r] for r in refs]
# )

# tri_bleu = bleu.compute(
#     predictions=tri_preds,
#     references=[[r] for r in refs]
# )

baseline_chrf = chrf.compute(
    predictions=baseline_preds,
    references=refs
)

# cultural_chrf = chrf.compute(
#     predictions=n_preds,
#     references=refs
# )

# tri_chrf = chrf.compute(
#     predictions=tri_preds,
#     references=refs
# )

baseline_ter = ter.compute(
    predictions=baseline_preds,
    references=refs
)

# cultural_ter = ter.compute(
#     predictions=n_preds,
#     references=refs
# )

# tri_ter = ter.compute(
#     predictions=tri_preds,
#     references=refs
# )

baseline_bleurt = bleurt.compute(
    predictions=baseline_preds,
    references=refs
)

# cultural_bleurt = bleurt.compute(
#     predictions=n_preds,
#     references=refs
# )
# tri_bleurt = bleurt.compute(
#     predictions=tri_preds,
#     references=refs
# )

# cs_bleu = bleu.compute(
#     predictions=predscs,
#     references=[[r] for r in refs]
# )
# cs_chrf = chrf.compute(
#     predictions=predscs,
#     references=refs
# )
# cs_ter = ter.compute(
#     predictions=predscs,
#     references=refs
# )
# cs_bleurt = bleurt.compute(
#     predictions=predscs,
#     references=refs
# )   


 # -----------------------
    # Calcul de la loss
    # -----------------------




# -------------------------------


print("\n===== RESULTS ON code-switching TEST SET =====")
print("Baseline BLEU :", baseline_bleu["score"])
# print("Cultural BLEU :", cultural_bleu["score"])
# print("Tri-lingual BLEU :", tri_bleu["score"])
# print("(CS) BLEU :", cs_bleu["score"])
print("Baseline chrF :", baseline_chrf["score"])
# print("Cultural chrF :", cultural_chrf["score"])
# print("Tri-lingual chrF :", tri_chrf["score"])
# print("(CS) chrF :", cs_chrf["score"])

print("Baseline TER :", baseline_ter["score"])
# print("Cultural TER :", cultural_ter["score"])
# print("Tri-lingual TER :", tri_ter["score"])
# print("(CS) TER :", cs_ter["score"])

print("Baseline BLEURT :", baseline_bleurt["scores"][0])
# print("Cultural BLEURT :", cultural_bleurt["scores"][0])
# print("Tri-lingual BLEURT :", tri_bleurt["scores"][0])
# print("(CS) BLEURT :", cs_bleurt["scores"][0])



df_scores = pd.DataFrame({
    "cr": sources,
    "reference_en": refs,
    "direction": ["hat-eng"] * len(sources),
    "Model": ["NLLB Baseline", "NLLB (CS)"],
    "BLEU": [baseline_bleu["score"]],
    "chrF": [baseline_chrf["score"]],
    "TER": [baseline_ter["score"]],
    "BLEURT": [baseline_bleurt["scores"][0]]
})

df_scores.to_csv(
    "Shared_task2026/result/code-switching_test_scores_all.csv",
    index=False,
    encoding="utf-8-sig"
)




df_results = pd.DataFrame({
    "cr": sources,
    "reference_en": refs,
    "baseline_prediction": baseline_preds,
    # "nllb_cs_prediction": n_preds,
    # "tri_prediction": tri_preds,
    # "cs_prediction": predscs
})

for filename, translations in [ (FILE_BS, baseline_preds)]:
    with open(filename,"w",encoding="utf-8") as f:

        for t in translations:
            f.write(t+"\n")


df_results.to_csv(
    "Shared_task2026/result/task2026_HAT_EN_test_comparison_all.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nComparaison sauvegardée : Shared_task2026/result/task2026_HAT_EN_test_comparison_all.csv")