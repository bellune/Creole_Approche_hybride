from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd

BASE_MODEL = "facebook/nllb-200-distilled-600M"

BASELINE_MODEL = "nllb200_baseline4/checkpoint-166260"
# CULTURAL_MODEL_TRI = "model/nllb_cultural_tri"
MODELCS = "model/nllb_CS_MT"

path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

ds = load_from_disk(save_path)
print(ds)

test_ds = ds["test"]

# -------------------------------
# Chargement du modèle et tokenizer
# -------------------------------
def keep_hat_en(example):
    t = example["translation"]
    return (t["src_lang"] == "hat") and (t["tgt_lang"] == "eng")


SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"


device = "cuda" if torch.cuda.is_available() else "cpu"

test_f = test_ds.filter(keep_hat_en)
print("Test:", len(test_f))

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


def translate_dataset(model, test_f):
    predictions = []
    references = []
    sources = []

    for example in test_f:
        src = example["translation"]["src_text"]
        ref = example["translation"]["tgt_text"]

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
sources, baseline_preds, refs = translate_dataset(baseline_model, test_f)

# print("Testing  model...")
# model = load_model(MODEL)
# _, n_preds, _, = translate_dataset(model, test_f)

# print("Testing tri-lingual model...")
# tri_model = load_model(CULTURAL_MODEL_TRI)
# _, tri_preds, _, = translate_dataset(tri_model, test_f)

print("Testing tri-lingual model CSS...")
model_cs = load_model(MODELCS)
_, predscs, _, = translate_dataset(model_cs, test_f)


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

cs_bleu = bleu.compute(
    predictions=predscs,
    references=[[r] for r in refs]
)
cs_chrf = chrf.compute(
    predictions=predscs,
    references=refs
)
cs_ter = ter.compute(
    predictions=predscs,
    references=refs
)
cs_bleurt = bleurt.compute(
    predictions=predscs,
    references=refs
)   


 # -----------------------
    # Calcul de la loss
    # -----------------------




# -------------------------------


print("\n===== RESULTS ON CULTURAL TEST SET =====")
print("Baseline BLEU :", baseline_bleu["score"])
# print("Cultural BLEU :", cultural_bleu["score"])
# print("Tri-lingual BLEU :", tri_bleu["score"])
print("Tri-lingual (CS) BLEU :", cs_bleu["score"])
print("Baseline chrF :", baseline_chrf["score"])
# print("Cultural chrF :", cultural_chrf["score"])
# print("Tri-lingual chrF :", tri_chrf["score"])
print("Tri-lingual (CS) chrF :", cs_chrf["score"])

print("Baseline TER :", baseline_ter["score"])
# print("Cultural TER :", cultural_ter["score"])
# print("Tri-lingual TER :", tri_ter["score"])
print("Tri-lingual (CS) TER :", cs_ter["score"])

print("Baseline BLEURT :", baseline_bleurt["scores"][0])
# print("Cultural BLEURT :", cultural_bleurt["scores"][0])
# print("Tri-lingual BLEURT :", tri_bleurt["scores"][0])
print("Tri-lingual (CS) BLEURT :", cs_bleurt["scores"][0])



df_scores = pd.DataFrame({
    "Model": ["Baseline", "NLLB (CS)"],
    "BLEU": [baseline_bleu["score"], cs_bleu["score"]],
    "chrF": [baseline_chrf["score"],  cs_chrf["score"]],
    "TER": [baseline_ter["score"], cs_ter["score"]],
    "BLEURT": [baseline_bleurt["scores"][0], cs_bleurt["scores"][0]]
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
    "cs_prediction": predscs
})

df_results.to_csv(
    "Shared_task2026/result/code-switching_test_comparison_all.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nComparaison sauvegardée : Shared_task2026/result/code-switching_test_comparison_all.csv")