from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd

BASE_MODEL = "facebook/nllb-200-distilled-600M"

CULTURAL_MODEL = "model/nllb_cultural_cr_en"
CULTURAL_MODEL_TRI = "model/nllb_cultural_tri"

TEST_FILE = "datasets/corpus_culturel/test/cr_en.jsonl"

SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"

device = "cuda" if torch.cuda.is_available() else "cpu"

test_data = load_dataset("json", data_files={"test": TEST_FILE})["test"]

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


def translate_dataset(model, dataset):
    predictions = []
    references = []
    sources = []

    for example in dataset:
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


print("Testing cultural-adapted model....")
cultural_model = load_model(CULTURAL_MODEL)
sources, cultural_preds, refs = translate_dataset(cultural_model, test_data)

print("Testing cultural-adapted model Trilingue...")
cultural_model_fr = load_model(CULTURAL_MODEL_TRI)
_, cultural_fr_preds, _, = translate_dataset(cultural_model_fr, test_data)

# -------------------------------
# Métriques
# -------------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")
bleurt = evaluate.load("bleurt", config_name="bleurt-base-128")


cultural_bleu = bleu.compute(
    predictions=cultural_preds,
    references=[[r] for r in refs]
)

cultural_fr_bleu = bleu.compute(
    predictions=cultural_fr_preds,
    references=[[r] for r in refs]
)

cultural_chrf = chrf.compute(
    predictions=cultural_preds,
    references=refs
)

cultural_fr_chrf = chrf.compute(
    predictions=cultural_fr_preds,
    references=refs
)


cultural_ter = ter.compute(
    predictions=cultural_preds,
    references=refs
)

cultural_fr_ter = ter.compute(
    predictions=cultural_fr_preds,
    references=refs
)

cultural_bleurt = bleurt.compute(
    predictions=cultural_preds,
    references=refs
)

cultural_fr_bleurt = bleurt.compute(
    predictions=cultural_fr_preds,
    references=refs
)

 # -----------------------
    # Calcul de la loss
    # -----------------------




# -------------------------------


print("\n===== RESULTS ON CULTURAL TEST SET =====")
print("cultural BLEU :", cultural_bleu["score"])
print("Cultural FR BLEU :", cultural_fr_bleu["score"])

print("Cultural chrF :", cultural_chrf["score"])
print("Cultural FR chrF :", cultural_fr_chrf["score"])

print("Cultural TER :", cultural_ter["score"])
print("Cultural FR TER :", cultural_fr_ter["score"])

print("Cultural BLEURT :", cultural_bleurt["scores"][0])
print("Cultural FR BLEURT :", cultural_fr_bleurt["scores"][0])




df_results = pd.DataFrame({
    "cr": sources,
    "reference_en": refs,
    "_prediction": cultural_preds,
    "cultural_prediction": cultural_fr_preds,

})

df_results.to_csv(
    "result/cultural_test_comparison_cultural_vs_adapted.csv",
    index=False,
    encoding="utf-8"
)

print("\nComparaison sauvegardée : result/cultural_test_comparison_cultural_vs_adapted.csv")