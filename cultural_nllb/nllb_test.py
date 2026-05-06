from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd

BASE_MODEL = "facebook/nllb-200-distilled-600M"

BASELINE_MODEL = "nllb200_baseline4/checkpoint-166260"
CULTURAL_MODEL = "model/nllb_cultural_cr_en"

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


print("Testing baseline...")
baseline_model = load_model(BASELINE_MODEL)
sources, baseline_preds, refs = translate_dataset(baseline_model, test_data)

print("Testing cultural-adapted model...")
cultural_model = load_model(CULTURAL_MODEL)
_, cultural_preds, _ = translate_dataset(cultural_model, test_data)

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

cultural_bleu = bleu.compute(
    predictions=cultural_preds,
    references=[[r] for r in refs]
)

baseline_chrf = chrf.compute(
    predictions=baseline_preds,
    references=refs
)

cultural_chrf = chrf.compute(
    predictions=cultural_preds,
    references=refs
)


baseline_ter = ter.compute(
    predictions=baseline_preds,
    references=refs
)

cultural_ter = ter.compute(
    predictions=cultural_preds,
    references=refs
)

baseline_bleurt = bleurt.compute(
    predictions=baseline_preds,
    references=refs
)

cultural_bleurt = bleurt.compute(
    predictions=cultural_preds,
    references=refs
)

print("\n===== RESULTS ON CULTURAL TEST SET =====")
print("Baseline BLEU :", baseline_bleu["score"])
print("Cultural BLEU :", cultural_bleu["score"])

print("Baseline chrF :", baseline_chrf["score"])
print("Cultural chrF :", cultural_chrf["score"])

print("Baseline TER :", baseline_ter["score"])
print("Cultural TER :", cultural_ter["score"])

print("Baseline BLEURT :", baseline_bleurt["scores"][0])
print("Cultural BLEURT :", cultural_bleurt["scores"][0])


df_results = pd.DataFrame({
    "cr": sources,
    "reference_en": refs,
    "baseline_prediction": baseline_preds,
    "cultural_prediction": cultural_preds
})

df_results.to_csv(
    "result/cultural_test_comparison_baseline_vs_adapted.csv",
    index=False,
    encoding="utf-8"
)

print("\nComparaison sauvegardée : result/cultural_test_comparison_baseline_vs_adapted.csv")