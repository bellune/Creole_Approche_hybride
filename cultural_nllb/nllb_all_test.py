from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from datasets import load_from_disk

BASE_MODEL = "facebook/nllb-200-distilled-600M"

BASELINE_MODEL = "backup_model/nllb200Baseline/checkpoint-45000"
CULTURAL_MODEL = "backup_model/nllb_cultural_cr_en/checkpoint-22000"
CULTURAL_MODEL_CS = "/root/model/nllb_CS/checkpoint-26000"

TEST_FILE = "datasets/corpus_culturel/test/cr_en.jsonl"


path_data = "datasets"
save_path = path_data + "/kreyol-mt-hat-eng"

ds = load_from_disk(save_path)
print(ds)



datasetest = load_dataset(
    "json",
    data_files={
        "test": TEST_FILE    }
)


train_ds = concatenate_datasets([
    ds["test"],
    datasetest["test"],
])
    
train_ds = train_ds.shuffle(seed=42)
print("Train:", len(train_ds))


# -------------------------------
# Chargement du modèle et du tokenizer
# -------------------------------


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
    ids = []

    for example in dataset:
        src = example["translation"]["src_text"]
        ref = example["translation"]["tgt_text"]
        id = example["id"]

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

        ids.append(id)
        sources.append(src)
        predictions.append(pred)
        references.append(ref)



    return ids,sources, predictions, references


print("Testing baseline...")
baseline_model = load_model(BASELINE_MODEL)
ids, sources, baseline_preds, refs = translate_dataset(baseline_model, test_data)

print("Testing cultural-adapted model...")
cultural_model = load_model(CULTURAL_MODEL)
_,_, cultural_preds, _, = translate_dataset(cultural_model, test_data)

print("Testing Czech-Slovak model...")
cs_model = load_model(CULTURAL_MODEL_CS)
_, _, cs_preds, _ = translate_dataset(cs_model, test_data)

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

cs_bleu = bleu.compute(
    predictions=cs_preds,
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

cs_chrf = chrf.compute(
    predictions=cs_preds,
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

cs_ter = ter.compute(
    predictions=cs_preds,
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
cs_bleurt = bleurt.compute(
    predictions=cs_preds,
    references=refs
)

 # -----------------------
    # Calcul de la loss
    # -----------------------




# -------------------------------


print("\n===== RESULTS ON CULTURAL TEST SET =====")
print("Baseline BLEU :", baseline_bleu["score"])
print("Cultural BLEU :", cultural_bleu["score"])
print("Czech-Slovak BLEU :", cs_bleu["score"])

print("Baseline chrF :", baseline_chrf["score"])
print("Cultural chrF :", cultural_chrf["score"])
print("Czech-Slovak chrF :", cs_chrf["score"])

print("Baseline TER :", baseline_ter["score"])
print("Cultural TER :", cultural_ter["score"])
print("Czech-Slovak TER :", cs_ter["score"])

print("Baseline BLEURT :", baseline_bleurt["scores"][0])
print("Cultural BLEURT :", cultural_bleurt["scores"][0])
print("Czech-Slovak BLEURT :", cs_bleurt["scores"][0])



df_scores = pd.DataFrame({
    "Model": ["Baseline", "Cultural", "Czech-Slovak"],
    "BLEU": [baseline_bleu["score"], cultural_bleu["score"], cs_bleu["score"]],
    "chrF": [baseline_chrf["score"], cultural_chrf["score"], cs_chrf["score"]],
    "TER": [baseline_ter["score"], cultural_ter["score"], cs_ter["score"]],
    "BLEURT": [baseline_bleurt["scores"][0], cultural_bleurt["scores"][0], cs_bleurt["scores"][0]          ]
})

df_scores.to_csv(
    "result/NLL20_test_scores_all.csv",
    index=False,
    encoding="utf-8-sig"
)




df_results = pd.DataFrame({
    "id": ids,
    "cr": sources,
    "reference_en": refs,
    "baseline_prediction": baseline_preds,
    "cultural_prediction": cultural_preds,
    "CS_prediction": cs_preds
})

df_results.to_csv(
    "result/NLL20_test_comparison_all.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nComparaison sauvegardée : result/NLL20_test_comparison_all.csv")