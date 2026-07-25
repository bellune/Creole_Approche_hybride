from pathlib import Path
from xml.parsers.expat import model

from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd
import numpy as np

BASE_MODEL = "facebook/nllb-200-distilled-600M"

BASELINE_MODEL = "nllb200_baseline4/checkpoint-166260"
MODEL_CS = "model/nllb_CSSHT"
MODELCSCULT= "model/nllb_CSCULT"
BASELINE_MODEL_FR = "model/nllb200-baseline-fra"
MODEL_CS_FR= "model/nllb200-CS-fra"


TESTSRC = "uqam_eval_srcs/hat-eng.hat"
TESTREF = "uqam_eval_srcs/eval.generale.uqam.googletranslate.hat-eng.hat.txt"
TESTSRCFR = "uqam_eval_srcs/hat-fra.hat"
TESTREFFR = "uqam_eval_srcs/eval.generale.uqam.googletranslate.hat-fra.hat.txt"
SRC_LANG = "hat_Latn"
TGT_LANG = "eng_Latn"
SRC_LANG_FR = "hat_Latn"
TGT_LANG_FR= "fra_Latn"
DIRECTION = "hat-eng"
DIRECTION_FR = "hat-fra"

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

#HAT-FRA

# NLLB + MT-Kreyòl +Code-switching + Culture
FILE_CS_FR = output_dir/"uqam.1a.primary.hat-fra.txt" 

# NLLB + MT-Kreyòl +Code-switching
FILE_BS_FR = output_dir/"uqam.1a.contrastive1.hat-fra.txt"


# path. score

scores_file = Path(
    "Shared_task2026/result/code-switching_test_scores_all.csv"
)

scores_file.parent.mkdir(parents=True, exist_ok=True)

# Construisons les propriete asscie au model

Models = [
    {"id":"BASELINE01", "model": BASELINE_MODEL, "test":TESTSRC, "testref":TESTREF, "scr":SRC_LANG, "tgt":TGT_LANG, "submitfile":FILE_BS},
    {"id":"MODCS01", "model": MODEL_CS, "test":TESTSRC, "testref":TESTREF, "scr":SRC_LANG, "tgt":TGT_LANG, "submitfile":FILE_CS},
    {"id":"MODELCSCULT01", "model": MODELCSCULT, "test":TESTSRC, "testref":TESTREF, "scr":SRC_LANG, "tgt":TGT_LANG, "submitfile":FILE_CS_CULT},
    {"id":"BASELINEFR02", "model": BASELINE_MODEL_FR, "test":TESTSRCFR, "testref":TESTREFFR, "scr":SRC_LANG_FR, "tgt":TGT_LANG_FR, "submitfile":FILE_BS_FR},
    {"id":"MODCSFR02", "model": MODEL_CS_FR, "test":TESTSRCFR, "testref":TESTREFFR, "scr":SRC_LANG_FR, "tgt":TGT_LANG_FR, "submitfile":FILE_CS_FR}
]


# -------------------------------
# Métriques
# -------------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")
bleurt = evaluate.load("bleurt", config_name="bleurt-base-128")



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


for model_info in Models:
    id = model_info["id"]
    best_model = model_info["model"]
    test_file = model_info["test"]
    test_ref_file = model_info["testref"]
    src_lang = model_info["scr"]
    tgt_lang = model_info["tgt"]
    submit_file = model_info["submitfile"]

    if not best_model or not Path(best_model).exists():
            print(
                f"Modèle introuvable : "
                f"{best_model or ' ce modele se trouve pas dans ce environement'}"
            )
            continue

        # Suite du traitement
    print("Modèle trouvé :", best_model)


# -------------------------------
# Chargement du modèle et tokenizer
# -------------------------------
    with open(test_file, encoding="utf-8") as f:
        test_sentences = [x.strip() for x in f]

    with open(test_ref_file, encoding="utf-8") as f:
        refs = [x.strip() for x in f]

    SRC_LANG = src_lang
    TGT_LANG = tgt_lang


    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.src_lang = SRC_LANG
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)


    print(f"Testing {id}...")
    mymodel = load_model(best_model)
    sources, preds, refs = translate_dataset(mymodel, test_sentences, refs)

# -------------------------------
# Calcul des métriques
# -------------------------------   

    bleu = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )


    chrf = chrf.compute(
        predictions=preds,
        references=refs
    )

    ter = ter.compute(
        predictions=preds,
        references=refs
    )

    bleurt = bleurt.compute(
        predictions=preds,
        references=refs
    )

    # Moyenne BLEURT sur toutes les phrases
    bleurt_scores = bleurt.get("scores", [])
    bleurt_mean = (
    float(np.mean(bleurt_scores))
    if len(bleurt_scores) > 0
    else None
)

    print(f"\n===== RESULTS ON {id} TEST SET =====")
    print(f"{id} BLEU :", bleu["score"])
    print(f"{id} chrF :", chrf["score"])
    print(f"{id} TER :", ter["score"])
    print(f"{id} BLEURT :", bleurt["scores"][0])
    



    df_scor = pd.DataFrame({
        "ID": [id],
        "src": [DIRECTION],
        "reference_en": ["google translate"],
        "Model": ["NLLB"],
        "BLEU": [bleu["score"]],
        "chrF": [chrf["score"]],
        "TER": [ter["score"]],
        "BLEURT": [bleurt["scores"][0]],
        "SUBMITFILE" : [submit_file]
    })

    
    if scores_file.exists():
        df_scores = pd.read_csv(scores_file)

        # Comparaison uniforme des identifiants
        df_scores["ID"] = df_scores["ID"].astype(str)
        mask = df_scores["ID"] == id

        if mask.any():
            # Modifier toutes les colonnes de la ligne existante
            for column, value in df_scor.items():
                df_scores.loc[mask, column] = value

            print(f"Ligne mise à jour : ID={id}")

        else:
            # Ajouter une nouvelle ligne
            df_scores = pd.concat(
                [df_scores, pd.DataFrame([df_scor])],
                ignore_index=True
            )

            print(f"Nouvelle ligne ajoutée : ID={id}")

    else:
        # Créer le fichier avec la première ligne
        df_scores = pd.DataFrame([df_scor])
        print(f"Fichier créé : ID={id}")

    df_scores.to_csv(
        scores_file,
        index=False,
        encoding="utf-8-sig"
    )


    df_scores.to_csv(
        "Shared_task2026/result/code-switching_test_scores_all.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # preds

    df_results = pd.DataFrame({
        "cr": sources,
        "reference_en": refs,
        "prediction": preds,
    })

    for filename, translations in [ (submit_file, preds)]:
        with open(filename,"w",encoding="utf-8") as f:

            for t in translations:
                f.write(t+"\n")


    df_results.to_csv(
        "Shared_task2026/result/task2026_preds_{id}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nComparaison sauvegardée : Shared_task2026/result/task2026_preds_{id}.csv")


