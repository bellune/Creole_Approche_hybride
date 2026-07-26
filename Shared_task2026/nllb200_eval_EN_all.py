from pathlib import Path
from xml.parsers.expat import model

from datasets import load_dataset, load_from_disk

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import evaluate
import pandas as pd
import numpy as np

from tqdm.auto import tqdm



BASE_MODEL = "facebook/nllb-200-distilled-600M"

BASELINE_MODEL = "nllb200_baseline4"
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
    model = AutoModelForSeq2SeqLM.from_pretrained(
        path,
        dtype=torch.float16,
        device_map="balanced_low_0",
        low_cpu_mem_usage=True
    )

    model.eval()

    print("Répartition du modèle sur les GPU :")

    device_map = getattr(model, "hf_device_map", None)

    if device_map is not None:
        print(device_map)
    else:
        print(
            "hf_device_map non disponible. "
            f"Premier paramètre placé sur : {next(model.parameters()).device}"
        )

    return model


def translate_dataset(model, test_sentences, refs):
    predictions = []
    references = []
    sources = []

    input_device = model.get_input_embeddings().weight.device
    print("GPU utilisé pour les entrées :", input_device)

    total = min(len(test_sentences), len(refs))

    for example, ref in tqdm(
        zip(test_sentences, refs),
        total=total,
        desc="Traduction"
    ):
        src = example

        inputs = tokenizer(
            src,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )

        inputs = {
            key: value.to(input_device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=128,
                num_beams=4
            )

        pred = tokenizer.batch_decode(
            outputs.cpu(),
            skip_special_tokens=True
        )[0]

        sources.append(src)
        predictions.append(pred)
        references.append(ref)

        del inputs
        del outputs

    return sources, predictions, references





from pathlib import Path
import json


def resolve_model_path(path):
    path = Path(path)

    if not path.exists():
        print(f"Chemin inexistant : {path}")
        return None

    # Tester d'abord le chemin donné, puis les checkpoints du plus récent au plus ancien
    checkpoints = sorted(
        path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1])
        if p.name.split("-")[-1].isdigit()
        else -1,
        reverse=True
    )

    candidates = [path] + checkpoints

    for candidate in candidates:
        config_file = candidate / "config.json"

        has_weights = (
            any(candidate.glob("model*.safetensors"))
            or any(candidate.glob("pytorch_model*.bin"))
        )

        if not config_file.exists() or not has_weights:
            continue

        try:
            with open(config_file, "r", encoding="utf-8") as file:
                config = json.load(file)

            if config.get("model_type"):
                print(f"Modèle valide trouvé : {candidate}")
                return str(candidate)

        except (json.JSONDecodeError, OSError) as error:
            print(f"Config invalide dans {candidate} : {error}")

    print(f"Aucun checkpoint valide trouvé dans : {path}")
    return None


for model_info in Models:
    id = model_info["id"]
    best_model = model_info["model"]
    test_file = model_info["test"]
    test_ref_file = model_info["testref"]
    src_lang = model_info["scr"]
    tgt_lang = model_info["tgt"]
    submit_file = model_info["submitfile"]

    best_model = resolve_model_path(best_model)

    if not best_model:
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
    "SUBMITFILE": [str(submit_file)]
    })

    if scores_file.exists() and scores_file.stat().st_size > 0:

        df_scores = pd.read_csv(
            scores_file,
            encoding="utf-8-sig",
            sep=None,
            engine="python"
        )

        # Nettoyer les noms de colonnes
        df_scores.columns = (
            df_scores.columns
            .astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

        # Trouver une éventuelle colonne id, ID, Id, etc.
        id_column = next(
            (
                column
                for column in df_scores.columns
                if column.lower() == "id"
            ),
            None
        )

        if id_column is None:
            print("Aucune colonne ID trouvée : remplacement du fichier.")

            # df_scor est déjà un DataFrame
            df_scores = df_scor.copy()

        else:
            if id_column != "ID":
                df_scores.rename(
                    columns={id_column: "ID"},
                    inplace=True
                )

            df_scores["ID"] = df_scores["ID"].astype(str)

            # Supprimer l'ancienne ligne ayant le même ID
            id_exists = (df_scores["ID"] == id).any()

            df_scores = df_scores[
                df_scores["ID"] != id
            ]

            # Ajouter la nouvelle version de la ligne
            df_scores = pd.concat(
                [df_scores, df_scor],
                ignore_index=True
            )

            if id_exists:
                print(f"Ligne remplacée : ID={id}")
            else:
                print(f"Nouvelle ligne ajoutée : ID={id}")

    else:
        df_scores = df_scor.copy()
        print("Nouveau fichier créé.")

    df_scores.to_csv(
        scores_file,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Résultats enregistrés dans : {scores_file}")


    

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
        f"Shared_task2026/result/task2026_preds_{id}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nComparaison sauvegardée : Shared_task2026/result/task2026_preds_{id}.csv")


