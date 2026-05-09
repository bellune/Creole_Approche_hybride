import pandas as pd
import evaluate

# -------------------------------
# Fichiers résultats
# -------------------------------

files = {
    "basic": {
        "path": "result/test_cr_en_with_GPT_predictions.csv",
        "pred_col": "prediction"
    },
    "Cultural": {
        "path": "result/cultural_with_GPT_predictions.csv",
        "pred_col": "prediction"
    },
    "LDP": {
        "path": "result/ldp_with_GPT_predictions.csv",
        "pred_col": "prediction"
    }
}

# -------------------------------
# Métriques
# -------------------------------

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")
bleurt = evaluate.load("bleurt", config_name="bleurt-base-128")

# -------------------------------
# Charger les fichiers
# -------------------------------

all_predictions = {}
refs = None
sources = None

for name, info in files.items():

    print(f"\nLoading {name} from {info['path']}")

    df = pd.read_csv(info["path"])

    # print(df.head())
    # print(df.columns)

    if refs is None:
        refs = df["reference"].astype(str).tolist()
        sources = df["src_text"].astype(str).tolist()

    preds = df[info["pred_col"]].astype(str).tolist()
    all_predictions[name] = preds

# -------------------------------
# Calcul des scores
# -------------------------------

score_rows = []

for name, preds in all_predictions.items():

    bleu_score = bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )

    chrf_score = chrf.compute(
        predictions=preds,
        references=refs
    )

    ter_score = ter.compute(
        predictions=preds,
        references=refs
    )
    bleurt_score = bleurt.compute(
        predictions=preds,        
         references=refs
    )

    score_rows.append({
        "Model": name,
        "BLEU": bleu_score["score"],
        "chrF": chrf_score["score"],
        "TER": ter_score["score"],
        "BLEURT": bleurt_score["score"]

    })

# -------------------------------
# Sauvegarder les scores
# -------------------------------

df_scores = pd.DataFrame(score_rows)

df_scores.to_csv(
    "result/cultural_test_scores_all.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===== RESULTS ON CULTURAL TEST SET =====")
print(df_scores)

# -------------------------------
# Comparaison phrase par phrase
# -------------------------------

df_results = pd.DataFrame({
    "cr": sources,
    "reference_en": refs,
    "basic_prediction": all_predictions["basic"],
    "cultural_prediction": all_predictions["Cultural"],
    "ldp_prediction": all_predictions["LDP"]
})

df_results.to_csv(
    "result/cultural_test_comparison_all.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nScores sauvegardés : result/cultural_test_scores_all.csv")
print("Comparaison sauvegardée : result/cultural_test_comparison_all.csv")