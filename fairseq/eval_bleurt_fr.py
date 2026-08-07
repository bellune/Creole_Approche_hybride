import evaluate
from pathlib import Path

ref_file = Path("datasets/kreyol-mt-hat-fra/mt_fr_fairseq/test.fr")
pred_file = Path("results/FR_fairseq/outputs/pred_transformer_base.fr")

references = [line.strip() for line in ref_file.open(encoding="utf-8")]
predictions = [line.strip() for line in pred_file.open(encoding="utf-8")]

assert len(references) == len(predictions), (
    f"Nombre différent de lignes: refs={len(references)}, preds={len(predictions)}"
)

bleurt = evaluate.load("bleurt", checkpoint="BLEURT-20")

results = bleurt.compute(
    predictions=predictions,
    references=references
)

scores = results["scores"]
average_bleurt = sum(scores) / len(scores)

print(f"BLEURT: {average_bleurt:.4f}")
print(f"Nombre de phrases évaluées: {len(scores)}")