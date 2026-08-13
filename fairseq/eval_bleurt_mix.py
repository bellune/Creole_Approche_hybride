import evaluate
from pathlib import Path

ref_file = [ Path("datasets/kreyol-mt-hat-eng/mt-fairseq/test.en") , Path("datasets/kreyol-mt-hat-eng/mt-fairseq/test.en"), Path("datasets/kreyol-mt-hat-eng/mt-fairseq/test.en")]
pred_file = [ Path("results/fairseq/outputs/pred_transformer_base.en") , Path("results/mix_fairseq/outputs/pred_transformer_base.en"), Path("results/cs_fairseq/outputs/pred_transformer_base.en")]

for ref ,pred in zip(ref_file, pred_file):
    references = [line.strip() for line in ref.open(encoding="utf-8")]
    predictions = [line.strip() for line in pred.open(encoding="utf-8")]

    assert len(references) == len(predictions), (
        f"Nombre différent de lignes. de {ref.name}: refs={len(references)}, preds={len(predictions)}"
    )

    bleurt = evaluate.load("bleurt", checkpoint="BLEURT-20")

    results = bleurt.compute(
        predictions=predictions,
        references=references
    )

    scores = results["scores"]
    average_bleurt = sum(scores) / len(scores)

    print(f"BLEURT de {ref.name}: {average_bleurt:.4f}")
    print(f"Nombre de phrases de {ref.name} évaluées: {len(scores)}")