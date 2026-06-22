import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import sentencepiece as spm


data_fairseq = "datasets/mix_corpus/fairseq"
base_dir = "model/fairseq"
results_dir = "results/fairseq"
BASELINE = "model/fairseq/checkpoints/transformer_base_ht_en/checkpoint_best.pt"




SRC = "ht"
TGT = "en"

RAW_DIR = Path(data_fairseq)
SPM_DIR = Path(f"{base_dir}/spm")
BIN_DIR = Path(f"model/mix_fairseq/data-bin/ht-en")
CKPT_DIR = Path(f"{base_dir}/checkpoints/transformer_base_ht_en")
OUT_DIR = Path(f"{results_dir}/outputs")

SPM_DIR.mkdir(parents=True, exist_ok=True)
BIN_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: str):
    print(f"\nRunning:\n{cmd}\n")
    result = subprocess.run(
        cmd,
        shell=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed:\n{cmd}")


def check_files():
    required_files = [
        RAW_DIR / f"test_culture_mix.{SRC}",
        RAW_DIR / f"test_culture_mix.{TGT}"
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    print("All files found.")



def generate_translations():
    output_file = OUT_DIR / "outputs_transformer_base_Cultural.txt"

    cmd = f"""
    CUDA_VISIBLE_DEVICES="" python3 -m fairseq_cli.generate {BIN_DIR} \
      --source-lang {SRC} \
      --target-lang {TGT} \
      --path {BASELINE} \
      --beam 5 \
      --batch-size 64 \
      --remove-bpe=sentencepiece \
      > {output_file}
    """

    run_cmd(cmd)


def extract_predictions():
    fairseq_output = OUT_DIR / "outputs_transformer_base_Cultural.txt"
    prediction_file = OUT_DIR / "pred_transformer_base_Cultural.en"

    cmd = f"""
    grep '^H-' {fairseq_output} \
      | sort -V \
      | cut -f3- \
      > {prediction_file}
    """

    run_cmd(cmd)


def run_sacrebleu(ref_file, pred_file):
    cmd = [
        "sacrebleu",
        str(ref_file),
        "-i", str(pred_file),
        "-m", "bleu", "chrf", "ter",
        "-f", "json"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True
    )

    scores_json = json.loads(result.stdout)

    scores = {}
    for metric in scores_json:
        name = metric["name"]
        scores[name] = metric["score"]

    return {
        "BLEU": scores.get("BLEU"),
        "chrF2": scores.get("chrF2"),
        "TER": scores.get("TER")
    }


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def evaluate():
    prediction_files = [
        OUT_DIR / "pred_transformer_base_Culture.en"
    ]

    reference_files = [
        RAW_DIR / f"test_culture_mix.{TGT}"
    ]

    test_names = [
        "Cultural test"
    ]

    rows = []

    for test_name, ref, pred in zip(test_names, reference_files, prediction_files):
        if not ref.exists():
            raise FileNotFoundError(f"Missing reference file: {ref}")

        if not pred.exists():
            raise FileNotFoundError(f"Missing prediction file: {pred}")

        print("-----------------------------------------------------")
        print(f"Evaluating {test_name}")
        print(f"Reference: {ref}")
        print(f"Prediction: {pred}")
        print("-----------------------------------------------------")

        scores = run_sacrebleu(ref, pred)

        rows.append({
            "Model": "Transformer Fairseq",
            "Training": "MT-Creole + Cultural Mix",
            "Test": test_name,
            "BLEU": scores["BLEU"],
            "chrF2": scores["chrF2"],
            "TER": scores["TER"]
        })

        print(scores)

    df_scores = pd.DataFrame(rows)

    score_path = OUT_DIR / "cultural_Baseline_MIX_test_scores_all.csv"
    df_scores.to_csv(score_path, index=False, encoding="utf-8-sig")

    print(f"\nScores sauvegardés : {score_path}")




def save_cultural_comparison():
        src_file = RAW_DIR / f"test_culture_mix.{SRC}"
        ref_file = RAW_DIR / f"test_culture_mix.{TGT}"
    
        ids  = "datasets/mix_corpus/json/test_culture.jsonl"

        #lire le fichier test jsonl pour recuperer les ids
        with open(ids, "r") as f:
            test_data = [json.loads(line) for line in f]
        ids = [item["id"] for item in test_data]

        general_pred_file = OUT_DIR / "pred_transformer_base_General.en"
        cultural_pred_file = OUT_DIR / "pred_transformer_base_Culture.en"

        sources = read_lines(src_file)
        refs = read_lines(ref_file)
       

        cultural_preds = read_lines(cultural_pred_file)

        n = len(refs)
   

        assert len(sources) == n, f"sources={len(sources)} refs={n}"
        assert len(cultural_preds) == n, f"cultural={len(cultural_preds)} refs={n}"

        df_results = pd.DataFrame({
            "id": ids,
            "cr": sources,
            "reference_en": refs,
            "cultural_prediction": cultural_preds
        })


        output_path = OUT_DIR / "cultural_MIX_test_comparison_all.csv"
        df_results.to_csv(output_path, index=False, encoding="utf-8-sig")


        print(f"Comparaison sauvegardée : {output_path}")



if __name__ == "__main__":
    check_files()


    generate_translations()
    
    extract_predictions()

    evaluate()

    save_cultural_comparison()

