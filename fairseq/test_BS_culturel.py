import os
import subprocess
from pathlib import Path
import sentencepiece as spm


data_fairseq = "datasets/mix_corpus/fairseq"
base_dir = "model/fairseq"
results_dir = "results/fairseq"
BASELINE = "model/fairseq/checkpoints/transformer_base_ht_en/checkpoint_best.pt"




SRC = "ht"
TGT = "en"

RAW_DIR = Path(data_fairseq)
SPM_DIR = Path(f"{base_dir}/spm")
BIN_DIR = Path(f"{base_dir}/data-bin/ht-en")
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


def evaluate():
    prediction_file = OUT_DIR / "pred_transformer_base_Cultural.en"
    reference_file = RAW_DIR / f"test_culture_mix.{TGT}"

    cmd = f"""
    sacrebleu {reference_file} \
      -i {prediction_file} \
      -m bleu chrf ter
    """

    run_cmd(cmd)





if __name__ == "__main__":
    check_files()



    if not (OUT_DIR / "outputs_transformer_base_Cultural.txt").exists():
      generate_translations()
    else:       
        print("Translations already generated. Skipping generation.")

    if not (OUT_DIR / "pred_transformer_base_Cultural.en").exists():
        extract_predictions()
    else:       
        print("Predictions already extracted. Skipping extraction.")

    evaluate()