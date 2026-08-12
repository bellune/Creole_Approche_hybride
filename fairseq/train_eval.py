import os
import subprocess
from pathlib import Path
import sentencepiece as spm


data_fairseq = "datasets/kreyol-mt-hat-eng/mt-fairseq"
base_dir = "model/fairseq"
results_dir = "results/fairseq"
base_dir_checkpoints = "/root/model/fairseq"
last_checkpoint = "/root/model/fairseq/transformer_base_ht_en/checkpoint_best.pt"



SRC = "ht"
TGT = "en"

RAW_DIR = Path(data_fairseq)
SPM_DIR = Path(f"{base_dir}/spm")
BIN_DIR = Path(f"{base_dir}/data-bin/ht-en")
CKPT_DIR = Path(f"{base_dir_checkpoints}/transformer_base_ht_en")
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
        RAW_DIR / f"train.{SRC}",
        RAW_DIR / f"train.{TGT}",
        RAW_DIR / f"valid.{SRC}",
        RAW_DIR / f"valid.{TGT}",
        RAW_DIR / f"test.{SRC}",
        RAW_DIR / f"test.{TGT}",
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    print("All files found.")


def train_sentencepiece():
 

    combined_file = SPM_DIR / "train_all.txt"

    with open(combined_file, "w", encoding="utf-8") as outfile:
        for file in [RAW_DIR / f"train.{SRC}", RAW_DIR / f"train.{TGT}"]:
            with open(file, "r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        outfile.write(line + "\n")

    print(f"Training SentencePiece on: {combined_file}")

    spm.SentencePieceTrainer.train(
        input=str(combined_file),
        model_prefix=str(SPM_DIR / "ht_en_spm"),
        vocab_size=32000,
        character_coverage=1.0,
        model_type="bpe"
    )

    print("SentencePiece training finished.")


def apply_sentencepiece():

    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_DIR / "ht_en_spm.model"))

    for split in ["train", "valid", "test"]:
        for lang in [SRC, TGT]:
            input_file = RAW_DIR / f"{split}.{lang}"
            output_file = RAW_DIR / f"{split}.spm.{lang}"

            print(f"Encoding {input_file} -> {output_file}")

            count = 0

            with open(input_file, "r", encoding="utf-8") as infile, \
                 open(output_file, "w", encoding="utf-8") as outfile:

                for line in infile:
                    line = line.strip()

                    if line:
                        pieces = sp.encode(line, out_type=str)
                        outfile.write(" ".join(pieces) + "\n")
                    else:
                        outfile.write("\n")

                    count += 1

                    if count % 1000 == 0:
                        print(f"{split}.{lang}: {count} lignes encodées")

            print(f"{split}.{lang}: terminé avec {count} lignes")


def fairseq_preprocess():
    cmd = f"""
    python3 -m fairseq_cli.preprocess \
      --source-lang {SRC} \
      --target-lang {TGT} \
      --trainpref {RAW_DIR}/train.spm \
      --validpref {RAW_DIR}/valid.spm \
      --testpref {RAW_DIR}/test.spm \
      --destdir {BIN_DIR} \
      --workers 8 \
      --joined-dictionary
    """

    run_cmd(cmd)


def train_transformer():
    cmd = f"""
    CUDA_VISIBLE_DEVICES=0 python3 -m fairseq_cli.train {BIN_DIR} \
     --arch transformer \
--source-lang {SRC} \
--target-lang {TGT} \
--share-decoder-input-output-embed \
--encoder-layers 6 \
--decoder-layers 6 \
--encoder-embed-dim 512 \
--decoder-embed-dim 512 \
--encoder-attention-heads 8 \
--decoder-attention-heads 8 \
--encoder-ffn-embed-dim 2048 \
--decoder-ffn-embed-dim 2048 \
--dropout 0.3 \
--attention-dropout 0.1 \
--activation-dropout 0.1 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--optimizer adam \
--adam-betas '(0.9,0.98)' \
--lr 0.0005 \
--lr-scheduler inverse_sqrt \
--warmup-updates 4000 \
--clip-norm 1.0 \
--max-tokens 2048 \
--update-freq 2 \
--max-epoch 30 \
--patience 5 \
--save-dir {CKPT_DIR} \
--keep-last-epochs 2 \
--keep-best-checkpoints 1 \
--best-checkpoint-metric loss \
--distributed-world-size 1
    """

    run_cmd(cmd)


def generate_translations():
    output_file = OUT_DIR / "outputs_transformer_base.txt"

    cmd = f"""
    CUDA_VISIBLE_DEVICES=0 python3 -m fairseq_cli.generate {BIN_DIR} \
      --source-lang {SRC} \
      --target-lang {TGT} \
      --path {CKPT_DIR}/checkpoint_best.pt \
      --beam 5 \
      --batch-size 64 \
      --remove-bpe=sentencepiece \
      > {output_file}
    """

    run_cmd(cmd)


def extract_predictions():
    fairseq_output = OUT_DIR / "outputs_transformer_base.txt"
    prediction_file = OUT_DIR / "pred_transformer_base.en"

    cmd = f"""
    grep '^H-' {fairseq_output} \
      | sort -V \
      | cut -f3- \
      > {prediction_file}
    """

    run_cmd(cmd)


def evaluate():
    prediction_file = OUT_DIR / "pred_transformer_base.en"
    reference_file = RAW_DIR / f"test.{TGT}"
  
    cmd = (
        f"sacrebleu {reference_file} "
        f"-i {prediction_file} "
        f"-m bleu chrf ter "
        f"--chrf-word-order 2"
    )

    run_cmd(cmd)


def generate_attentions():
    output_file = OUT_DIR / "outputs_transformer_base.txt"

    cmd = f"""
    CUDA_VISIBLE_DEVICES=0 python3 -m fairseq_cli.generate {BIN_DIR} \
      --source-lang {SRC} \
      --target-lang {TGT} \
      --path {CKPT_DIR}/checkpoint_best.pt \
      --beam 5 \
      --batch-size 64 \
      --remove-bpe=sentencepiece \
      > {output_file}
    """

    run_cmd(cmd)


if __name__ == "__main__":
    check_files()

    if not (SPM_DIR / "ht_en_spm.model").exists():
        train_sentencepiece()
    else:
        print("SentencePiece model already exists. Skipping training.")

    if not (RAW_DIR / f"train.spm.{SRC}").exists():
        apply_sentencepiece()
    else:
        print("SentencePiece files already exist. Skipping encoding.")

    if not (BIN_DIR / f"dict.{SRC}.txt").exists():
        fairseq_preprocess()
    else:
        print("Fairseq binary data already exists. Skipping preprocess.")
    
    # if not (CKPT_DIR / "checkpoint_best.pt").exists():
    train_transformer()
    # else:      
        # print("Checkpoint already exists. Skipping training.")

    # if not (OUT_DIR / "outputs_transformer_base.txt").exists():
    generate_translations()
    # else:       
    #    print("Translations already generated. Skipping generation.")

    # if not (OUT_DIR / "pred_transformer_base.en").exists():
    extract_predictions()
    # else:       
        # print("Predictions already extracted. Skipping extraction.")

    evaluate()