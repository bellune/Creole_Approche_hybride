import json
import os
import subprocess
from pathlib import Path
from turtle import pd
import sentencepiece as spm
import subprocess
import pandas as pd
from pathlib import Path



data_fairseq = "datasets/mix_corpus/fairseq"
base_dir = "model/mix_fairseq"
results_dir = "results/mix_fairseq"
BASELINE = "model/fairseq/checkpoints/transformer_base_ht_en/checkpoint_best.pt"

SRC = "ht"
TGT = "en"

RAW_DIR = Path(data_fairseq)
SPM_DIR = Path(f"{base_dir}/spm")
BIN_DIR = Path(f"{base_dir}/data-bin/ht-en")
CKPT_DIR = Path(f"{base_dir}/checkpoints/transformer_base_mix_ht_en")
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
        RAW_DIR / f"train_mix.{SRC}",
        RAW_DIR / f"train_mix.{TGT}",
        RAW_DIR / f"valid_mix.{SRC}",
        RAW_DIR / f"valid_mix.{TGT}",
        RAW_DIR / f"test_culture_mix.{SRC}",
        RAW_DIR / f"test_culture_mix.{TGT}",
        RAW_DIR / f"test_general_mix.{SRC}",
        RAW_DIR / f"test_general_mix.{TGT}",
    ]

    for file in required_files:
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

    print("All files found.")


def train_sentencepiece():
 

    combined_file = SPM_DIR / "train_all.txt"

    with open(combined_file, "w", encoding="utf-8") as outfile:
        for file in [RAW_DIR / f"train_mix.{SRC}", RAW_DIR / f"train_mix.{TGT}"]:
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

    for split in ["train", "valid", "test_culture","test_general"]:
        for lang in [SRC, TGT]:
            input_file = RAW_DIR / f"{split}_mix.{lang}"
            output_file = RAW_DIR / f"{split}_mix.spm.{lang}"

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

                    if count % 10000 == 0:
                        print(f"{split}.{lang}: {count} lignes encodées")

            print(f"{split}.{lang}: terminé avec {count} lignes")


def fairseq_preprocess():
    cmd = f"""
    CUDA_VISIBLE_DEVICES=3 python3 -m fairseq_cli.preprocess \
      --source-lang {SRC} \
      --target-lang {TGT} \
      --trainpref {RAW_DIR}/train_mix.spm \
      --validpref {RAW_DIR}/valid_mix.spm \
      --testpref {RAW_DIR}/test_culture_mix.spm,{RAW_DIR}/test_general_mix.spm \
      --destdir {BIN_DIR} \
      --srcdict model/fairseq/data-bin/ht-en/dict.ht.txt \
      --tgtdict model/fairseq/data-bin/ht-en/dict.en.txt \
      --workers 8 
    """

    run_cmd(cmd)


def train_transformer():
    cmd = f"""
    CUDA_VISIBLE_DEVICES=3 python3 -m fairseq_cli.train {BIN_DIR} \
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
      --lr 0.0001 \
      --lr-scheduler inverse_sqrt \
      --warmup-updates 5000 \
      --max-tokens 2048 \
      --update-freq 2 \
      --max-epoch 10 \
      --patience 3 \
      --restore-file {BASELINE} \
      --reset-optimizer \
      --reset-dataloader \
      --reset-meters \
      --reset-lr-scheduler \
      --save-dir {CKPT_DIR} \
      --keep-best-checkpoints 1 \
      --no-epoch-checkpoints \
      --best-checkpoint-metric loss 
    """

    run_cmd(cmd)


def generate_translations():
    output_file = [ OUT_DIR / "outputs_transformer_base_Culture.txt", OUT_DIR / "outputs_transformer_base_General.txt" ]
    test_subset = [ "test", "test1" ]
    
    for file, subset in zip(output_file, test_subset):
        cmd = f"""
        CUDA_VISIBLE_DEVICES="" python3 -m fairseq_cli.generate {BIN_DIR} \
        --source-lang {SRC} \
        --target-lang {TGT} \
        --path {CKPT_DIR}/checkpoint_best.pt \
        --gen-subset {subset} \
        --beam 5 \
        --batch-size 64 \
        --remove-bpe=sentencepiece \
        --cpu \
        > {file}
        """

        run_cmd(cmd)


def extract_predictions():
    fairseq_output = [OUT_DIR / "outputs_transformer_base_Culture.txt", OUT_DIR / "outputs_transformer_base_General.txt"]
    prediction_file = [OUT_DIR / "pred_transformer_base_Culture.en", OUT_DIR / "pred_transformer_base_General.en"]

    for output, pred in zip(fairseq_output, prediction_file):
        cmd = f"""
        grep '^H-' {output} \
          | sort -V \
          | cut -f3- \
          > {pred}
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
        OUT_DIR / "pred_transformer_base_Culture.en",
        OUT_DIR / "pred_transformer_base_General.en"
    ]

    reference_files = [
        RAW_DIR / f"test_culture_mix.{TGT}",
        RAW_DIR / f"test_general_mix.{TGT}"
    ]

    test_names = [
        "Cultural test",
        "General test"
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

    score_path = OUT_DIR / "cultural_MIX_test_scores_all.csv"
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
        general_preds = read_lines(general_pred_file)
        cultural_preds = read_lines(cultural_pred_file)

        n = len(refs)

        assert len(sources) == n, f"sources={len(sources)} refs={n}"
        assert len(general_preds) == n, f"general={len(general_preds)} refs={n}"
        assert len(cultural_preds) == n, f"cultural={len(cultural_preds)} refs={n}"

        df_results = pd.DataFrame({
            "id": ids,
            "cr": sources,
            "reference_en": refs,
            "general_prediction": general_preds,
            "cultural_prediction": cultural_preds
        })

        output_path = OUT_DIR / "cultural_MIX_test_comparison_all.csv"
        df_results.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"Comparaison sauvegardée : {output_path}")



if __name__ == "__main__":
    check_files()

    # if not (SPM_DIR / "ht_en_spm.model").exists():
    train_sentencepiece()
    # else:
    #     print("SentencePiece model already exists. Skipping training.")

    # if not (RAW_DIR / f"train_mix.spm.{SRC}").exists():
    apply_sentencepiece()
    # else:
    #     print("SentencePiece files already exist. Skipping encoding.")


    # if not (BIN_DIR / f"dict.{SRC}.txt").exists():
    fairseq_preprocess()
    # else:
    #    print("Fairseq binary data already exists. Skipping preprocess.")
    
    # if not (CKPT_DIR / "checkpoint_best.pt").exists():
    train_transformer()
    # else:      
    #    print("Checkpoint already exists. Skipping training.")

    # if not (OUT_DIR / "outputs_transformer_base_Culture.txt").exists() and not (OUT_DIR / "outputs_transformer_base_General.txt").exists():
    generate_translations()
    # else:       
        #  print("Translations already generated. Skipping generation.")

    # if not (OUT_DIR / "pred_transformer_base_Culture.en").exists() and not (OUT_DIR / "pred_transformer_base_General.en").exists():
    extract_predictions()
    # else:       
        # print("Predictions already extracted. Skipping extraction.")

    evaluate()

    save_cultural_comparison()