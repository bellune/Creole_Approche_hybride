from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


FAIRSEQ_OUTPUT =[ Path("results/fairseq/outputs/outputs_attention.txt"), 
                 Path("results/mix_fairseq/outputs/outputs_mix_cultural_attention.txt") ]
OUT_DIR = [Path("results/fairseq/attention_plots"), 
           Path("results/mix_fairseq/attention_plots_2")]
for dir in OUT_DIR:
    dir.mkdir(parents=True, exist_ok=True)

EXAMPLE_ID = 0  # change: 0, 1, 2, 10, etc.


def parse_fairseq_output(path):
    records = defaultdict(dict)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line or len(line) < 3 or line[1] != "-":
                continue

            kind = line[0]
            parts = line[2:].split("\t")

            if len(parts) < 2:
                continue

            try:
                idx = int(parts[0])
            except ValueError:
                continue

            if kind == "S":
                records[idx]["source"] = parts[-1]
            elif kind == "T":
                records[idx]["reference"] = parts[-1]
            elif kind == "H":
                records[idx]["hypothesis"] = parts[-1]
            elif kind == "A":
                records[idx]["alignment"] = parts[-1]

    return records


def clean_token(token):
    token = token.replace("▁", "")
    token = token.replace("@@ ", "")
    token = token.strip()
    return token if token else "_"


def build_alignment_matrix(source_tokens, target_tokens, alignment_string):
    matrix = np.zeros((len(target_tokens), len(source_tokens)))

    for pair in alignment_string.split():
        if "-" not in pair:
            continue

        src_idx, tgt_idx = pair.split("-")

        try:
            src_idx = int(src_idx)
            tgt_idx = int(tgt_idx)
        except ValueError:
            continue

        if src_idx < len(source_tokens) and tgt_idx < len(target_tokens):
            matrix[tgt_idx, src_idx] = 1.0

    return matrix


def plot_heatmap(example_id):

    for path, out_dir in zip(FAIRSEQ_OUTPUT, OUT_DIR):
        records = parse_fairseq_output(path)

        if example_id not in records:
            raise ValueError(f"Example {example_id} not found.")

        rec = records[example_id]

        source = rec.get("source")
        hypothesis = rec.get("hypothesis")
        alignment = rec.get("alignment")

        if source is None or hypothesis is None or alignment is None:
            raise ValueError(
                f"Missing source/hypothesis/alignment for example {example_id}."
            )

        source_tokens = source.split()
        target_tokens = hypothesis.split()

        matrix = build_alignment_matrix(source_tokens, target_tokens, alignment)

        source_labels = [clean_token(tok) for tok in source_tokens]
        target_labels = [clean_token(tok) for tok in target_tokens]

        fig_width = max(8, len(source_labels) * 0.6)
        fig_height = max(6, len(target_labels) * 0.45)

        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(matrix, aspect="auto", interpolation="nearest")

        plt.xticks(
            ticks=np.arange(len(source_labels)),
            labels=source_labels,
            rotation=90,
            fontsize=11
        )

        plt.yticks(
            ticks=np.arange(len(target_labels)),
            labels=target_labels,
            fontsize=11
        )

        plt.xlabel("Tokens sources")
        plt.ylabel("Tokens générés")
        plt.title(f"Alignement attentionnel — exemple {example_id}")

        plt.colorbar(label="Poids / alignement")
        plt.tight_layout()

        output_png = out_dir / f"attention_heatmap_example_{example_id}.png"
        output_pdf = out_dir / f"attention_heatmap_example_{example_id}.pdf"

        plt.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.savefig(output_pdf, bbox_inches="tight")

        print(f"PNG sauvegardé : {output_png}")
        print(f"PDF sauvegardé : {output_pdf}")


if __name__ == "__main__":
    plot_heatmap(EXAMPLE_ID)