import json
from pathlib import Path


# =========================
# CONFIGURATION
# =========================

TESTSRC = Path("uqam_eval_srcs/hat-eng.hat")
TESTREF = Path("uqam_eval_srcs/eval.generale.uqam.googletranslate.hat-eng.hat.txt")
TESTSRCFR = Path("uqam_eval_srcs/hat-fra.hat")
TESTREFFR = Path("uqam_eval_srcs/eval.generale.uqam.googletranslate.hat-fra.hat.txt")


OUTPUT_FILE = Path("uqam_eval_srcs/hat_eng.jsonl")
OUTPUT_FILE_FR = Path("uqam_eval_srcs/hat_fra.jsonl")

SOURCE_LANGUAGE = "hat"
TARGET_LANGUAGE_FR = "fra"
TARGET_LANGUAGE = "eng"


# =========================
# CONVERSION
# =========================

def convert_aligned_files_to_json(
    source_file,
    target_file,
    output_file,
    source_language,
    target_language
):
    source_file = Path(source_file)
    target_file = Path(target_file)
    output_file = Path(output_file)

    if not source_file.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {source_file}"
        )

    if not target_file.exists():
        raise FileNotFoundError(
            f"Fichier cible introuvable : {target_file}"
        )

    with source_file.open(
        "r",
        encoding="utf-8"
    ) as source_stream:
        source_lines = [
            line.strip()
            for line in source_stream
        ]

    with target_file.open(
        "r",
        encoding="utf-8"
    ) as target_stream:
        target_lines = [
            line.strip()
            for line in target_stream
        ]

    if len(source_lines) != len(target_lines):
        raise ValueError(
            "Les fichiers ne sont pas correctement alignés :\n"
            f"- Nombre de lignes source : {len(source_lines)}\n"
            f"- Nombre de lignes cible : {len(target_lines)}"
        )

    dataset = []
    empty_pairs = 0

    for line_number, (source_text, target_text) in enumerate(
        zip(source_lines, target_lines),
        start=1
    ):
        # Ignorer les paires entièrement vides
        if not source_text and not target_text:
            empty_pairs += 1
            continue

        # Signaler les lignes partiellement vides
        if not source_text or not target_text:
            print(
                f"Attention — ligne {line_number} partiellement vide : "
                f"source={source_text!r}, cible={target_text!r}"
            )

        example = {
            "translation": {
                "src_lang": source_language,
                "src_text": source_text,
                "tgt_lang": target_language,
                "tgt_text": target_text
            }
        }

        dataset.append(example)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as output_stream:
        json.dump(
            dataset,
            output_stream,
            ensure_ascii=False,
            indent=2
        )

    print(f"Conversion terminée.")
    print(f"Paires enregistrées : {len(dataset)}")
    print(f"Paires vides ignorées : {empty_pairs}")
    print(f"Fichier créé : {output_file}")


if __name__ == "__main__":

    for SOURCE_FILE, TARGET_FILE, OUTPUT_FILE, TARGET_LANGUAGE in [TESTSRC, TESTREF, OUTPUT_FILE, TARGET_LANGUAGE], [TESTSRCFR, TESTREFFR, OUTPUT_FILE_FR, TARGET_LANGUAGE_FR]:
        convert_aligned_files_to_json(
            source_file=SOURCE_FILE,
            target_file=TARGET_FILE,
            output_file=OUTPUT_FILE,
            source_language=SOURCE_LANGUAGE,
            target_language=TARGET_LANGUAGE
        )