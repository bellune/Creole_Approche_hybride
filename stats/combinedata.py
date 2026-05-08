from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path("datasets/corpus_culturel/all_data")
OUTPUT_DIR = Path("datasets/corpus_culturel")

for split in ["train", "dev", "test"]:
    (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def read_lines_keep_empty(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]    
    
def build_dataset(input_dirs, src_suffix, tgt_suffix,
                  src_lang, tgt_lang, direction):

    rows = []

    print(f"\n==============================")
    print(f"Construction dataset {direction}")
    print(f"==============================")

    for one_dir in input_dirs:

        if not one_dir.exists():
            print(f"Attention : {one_dir} n'existe pas.")
            continue

        print(f"\nLecture dossier : {one_dir}")

        for src_file in sorted(one_dir.glob(f"*_{src_suffix}.txt")):

            base_name = src_file.name.replace(
                f"_{src_suffix}.txt", ""
            )

            tgt_file = one_dir / f"{base_name}_{tgt_suffix}.txt"

            if not tgt_file.exists():
                print(f"Traduction manquante : {base_name}")
                continue

            src_lines = read_lines_keep_empty(src_file)
            tgt_lines = read_lines_keep_empty(tgt_file)

            print(f"\nFichier : {base_name}")
            print(f"Lignes source      : {len(src_lines)}")
            print(f"Lignes traduction  : {len(tgt_lines)}")

            max_len = max(len(src_lines), len(tgt_lines))

            src_lines = src_lines + [""] * (max_len - len(src_lines))
            tgt_lines = tgt_lines + [""] * (max_len - len(tgt_lines))

            for i, (src_text, tgt_text) in enumerate(
                    zip(src_lines, tgt_lines),
                    start=1
            ):
                
                src_text = src_text.strip()
                tgt_text = tgt_text.strip()

                if not src_text or not tgt_text:
                    continue

                rows.append({
                    "id": f"{base_name}_{i+1}_{direction.replace('-', '_')}",
                    "translation": {
                        "src_lang": src_lang,
                        "src_text": src_text,
                        "tgt_lang": tgt_lang,
                        "tgt_text": tgt_text
                    },
                    "source": base_name,
                    "line_id": i + 1,
                    "direction": direction
                })

    df = pd.DataFrame(rows)

    print(f"\nTOTAL {direction} : {len(df)} paires")

    return df


# ============================
# Split train/dev/test
# ============================

def split_train_dev_test(dataframe, name):

    print(f"\n==============================")
    print(f"SPLIT {name}")
    print(f"==============================")

    train_df, temp_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    dev_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    print(f"Taille totale : {len(dataframe)}")
    print(f"Train : {len(train_df)}")
    print(f"Dev   : {len(dev_df)}")
    print(f"Test  : {len(test_df)}")

    train_df.to_json(
        OUTPUT_DIR / "train" / f"{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    dev_df.to_json(
        OUTPUT_DIR / "dev" / f"{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    test_df.to_json(
        OUTPUT_DIR / "test" / f"{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )


# ============================
# Split train/dev seulement
# ============================

def split_train_dev(dataframe, name):

    print(f"\n==============================")
    print(f"SPLIT {name}")
    print(f"==============================")

    train_df, dev_df = train_test_split(
        dataframe,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    print(f"Taille totale : {len(dataframe)}")
    print(f"Train : {len(train_df)}")
    print(f"Dev   : {len(dev_df)}")

    train_df.to_json(
        OUTPUT_DIR / "train" / f"{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    dev_df.to_json(
        OUTPUT_DIR / "dev" / f"{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )


# ==================================================
# 1. DATASET CR -> EN
# ==================================================

cr_en_dirs = [
    BASE_DIR / "trilingue",
    BASE_DIR / "bilingue/cr_en"
]

cr_en_df = build_dataset(
    input_dirs=cr_en_dirs,
    src_suffix="cr",
    tgt_suffix="en",
    src_lang="hat_Latn",
    tgt_lang="eng_Latn",
    direction="hat-eng"
)

split_train_dev_test(cr_en_df, "cr_en")


# ==================================================
# 2. DATASET CR -> FR
# ==================================================

cr_fr_dirs = [
    BASE_DIR / "trilingue",
    BASE_DIR / "bilingue/cr_fr"
]

cr_fr_df = build_dataset(
    input_dirs=cr_fr_dirs,
    src_suffix="cr",
    tgt_suffix="fr",
    src_lang="hat_Latn",
    tgt_lang="fra_Latn",
    direction="hat-fra"
)

# PAS DE TEST
split_train_dev(cr_fr_df, "cr_fr")


print("\n==============================")
print("FIN DU TRAITEMENT")
print("==============================")
print(f"Fichiers sauvegardés dans : {OUTPUT_DIR}")