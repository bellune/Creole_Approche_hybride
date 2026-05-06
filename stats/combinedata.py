from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path("datasets/corpus_culturel/all_data")
OUTPUT_DIR = Path("datasets/corpus_culturel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_lines_keep_empty(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n").strip() for line in f]


def pad_lines(lines, max_len):
    return lines + [""] * (max_len - len(lines))


rows = []

# ============================
# 1. Lire le dossier trilingue
# ============================

tri_dir = BASE_DIR / "trilingue"
bi1_dir = BASE_DIR / "bilingue/cr_en"
bi2_dir = BASE_DIR / "bilingue/cr_fr"


for one_dir in [tri_dir, bi1_dir, bi2_dir]:

    if not one_dir.exists():
        print(f"Erreur : le dossier {one_dir} n'existe pas.")
        exit(1)

    for cr_file in sorted(one_dir.glob("*_cr.txt")):

        base_name = cr_file.name.replace("_cr.txt", "")

        en_file = one_dir / f"{base_name}_en.txt"
        fr_file = one_dir / f"{base_name}_fr.txt"

        if not en_file.exists() and not fr_file.exists():
            print(f"Aucune traduction trouvée pour {base_name}")
            continue

        cr_lines = read_lines_keep_empty(cr_file)

        if en_file.exists():
            en_lines = read_lines_keep_empty(en_file)
        else:
            en_lines = []

        if fr_file.exists():
            fr_lines = read_lines_keep_empty(fr_file)
        else:
            fr_lines = []

        max_len = max(len(cr_lines), len(en_lines), len(fr_lines))

        cr_lines = pad_lines(cr_lines, max_len)
        en_lines = pad_lines(en_lines, max_len)
        fr_lines = pad_lines(fr_lines, max_len)

        for i, (cr, en, fr) in enumerate(
            zip(cr_lines, en_lines, fr_lines),
            start=1
        ):

            if not cr:
                continue

            rows.append({
                "source": base_name,
                "line_id": i,
                "cr": cr,
                "en": en,
                "fr": fr
            })


df = pd.DataFrame(rows)

print("Total lignes trilingues lues :", len(df))
print("CR-EN disponibles :", ((df["cr"] != "") & (df["en"] != "")).sum())
print("CR-FR disponibles :", ((df["cr"] != "") & (df["fr"] != "")).sum())


# ============================
# 2. Créer dataset CR -> EN
# ============================

cr_en_rows = []

for _, row in df.iterrows():
    cr = row["cr"].strip()
    en = row["en"].strip()

    if cr and en:
        cr_en_rows.append({
            "id": f"{row['source']}_{row['line_id']}_hat_eng",
            "translation": {
                "src_lang": "hat_Latn",
                "src_text": cr,
                "tgt_lang": "eng_Latn",
                "tgt_text": en
            },
            "source": row["source"],
            "direction": "hat-eng"
        })

cr_en_df = pd.DataFrame(cr_en_rows)


# ============================
# 3. Créer dataset CR -> FR
# ============================

cr_fr_rows = []

for _, row in df.iterrows():
    cr = row["cr"].strip()
    fr = row["fr"].strip()

    if cr and fr:
        cr_fr_rows.append({
            "id": f"{row['source']}_{row['line_id']}_hat_fra",
            "translation": {
                "src_lang": "hat_Latn",
                "src_text": cr,
                "tgt_lang": "fra_Latn",
                "tgt_text": fr
            },
            "source": row["source"],
            "direction": "hat-fra"
        })

cr_fr_df = pd.DataFrame(cr_fr_rows)


# ============================
# 4. Fonction split train/dev/test
# ============================

def split_dataset(dataframe, name):
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

    print(f"\n{name}")
    print("Train :", len(train_df))
    print("Dev   :", len(dev_df))
    print("Test  :", len(test_df))

    train_df.to_json(
        OUTPUT_DIR / f"train/{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    dev_df.to_json(
        OUTPUT_DIR / f"dev/{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )

    test_df.to_json(
        OUTPUT_DIR / f"test/{name}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False
    )


# ============================
# 5. Sauvegarder les deux datasets
# ============================

split_dataset(cr_en_df, "cr_en")
# split_dataset(cr_fr_df, "cr_fr")

print("\nFichiers créés dans :", OUTPUT_DIR)