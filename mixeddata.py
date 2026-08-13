import os
import json
import random
from datasets import load_from_disk, load_dataset

random.seed(42)

# ============================================================
# 1. Chemins
# ============================================================

path_data = "datasets"

# Corpus général déjà sauvegardé au format Hugging Face Dataset
save_path = path_data + "/kreyol-mt-hat-eng"

# Corpus culturel en JSONL
TRAIN_FILE = "datasets/corpus_culturel/train/cr_en.jsonl"
DEV_FILE   = "datasets/corpus_culturel/dev/cr_en.jsonl"
TEST_FILE  = "datasets/corpus_culturel/test/cr_en.jsonl"

# Dossier de sortie
output_base = "datasets/mix_corpus_all"

json_dir = output_base + "/json"
fairseq_dir = output_base + "/fairseq"

os.makedirs(json_dir, exist_ok=True)
os.makedirs(fairseq_dir, exist_ok=True)

# ============================================================
# 2. Charger le corpus général
# ============================================================

general_ds = load_from_disk(save_path)

general_train = general_ds["train"]
general_val   = general_ds["validation"]
general_test  = general_ds["test"]

print("Corpus général :")
print(general_ds)
print("Exemple général :")
print(general_train[0])

# ============================================================
# 3. Charger le corpus culturel
# ============================================================

culture_ds = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": DEV_FILE,
        "test": TEST_FILE
    }
)

culture_train = culture_ds["train"]
culture_val   = culture_ds["validation"]
culture_test  = culture_ds["test"]

print("\nCorpus culturel :")
print(culture_ds)
print("Exemple culturel :")
print(culture_train[0])

# ============================================================
# 4. Extraire les paires créole-anglais
# ============================================================

def extract_pair(example):
      
    id = example.get("id")

    if "translation" in example:
        trans = example["translation"]

        src = (
            trans.get("src_text")
        )

        tgt = (
            trans.get("tgt_text")
    
        )

        return id, src, tgt

    src = (
        example.get("src_text")
    )

    tgt = (
        example.get("tgt_text")
    )

    return id, src, tgt


def dataset_to_pairs(dataset):
    pairs = []

    for example in dataset:
        id, src, tgt = extract_pair(example)

        if src is not None and tgt is not None:
            src = str(src).strip()
            tgt = str(tgt).strip()

            if src and tgt:
                pairs.append((id, src, tgt))

    return pairs

# Conversion en listes de paires
general_train_pairs = dataset_to_pairs(general_train)
general_val_pairs   = dataset_to_pairs(general_val)
general_test_pairs  = dataset_to_pairs(general_test)

culture_train_pairs = dataset_to_pairs(culture_train)
culture_val_pairs   = dataset_to_pairs(culture_val)
culture_test_pairs  = dataset_to_pairs(culture_test)

print("\nTailles après extraction :")
print("General train :", len(general_train_pairs))
print("General val   :", len(general_val_pairs))
print("General test  :", len(general_test_pairs))
print("Culture train :", len(culture_train_pairs))
print("Culture val   :", len(culture_val_pairs))
print("Culture test  :", len(culture_test_pairs))

# Vérification importante
if len(culture_train_pairs) == 0:
    raise ValueError("Aucune donnée culturelle train extraite. Vérifie les noms de colonnes du JSON culturel.")

if len(general_train_pairs) == 0:
    raise ValueError("Aucune donnée générale train extraite. Vérifie les noms de colonnes du corpus général.")

# ============================================================
# 5. Construire le corpus mixte
# ============================================================

def build_mixed_corpus(
    general_pairs,
    culture_pairs,
    culture_ratio=0.30,
    general_ratio=0.70,
    use_tags=True
):
    """
    Construit un corpus mixte contrôlé :

    - utilise 100% des données culturelles
    - ajoute un échantillon du corpus général
    - respecte le ratio choisi

    Exemple :
    culture_ratio = 0.30
    general_ratio = 0.70

    Si culture = 10 000 phrases :
    général ajouté = (0.70 / 0.30) * 10 000 = 23 333 phrases
    """

    culture_sample = culture_pairs
    n_culture = len(culture_sample)

    n_general = int((general_ratio / culture_ratio) * n_culture)

    print(f"Culture : {n_culture} phrases utilisées")
    print(f"Général : {n_general} phrases demandées")
    print(f"Général : {len(general_pairs)} phrases disponibles")

    if n_general <= len(general_pairs):
        general_sample = random.sample(general_pairs, n_general)
    else:
        general_sample = general_pairs
        print("Attention : le corpus général est plus petit que le nombre demandé.")

    mixed_data = []

    for id, src, tgt in culture_sample:
        if use_tags:
            src = f"{src}"
        mixed_data.append((id, src, tgt))

    for id, src, tgt in general_sample:
        if use_tags:
            src = f"{src}"
        mixed_data.append((id, src, tgt))

    random.shuffle(mixed_data)

    return mixed_data, len(culture_sample), len(general_sample)

# Ratio choisi : 30% culturel / 70% général
culture_ratio = 0.30
general_ratio = 0.70

train_mix, train_culture_used, train_general_used = build_mixed_corpus(
    general_pairs=general_train_pairs,
    culture_pairs=culture_train_pairs,
    culture_ratio=culture_ratio,
    general_ratio=general_ratio,
    use_tags=True
)

valid_mix, valid_culture_used, valid_general_used = build_mixed_corpus(
    general_pairs=general_val_pairs,
    culture_pairs=culture_val_pairs,
    culture_ratio=culture_ratio,
    general_ratio=general_ratio,
    use_tags=True
)

# Test culturel seulement
test_culture_tagged = [
    (id, f"{src}", tgt)
    for id, src, tgt in culture_test_pairs
]

# Optionnel : test général seulement
test_general_tagged = [
    (f"general_{i}", f"{src}", tgt)
    for i, (id, src, tgt) in enumerate(general_test_pairs)
]

print("\nCorpus mixte construit :")
print("Train mix total :", len(train_mix))
print("  Culture train utilisée :", train_culture_used)
print("  Général train utilisé  :", train_general_used)

print("Valid mix total :", len(valid_mix))
print("  Culture valid utilisée :", valid_culture_used)
print("  Général valid utilisé  :", valid_general_used)

print("Test culturel :", len(test_tagged))
# print("Test général  :", len(test_general_tagged))

# ============================================================
# 6. Sauvegarder en JSONL pour NLLB-200
# ============================================================


def save_jsonl(pairs, output_file, src_key="src_text", tgt_key="tgt_text"):
    """
    Format JSONL avec colonne translation :
    {"translation": {"cr": "...", "en": "..."}}
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for id, src, tgt in pairs:
            example = {
                "id": id,
                "translation": {
                    src_key: src,
                    tgt_key: tgt
                }
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

# Choisis le format attendu par ton code NLLB
# "flat" donne : {"cr": "...", "en": "..."}
# "translation" donne : {"translation": {"cr": "...", "en": "..."}}



save_jsonl(
    train_mix,
    f"{json_dir}/train_mix.jsonl"
)

save_jsonl(
    valid_mix,
    f"{json_dir}/valid_mix.jsonl"
)

save_jsonl(
    test_tagged,
    f"{json_dir}/test.jsonl"
)

# save_jsonl(
#     test_general_tagged,
#     f"{json_dir}/test.jsonl"
# )

# ============================================================
# 7. Sauvegarder en TXT parallèle pour Fairseq
# ============================================================

def save_fairseq_txt(pairs, src_file, tgt_file):
    """
    Format Fairseq :
    train_mix.ht
    train_mix.en
    """

    os.makedirs(os.path.dirname(src_file), exist_ok=True)
    os.makedirs(os.path.dirname(tgt_file), exist_ok=True)

    with open(src_file, "w", encoding="utf-8") as f_src, \
         open(tgt_file, "w", encoding="utf-8") as f_tgt:

        for id,src, tgt in pairs:
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")

save_fairseq_txt(
    train_mix,
    f"{fairseq_dir}/train_mix.ht",
    f"{fairseq_dir}/train_mix.en"
)

save_fairseq_txt(
    valid_mix,
    f"{fairseq_dir}/valid_mix.ht",
    f"{fairseq_dir}/valid_mix.en"
)

save_fairseq_txt(
    test_tagged,
    f"{fairseq_dir}/test_mix.ht",
    f"{fairseq_dir}/test_mix.en"
)

# save_fairseq_txt(
#     test_general_tagged,
#     f"{fairseq_dir}/test_general_mix.ht",
#     f"{fairseq_dir}/test_general_mix.en"
# )

# # ============================================================
# 8. Résumé final
# ============================================================

print("\nFichiers générés avec succès.")

print("\nJSONL pour NLLB-200 :")
print(f"{json_dir}/train_mix.jsonl")
print(f"{json_dir}/valid_mix.jsonl")
print(f"{json_dir}/test.jsonl")
# print(f"{json_dir}/test_general.jsonl")

print("\nTXT pour Fairseq :")
print(f"{fairseq_dir}/train_mix.ht")
print(f"{fairseq_dir}/train_mix.en")
print(f"{fairseq_dir}/valid_mix.ht")
print(f"{fairseq_dir}/valid_mix.en")
print(f"{fairseq_dir}/test_mix.ht")
print(f"{fairseq_dir}/test_mix.en")
# print(f"{fairseq_dir}/test_general_mix.ht")
# print(f"{fairseq_dir}/test_general_mix.en")