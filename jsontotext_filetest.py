import json
from pathlib import Path
import shutil

# Fichier test unique
TEST_JSON = Path("datasets/corpus_culturel/test/cr_en_all.jsonl")

# Les 3 dossiers Fairseq
FOLDERS = [
    Path("datasets/kreyol-mt-hat-eng/mt-fairseq"),   # E1
    Path("datasets/mix_corpus_all/fairseq"),         # E2
    Path("datasets/mixcs_corpus/fairseq"),          # E3
]

# Fichiers maîtres temporaires
master_ht = Path("/tmp/test.ht")
master_en = Path("/tmp/test.en")

with TEST_JSON.open("r", encoding="utf-8") as f, \
     master_ht.open("w", encoding="utf-8") as f_ht, \
     master_en.open("w", encoding="utf-8") as f_en:

    n = 0

    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)

        ht = row["translation"]["src_text"].strip()
        en = row["translation"]["tgt_text"].strip()

        f_ht.write(ht + "\n")
        f_en.write(en + "\n")

        n += 1

print(f"{n} paires extraites.")

# Copier EXACTEMENT les mêmes fichiers dans E1 / E2 / E3
for folder in FOLDERS:
    folder.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(master_ht, folder / "test.ht")
    shutil.copyfile(master_en, folder / "test.en")

    print("Copié vers:", folder)

print("TERMINÉ")