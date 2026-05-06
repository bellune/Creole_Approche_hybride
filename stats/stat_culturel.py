import glob, os
from typing import Iterable, Tuple


# -------------------------------
# Stats bilingue (CR–FR)
# -------------------------------

CR_GLOB = "datasets/corpus_culturel/all_data/bilingue/cr_fr/*_cr.txt"
FR_GLOB = "datasets/corpus_culturel/all_data/bilingue/cr_fr/*_fr.txt"

cr_files = sorted(glob.glob(CR_GLOB))
fr_files = sorted(glob.glob(FR_GLOB))

def base_id(path, suffix):
    return os.path.basename(path).replace(suffix, "")

fr_map = {base_id(p, "_fr.txt"): p for p in fr_files}

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f]

pairs_bi1 = []
missing_fr_docs = []

for cr_path in cr_files:
    base = base_id(cr_path, "_cr.txt")
    fr_path = fr_map.get(base)

    if not fr_path:
        missing_fr_docs.append(base)
        continue

    cr_lines = read_lines(cr_path)
    fr_lines = read_lines(fr_path)

    n = min(len(cr_lines), len(fr_lines))  # alignement strict
    for i in range(n):
        cr = cr_lines[i].strip()
        fr = fr_lines[i].strip()
        if cr and fr:
            pairs_bi1.append((cr, fr))

print("Paires bilingues (CR–FR):", len(pairs_bi1))
print("Docs CR:", len(cr_files), "| Docs FR:", len(fr_files))
print("Docs CR sans FR match:", len(missing_fr_docs))
if missing_fr_docs[:5]:
    print("Exemples:", missing_fr_docs[:5])





# -------------------------------
# Stats bilingue (CR–EN)
# -------------------------------

CR_GLOB = "datasets/corpus_culturel/all_data/bilingue/cr_en/*_cr.txt"
EN_GLOB = "datasets/corpus_culturel/all_data/bilingue/cr_en/*_en.txt"

cr_files = sorted(glob.glob(CR_GLOB))
en_files = sorted(glob.glob(EN_GLOB))

def base_id(path, suffix):
    return os.path.basename(path).replace(suffix, "")

en_map = {base_id(p, "_en.txt"): p for p in en_files}

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f]

pairs_bi2 = []
missing_en_docs = []

for cr_path in cr_files:
    base = base_id(cr_path, "_cr.txt")
    en_path = en_map.get(base)

    if not en_path:
        missing_en_docs.append(base)
        continue

    cr_lines = read_lines(cr_path)
    en_lines = read_lines(en_path)

    n = min(len(cr_lines), len(en_lines))  # alignement strict
    for i in range(n):
        cr = cr_lines[i].strip()
        en = en_lines[i].strip()
        if cr and en:
            pairs_bi2.append((cr, en))

print("Paires bilingues (CR–EN):", len(pairs_bi2    ))
print("Docs CR:", len(cr_files), "| Docs EN:", len(en_files))
print("Docs CR sans EN match:", len(missing_en_docs))
if missing_en_docs[:5]:
    print("Exemples:", missing_en_docs[:5])



# -------------------------------
# Stats trilingue (CR–EN–FR)
# -------------------------------


CR_GLOB = "datasets/corpus_culturel/all_data/trilingue/*_cr.txt"
EN_GLOB = "datasets/corpus_culturel/all_data/trilingue/*_en.txt"
FR_GLOB = "datasets/corpus_culturel/all_data/trilingue/*_fr.txt"

cr_files = sorted(glob.glob(CR_GLOB))
en_files = sorted(glob.glob(EN_GLOB))
fr_files = sorted(glob.glob(FR_GLOB))

def base_id(path, suffix):
    return os.path.basename(path).replace(suffix, "")

en_map = {base_id(p, "_en.txt"): p for p in en_files}
fr_map = {base_id(p, "_fr.txt"): p for p in fr_files}

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f]

pairs_tri = []
missing = {"en": [], "fr": []}

for cr_path in cr_files:
    base = base_id(cr_path, "_cr.txt")
    en_path = en_map.get(base)
    fr_path = fr_map.get(base)

    if not en_path: missing["en"].append(base)
    if not fr_path: missing["fr"].append(base)
    if not en_path or not fr_path:
        continue

    cr_lines = read_lines(cr_path)
    en_lines = read_lines(en_path)
    fr_lines = read_lines(fr_path)

    n = min(len(cr_lines), len(en_lines), len(fr_lines))
    for i in range(n):
        cr = cr_lines[i].strip()
        en = en_lines[i].strip()
        fr = fr_lines[i].strip()
        if cr and en and fr:
            pairs_tri.append((cr, en, fr))

print("Triples trilingues (CR–EN–FR):", len(pairs_tri))
print("Docs CR:", len(cr_files), "| Docs EN:", len(en_files), "| Docs FR:", len(fr_files))
print("Docs CR sans EN match:", len(missing["en"]), "| sans FR match:", len(missing["fr"]))


# -------------------------------
# Statistiques détaillées
# -------------------------------

def _wc(s: str) -> int:
    # split() sans argument gère bien les espaces multiples
    return len(s.split()) if s else 0

def stats_trilingue(pairs_tri: Iterable[Tuple[str, str, str]]):
    n = 0
    cr_w = en_w = fr_w = 0

    for cr, en, fr in pairs_tri:
        n += 1
        cr_w += _wc(cr)
        en_w += _wc(en)
        fr_w += _wc(fr)

    return {
        "pairs": n,
        "cr_words": cr_w,
        "en_words": en_w,
        "fr_words": fr_w,
        "cr_avg": (cr_w / n) if n else 0.0,
        "en_avg": (en_w / n) if n else 0.0,
        "fr_avg": (fr_w / n) if n else 0.0,
    }

def stats_bilingue1(pairs_bi1: Iterable[Tuple[str, str]]):
    n = 0
    cr_w = en_w = 0

    for cr, en in pairs_bi1:
        n += 1
        cr_w += _wc(cr)
        en_w += _wc(en)

    return {
        "pairs": n,
        "cr_words": cr_w,
        "fr_words": en_w,
        "cr_avg": (cr_w / n) if n else 0.0,
        "fr_avg": (en_w / n) if n else 0.0,
    }


def stats_bilingue2(pairs_bi2: Iterable[Tuple[str, str]]):
    n = 0
    cr_w = en_w = 0

    for cr, en in pairs_bi2:
        n += 1
        cr_w += _wc(cr)
        en_w += _wc(en)

    return {
        "pairs": n,
        "cr_words": cr_w,
        "en_words": en_w,
        "cr_avg": (cr_w / n) if n else 0.0,
        "en_avg": (en_w / n) if n else 0.0,
    }

def pretty_print_tri(name: str, s: dict):
    print(f"\n {name}")
    print("Paires :", s["pairs"])
    print("Mots CR :", s["cr_words"])
    print("Mots EN :", s["en_words"])
    print("Mots FR :", s["fr_words"])
    print("Moy. mots/phrase CR :", round(s["cr_avg"], 2))
    print("Moy. mots/phrase EN :", round(s["en_avg"], 2))
    print("Moy. mots/phrase FR :", round(s["fr_avg"], 2))

def pretty_print_bi1(name: str, s: dict):
    print(f"\n {name}")
    print("Paires :", s["pairs"])
    print("Mots CR :", s["cr_words"])
    print("Mots FR :", s["fr_words"])
    print("Moy. mots/phrase CR :", round(s["cr_avg"], 2))
    print("Moy. mots/phrase FR :", round(s["fr_avg"], 2))

def pretty_print_bi2(name: str, s: dict):
    print(f"\n {name}")
    print("Paires :", s["pairs"])
    print("Mots CR :", s["cr_words"])
    print("Mots EN :", s["en_words"])
    print("Moy. mots/phrase CR :", round(s["cr_avg"], 2))
    print("Moy. mots/phrase EN :", round(s["en_avg"], 2))

# --- Utilisation ---
tri_stats = stats_trilingue(pairs_tri)  # (cr,en,fr)
bi1_stats = stats_bilingue1(pairs_bi1)    # (cr,fr)
bi2_stats = stats_bilingue2(pairs_bi2)   # (cr,en)

pretty_print_tri("TRILINGUE (CR–EN–FR)", tri_stats)
pretty_print_bi1("BILINGUE (CR–FR)", bi1_stats)
pretty_print_bi2("BILINGUE (CR–EN)", bi2_stats)

print("\n Check total paires :", tri_stats["pairs"] + bi1_stats["pairs"] + bi2_stats["pairs"])
print("   (tri =", tri_stats["pairs"], ", bi1 =", bi1_stats["pairs"], ", bi2 =", bi2_stats["pairs"], ")")
