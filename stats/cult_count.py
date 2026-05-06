import glob, os

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



