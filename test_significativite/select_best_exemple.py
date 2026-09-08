from pathlib import Path

from sacrebleu.metrics import CHRF

chrf = CHRF(word_order=2)


def lire_fichier(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


# ============================================================
# FICHIERS
# ============================================================

sources = lire_fichier("datasets/kreyol-mt-hat-eng/mt-fairseq/test.ht")
refs = lire_fichier("datasets/kreyol-mt-hat-eng/mt-fairseq/test.en")

# Transformer
e1 = lire_fichier("results/fairseq/outputs/pred_transformer_base.en")
e2 = lire_fichier("results/mix_fairseq/outputs/pred_transformer_base.en")
e3 = lire_fichier("results/cs_fairseq/outputs/pred_transformer_base.en")

# NLLB
e4 = lire_fichier("test_significativite/data_pred_nllb/pred_nllb_E1.en")
e5 = lire_fichier("test_significativite/data_pred_nllb/pred_nllb_E2.en")
e6 = lire_fichier("test_significativite/data_pred_nllb/pred_nllb_E3.en")


assert len(sources) == len(refs) == len(e1) == len(e2) == len(e3) \
       == len(e4) == len(e5) == len(e6), \
       "Tous les fichiers doivent avoir le même nombre de lignes."


# ============================================================
# SCORES PAR PHRASE
# ============================================================

resultats = []

for i, (src, ref, h1, h2, h3, h4, h5, h6) in enumerate(
    zip(sources, refs, e1, e2, e3, e4, e5, e6)
):

    s1 = chrf.sentence_score(h1, [ref]).score
    s2 = chrf.sentence_score(h2, [ref]).score
    s3 = chrf.sentence_score(h3, [ref]).score

    s4 = chrf.sentence_score(h4, [ref]).score
    s5 = chrf.sentence_score(h5, [ref]).score
    s6 = chrf.sentence_score(h6, [ref]).score

    resultats.append({
        "index": i,
        "source": src,
        "reference": ref,

        "E1": h1,
        "E2": h2,
        "E3": h3,
        "E4": h4,
        "E5": h5,
        "E6": h6,

        "score_E1": s1,
        "score_E2": s2,
        "score_E3": s3,
        "score_E4": s4,
        "score_E5": s5,
        "score_E6": s6,

        # Gains Transformer
        "culture_T": s2 - s1,
        "codeswitch_T": s3 - s2,
        "global_T": s3 - s1,

        # Gains NLLB
        "culture_N": s5 - s4,
        "codeswitch_N": s6 - s5,
        "global_N": s6 - s4,

        # Effet commun aux deux architectures
        "culture_total": (s2 - s1) + (s5 - s4),
        "codeswitch_total": (s3 - s2) + (s6 - s5),
        "global_total": (s3 - s1) + (s6 - s4),

        # Comparaison finale NLLB vs Transformer
        "ecart_final_NLLB": s6 - s3,
    })


# ============================================================
# SÉLECTION DES TROIS EXEMPLES
# ============================================================

selection = []
utilises = set()

# ============================================================
# CRITÈRE DE LONGUEUR
# ============================================================

MIN_WORDS = 6
MAX_WORDS = 40


def longueur_phrase(phrase):
    return len(phrase.split())


def longueur_valide(phrase):
    n = longueur_phrase(phrase)
    return MIN_WORDS <= n <= MAX_WORDS
# ------------------------------------------------------------
# 1. Progression complète dans LES DEUX architectures
# E1 < E2 < E3 ET E4 < E5 < E6
# ------------------------------------------------------------

progression = [
    r for r in resultats
    if r["score_E1"] < r["score_E2"] < r["score_E3"]
    and r["score_E4"] < r["score_E5"] < r["score_E6"]
    and longueur_valide(r["source"])
]

progression.sort(
    key=lambda r: r["global_total"],
    reverse=True
)

if progression:
    r = progression[0]
    r["type_exemple"] = "Progression complète des deux architectures"
    selection.append(r)
    utilises.add(r["index"])


# ------------------------------------------------------------
# 2. Meilleur effet culturel dans LES DEUX architectures
# ------------------------------------------------------------

culture = [
    r for r in resultats
    if r["index"] not in utilises
    and r["culture_T"] > 0
    and r["culture_N"] > 0
    and longueur_valide(r["source"])
]

culture.sort(
    key=lambda r: r["culture_total"],
    reverse=True
)

if culture:
    r = culture[0]
    r["type_exemple"] = "Apport des données culturelles"
    selection.append(r)
    utilises.add(r["index"])


# ------------------------------------------------------------
# 3. Meilleur effet de l'alternance codique
# dans LES DEUX architectures
# ------------------------------------------------------------

codeswitch = [
    r for r in resultats
    if r["index"] not in utilises
    and r["codeswitch_T"] > 0
    and r["codeswitch_N"] > 0
    and longueur_valide(r["source"])
]

codeswitch.sort(
    key=lambda r: r["codeswitch_total"],
    reverse=True
)

if codeswitch:
    r = codeswitch[0]
    r["type_exemple"] = "Apport de l'alternance codique"
    selection.append(r)
    utilises.add(r["index"])


# ============================================================
# FALLBACK
# Si aucune progression parfaite n'existe
# ============================================================

if len(selection) < 3:

    restants = [
        r for r in resultats
        if r["index"] not in utilises and longueur_valide(r["source"])
    ]

    restants.sort(
        key=lambda r: r["global_total"],
        reverse=True
    )

    for r in restants:
        if len(selection) == 3:
            break

        r["type_exemple"] = "Gain global important"
        selection.append(r)
        utilises.add(r["index"])


# ============================================================
# AFFICHAGE
# ============================================================

for numero, r in enumerate(selection, start=1):

    print("\n" + "=" * 100)
    print(f"EXEMPLE {numero}")
    print(f"TYPE : {r['type_exemple']}")
    print(f"INDEX : {r['index']}")
    print("=" * 100)

    print("\nSOURCE HT :")
    print(r["source"])

    print("\nRÉFÉRENCE EN :")
    print(r["reference"])

    print("\n--- TRANSFORMER ---")

    print(f"\nE1 - Référence [{r['score_E1']:.2f}]")
    print(r["E1"])

    print(f"\nE2 - Culturel [{r['score_E2']:.2f}]")
    print(r["E2"])

    print(f"\nE3 - Alternance codique [{r['score_E3']:.2f}]")
    print(r["E3"])

    print("\n--- NLLB-200 ---")

    print(f"\nE4 - Référence [{r['score_E4']:.2f}]")
    print(r["E4"])

    print(f"\nE5 - Culturel [{r['score_E5']:.2f}]")
    print(r["E5"])

    print(f"\nE6 - Alternance codique [{r['score_E6']:.2f}]")
    print(r["E6"])

    print("\n--- GAINS TRANSFORMER ---")
    print(f"Culturel E2-E1 : {r['culture_T']:+.2f}")
    print(f"Alternance E3-E2 : {r['codeswitch_T']:+.2f}")
    print(f"Global E3-E1 : {r['global_T']:+.2f}")

    print("\n--- GAINS NLLB-200 ---")
    print(f"Culturel E5-E4 : {r['culture_N']:+.2f}")
    print(f"Alternance E6-E5 : {r['codeswitch_N']:+.2f}")
    print(f"Global E6-E4 : {r['global_N']:+.2f}")

    print("\n--- COMPARAISON FINALE ---")
    print(
        f"NLLB E6 - Transformer E3 : "
        f"{r['ecart_final_NLLB']:+.2f}"
    )