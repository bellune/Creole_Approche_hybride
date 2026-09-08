#!/usr/bin/env python3
"""
Tests de significativité pour les expériences Transformer / NLLB.

H1 : E2 vs E1 -> effet des données culturelles
H2 : E3 vs E2 -> effet du code-switching
H3 : comparaison du gain global (E3-E1) entre Transformer et NLLB

H1/H2 utilisent l'implémentation officielle du paired bootstrap resampling
fournie par SacreBLEU (PairedTest, test_type='bs').
H3 étend le même principe de rééchantillonnage apparié à une différence de gains.

Sortie : test_significativite/resultats/bootstrap_results.xlsx

Dépendances :
    pip install sacrebleu numpy xlsxwriter
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import sacrebleu
import xlsxwriter
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest


# ============================================================
# CONFIGURATION
# ============================================================

B = 1000
ALPHA = 0.05
SEED = 42

# Même fichier de référence pour TOUTES les expériences
REFERENCE = Path("datasets/kreyol-mt-hat-eng/mt-fairseq/test.en")

# Transformer
T_E1 = Path("results/fairseq/outputs/pred_transformer_base.en")
T_E2 = Path("results/mix_fairseq/outputs/pred_transformer_base.en")
T_E3 = Path("results/cs_fairseq/outputs/pred_transformer_base.en")

# NLLB
N_E1 = Path("test_significativite/data_pred_nllb/pred_nllb_E1.en")
N_E2 = Path("test_significativite/data_pred_nllb/pred_nllb_E2.en")
N_E3 = Path("test_significativite/data_pred_nllb/pred_nllb_E3.en")

OUTPUT_XLSX = Path("test_significativite/resultats/bootstrap_results.xlsx")

# Force SacreBLEU à utiliser le même seed pour les tests H1/H2.
os.environ["SACREBLEU_SEED"] = str(SEED)


# ============================================================
# UTILITAIRES
# ============================================================

def read_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n\r") for line in f]


def check_alignment(named_data: Dict[str, Sequence[str]]) -> int:
    """Vérifie que tous les fichiers ont exactement le même nombre de lignes."""
    lengths = {name: len(lines) for name, lines in named_data.items()}
    expected = lengths["Référence"]

    bad = {name: n for name, n in lengths.items() if n != expected}
    if bad:
        details = "\n".join(f"  - {name}: {n} lignes" for name, n in lengths.items())
        raise ValueError(
            "Les fichiers n'ont pas tous le même nombre de lignes.\n"
            "Le bootstrap apparié exige un alignement phrase par phrase.\n" + details
        )

    if expected == 0:
        raise ValueError("Le fichier de référence est vide.")

    return expected


def metric_label(name: str) -> str:
    if name.upper().startswith("BLEU"):
        return "BLEU"
    if name.lower().startswith("chrf"):
        return "chrF++"
    return name


def decision_h1_h2(diff: float, p_value: float) -> str:
    if p_value >= ALPHA:
        return "Non significatif"
    if diff > 0:
        return "Soutenue"
    if diff < 0:
        return "Non soutenue (baisse significative)"
    return "Non significatif"


def decision_h3(diff: float, p_value: float) -> str:
    if p_value >= ALPHA:
        return "Non significatif"
    if diff > 0:
        return "Soutenue"
    if diff < 0:
        return "Non soutenue (gain NLLB supérieur)"
    return "Non significatif"


# ============================================================
# H1 / H2 : SACREBLEU PAIRED BOOTSTRAP OFFICIEL
# ============================================================

def run_sacrebleu_paired(
    hypothesis: str,
    model: str,
    comparison: str,
    baseline_name: str,
    baseline: Sequence[str],
    candidate_name: str,
    candidate: Sequence[str],
    references: Sequence[str],
) -> List[dict]:
    """
    Lance le paired bootstrap officiel de SacreBLEU sur BLEU et chrF++.
    Le premier système est la baseline, le second est le système comparé.
    """

    metrics = {
        "BLEU": BLEU(),
        "chrF++": CHRF(word_order=2),
    }

    named_systems = [
        (baseline_name, baseline),
        (candidate_name, candidate),
    ]

    test = PairedTest(
        named_systems=named_systems,
        metrics=metrics,
        references=[references],
        test_type="bs",
        n_samples=B,
        n_jobs=1,
    )

    signatures, scores = test()
    rows: List[dict] = []

    for returned_metric_name, result_list in scores.items():
        if returned_metric_name == "System":
            continue

        baseline_result = result_list[0]
        candidate_result = result_list[1]

        before = float(baseline_result.score)
        after = float(candidate_result.score)
        diff = after - before
        p_value = float(candidate_result.p_value)

        # SacreBLEU renvoie mean +/- ci pour chaque système.
        bl_mean = baseline_result.mean
        bl_ci = baseline_result.ci
        cand_mean = candidate_result.mean
        cand_ci = candidate_result.ci

        rows.append(
            {
                "Hypothese": hypothesis,
                "Modele": model,
                "Comparaison": comparison,
                "Metrique": metric_label(returned_metric_name),
                "Score_avant": before,
                "Score_apres": after,
                "Difference": diff,
                "p_value": p_value,
                "Conclusion": decision_h1_h2(diff, p_value),
                "Methode": "SacreBLEU paired bootstrap",
                "Baseline_mean": bl_mean,
                "Baseline_IC95_bas": None if bl_mean is None or bl_ci is None else bl_mean - bl_ci,
                "Baseline_IC95_haut": None if bl_mean is None or bl_ci is None else bl_mean + bl_ci,
                "Systeme_mean": cand_mean,
                "Systeme_IC95_bas": None if cand_mean is None or cand_ci is None else cand_mean - cand_ci,
                "Systeme_IC95_haut": None if cand_mean is None or cand_ci is None else cand_mean + cand_ci,
                "Signature": str(signatures.get(returned_metric_name, "")),
            }
        )

    return rows


# ============================================================
# H3 : BOOTSTRAP APPARIÉ SUR LA DIFFÉRENCE DE GAINS
# ============================================================

def _stats_array(metric, hypotheses: Sequence[str], references: Sequence[str]) -> np.ndarray:
    """
    Extrait les statistiques suffisantes de SacreBLEU une seule fois.
    Les méthodes commencent par '_' car SacreBLEU ne fournit pas actuellement
    d'API publique directe pour une différence-de-différences (H3).
    """
    stats = metric._extract_corpus_statistics(hypotheses, [references])
    return np.asarray(stats, dtype=np.float32)


def _score_from_resampled_stats(metric, stats: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    scores = np.empty(len(idxs), dtype=np.float64)
    for b, sample_idx in enumerate(idxs):
        summed = stats[sample_idx].sum(axis=0)
        scores[b] = metric._compute_score_from_stats(summed).score
    return scores


def _observed_score(metric, hypotheses: Sequence[str], references: Sequence[str]) -> float:
    return float(metric.corpus_score(hypotheses, [references]).score)


def bootstrap_h3_metric(
    metric,
    metric_name: str,
    t_e1: Sequence[str],
    t_e3: Sequence[str],
    n_e1: Sequence[str],
    n_e3: Sequence[str],
    references: Sequence[str],
) -> dict:
    """
    H3 : [Transformer(E3)-Transformer(E1)] - [NLLB(E3)-NLLB(E1)].

    Le même jeu d'indices bootstrap est appliqué aux quatre systèmes afin de
    conserver l'appariement phrase par phrase.

    Pour H3, la p-value bilatérale est calculée à partir de la proportion de
    différences bootstrap de signe opposé à la différence observée.
    """

    n = len(references)
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(n, size=(B, n), replace=True)

    # Scores observés
    t1 = _observed_score(metric, t_e1, references)
    t3 = _observed_score(metric, t_e3, references)
    n1 = _observed_score(metric, n_e1, references)
    n3 = _observed_score(metric, n_e3, references)

    gain_t = t3 - t1
    gain_n = n3 - n1
    observed_diff = gain_t - gain_n

    # Statistiques par phrase, puis mêmes rééchantillonnages pour les 4 systèmes
    st_t1 = _stats_array(metric, t_e1, references)
    st_t3 = _stats_array(metric, t_e3, references)
    st_n1 = _stats_array(metric, n_e1, references)
    st_n3 = _stats_array(metric, n_e3, references)

    bs_t1 = _score_from_resampled_stats(metric, st_t1, idxs)
    bs_t3 = _score_from_resampled_stats(metric, st_t3, idxs)
    bs_n1 = _score_from_resampled_stats(metric, st_n1, idxs)
    bs_n3 = _score_from_resampled_stats(metric, st_n3, idxs)

    bootstrap_diffs = (bs_t3 - bs_t1) - (bs_n3 - bs_n1)

    # Intervalle percentile 95 % de la différence de gains
    ci_low, ci_high = np.percentile(bootstrap_diffs, [2.5, 97.5])

    # Test bilatéral bootstrap basé sur le signe de la différence.
    # Correction +1 pour éviter p=0 avec un nombre fini de rééchantillonnages.
    n_nonpos = int(np.sum(bootstrap_diffs <= 0))
    n_nonneg = int(np.sum(bootstrap_diffs >= 0))
    p_left = (n_nonpos + 1) / (B + 1)
    p_right = (n_nonneg + 1) / (B + 1)
    p_value = min(1.0, 2.0 * min(p_left, p_right))

    return {
        "Hypothese": "H3",
        "Modele": "Transformer vs NLLB-200",
        "Comparaison": "ΔT(E3-E1) - ΔN(E3-E1)",
        "Metrique": metric_name,
        "Score_avant": gain_n,   # gain NLLB
        "Score_apres": gain_t,   # gain Transformer
        "Difference": observed_diff,
        "p_value": p_value,
        "Conclusion": decision_h3(observed_diff, p_value),
        "Methode": "Bootstrap apparié sur différence de gains",
        "Baseline_mean": None,
        "Baseline_IC95_bas": None,
        "Baseline_IC95_haut": None,
        "Systeme_mean": float(np.mean(bootstrap_diffs)),
        "Systeme_IC95_bas": float(ci_low),
        "Systeme_IC95_haut": float(ci_high),
        "Signature": f"seed={SEED}|bs={B}|extension-H3",
    }


def run_h3(
    t_e1: Sequence[str],
    t_e3: Sequence[str],
    n_e1: Sequence[str],
    n_e3: Sequence[str],
    references: Sequence[str],
) -> List[dict]:
    return [
        bootstrap_h3_metric(
            BLEU(), "BLEU", t_e1, t_e3, n_e1, n_e3, references
        ),
        bootstrap_h3_metric(
            CHRF(word_order=2), "chrF++", t_e1, t_e3, n_e1, n_e3, references
        ),
    ]


# ============================================================
# EXCEL
# ============================================================

def write_excel(results: List[dict], n_sentences: int) -> None:
    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)

    workbook = xlsxwriter.Workbook(OUTPUT_XLSX)

    # Formats
    fmt_title = workbook.add_format({
        "bold": True, "font_size": 14, "align": "center", "valign": "vcenter",
        "bg_color": "#1F4E78", "font_color": "#FFFFFF"
    })
    fmt_header = workbook.add_format({
        "bold": True, "bg_color": "#D9EAF7", "border": 1, "align": "center",
        "valign": "vcenter", "text_wrap": True
    })
    fmt_text = workbook.add_format({"border": 1, "valign": "top"})
    fmt_num = workbook.add_format({"border": 1, "num_format": "0.0000"})
    fmt_p = workbook.add_format({"border": 1, "num_format": "0.000000"})
    fmt_good = workbook.add_format({"border": 1, "bg_color": "#E2F0D9"})
    fmt_bad = workbook.add_format({"border": 1, "bg_color": "#FCE4D6"})
    fmt_note = workbook.add_format({"text_wrap": True, "valign": "top"})

    # --------------------------------------------------------
    # Feuille 1 : tableau prêt à copier dans le mémoire
    # --------------------------------------------------------
    ws = workbook.add_worksheet("Tableau mémoire")
    ws.merge_range("A1:G1", "Résultats des tests de significativité statistique", fmt_title)

    headers = [
        "Hyp.", "Modèle", "Comparaison", "Métrique",
        "Différence", "p-value", "Conclusion"
    ]
    for col, h in enumerate(headers):
        ws.write(2, col, h, fmt_header)

    for row_idx, r in enumerate(results, start=3):
        ws.write(row_idx, 0, r["Hypothese"], fmt_text)
        ws.write(row_idx, 1, r["Modele"], fmt_text)
        ws.write(row_idx, 2, r["Comparaison"], fmt_text)
        ws.write(row_idx, 3, r["Metrique"], fmt_text)
        ws.write_number(row_idx, 4, r["Difference"], fmt_num)
        ws.write_number(row_idx, 5, r["p_value"], fmt_p)
        conclusion_fmt = fmt_good if r["Conclusion"] == "Soutenue" else fmt_bad
        ws.write(row_idx, 6, r["Conclusion"], conclusion_fmt)

    ws.freeze_panes(3, 0)
    ws.autofilter(2, 0, 2 + len(results), len(headers) - 1)
    ws.set_column("A:A", 8)
    ws.set_column("B:B", 25)
    ws.set_column("C:C", 30)
    ws.set_column("D:D", 12)
    ws.set_column("E:F", 14)
    ws.set_column("G:G", 38)

    # --------------------------------------------------------
    # Feuille 2 : détails pour l'analyse
    # --------------------------------------------------------
    wd = workbook.add_worksheet("Détails")
    detail_headers = [
        "Hypothèse", "Modèle", "Comparaison", "Métrique",
        "Score avant / gain NLLB", "Score après / gain Transformer",
        "Différence", "p-value", "Conclusion", "Méthode",
        "IC95 système - bas", "IC95 système - haut", "Signature SacreBLEU"
    ]
    for col, h in enumerate(detail_headers):
        wd.write(0, col, h, fmt_header)

    for row_idx, r in enumerate(results, start=1):
        vals = [
            r["Hypothese"], r["Modele"], r["Comparaison"], r["Metrique"],
            r["Score_avant"], r["Score_apres"], r["Difference"], r["p_value"],
            r["Conclusion"], r["Methode"], r["Systeme_IC95_bas"],
            r["Systeme_IC95_haut"], r["Signature"],
        ]
        for col, value in enumerate(vals):
            if isinstance(value, (int, float, np.floating)) and value is not None:
                wd.write_number(row_idx, col, float(value), fmt_p if col == 7 else fmt_num)
            elif value is None:
                wd.write_blank(row_idx, col, None, fmt_text)
            else:
                wd.write(row_idx, col, str(value), fmt_text)

    wd.freeze_panes(1, 0)
    wd.autofilter(0, 0, len(results), len(detail_headers) - 1)
    wd.set_column("A:A", 10)
    wd.set_column("B:B", 25)
    wd.set_column("C:C", 32)
    wd.set_column("D:D", 12)
    wd.set_column("E:H", 21)
    wd.set_column("I:I", 38)
    wd.set_column("J:J", 42)
    wd.set_column("K:L", 19)
    wd.set_column("M:M", 55)

    # --------------------------------------------------------
    # Feuille 3 : configuration / traçabilité
    # --------------------------------------------------------
    wc = workbook.add_worksheet("Configuration")
    wc.write("A1", "Paramètre", fmt_header)
    wc.write("B1", "Valeur", fmt_header)

    config_rows = [
        ("Nombre de phrases de test", n_sentences),
        ("Rééchantillonnages bootstrap", B),
        ("Seuil alpha", ALPHA),
        ("Seed", SEED),
        ("Version SacreBLEU", sacrebleu.__version__),
        ("Référence", str(REFERENCE)),
        ("Transformer E1", str(T_E1)),
        ("Transformer E2", str(T_E2)),
        ("Transformer E3", str(T_E3)),
        ("NLLB E1", str(N_E1)),
        ("NLLB E2", str(N_E2)),
        ("NLLB E3", str(N_E3)),
        ("H1", "E2 - E1 : effet des données culturelles"),
        ("H2", "E3 - E2 : effet du code-switching"),
        ("H3", "[Transformer(E3-E1)] - [NLLB(E3-E1)]"),
    ]

    for i, (key, value) in enumerate(config_rows, start=1):
        wc.write(i, 0, key, fmt_text)
        if isinstance(value, (int, float)):
            wc.write_number(i, 1, value, fmt_num)
        else:
            wc.write(i, 1, str(value), fmt_text)

    wc.write(len(config_rows) + 2, 0, "Note H3", fmt_header)
    wc.write(
        len(config_rows) + 2,
        1,
        "H3 n'est pas une comparaison baseline-système standard de SacreBLEU. "
        "Le script applique le même rééchantillonnage phrase par phrase aux quatre sorties "
        "et compare la différence des gains.",
        fmt_note,
    )

    wc.set_column("A:A", 30)
    wc.set_column("B:B", 90)

    workbook.close()


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main() -> None:
    print("Chargement des fichiers...")

    references = read_lines(REFERENCE)
    t_e1 = read_lines(T_E1)
    t_e2 = read_lines(T_E2)
    t_e3 = read_lines(T_E3)
    n_e1 = read_lines(N_E1)
    n_e2 = read_lines(N_E2)
    n_e3 = read_lines(N_E3)

    n_sentences = check_alignment(
        {
            "Référence": references,
            "Transformer E1": t_e1,
            "Transformer E2": t_e2,
            "Transformer E3": t_e3,
            "NLLB E1": n_e1,
            "NLLB E2": n_e2,
            "NLLB E3": n_e3,
        }
    )

    print(f"Alignement OK : {n_sentences} phrases dans chaque fichier.")
    print(f"Bootstrap : {B} rééchantillonnages | alpha={ALPHA} | seed={SEED}")

    results: List[dict] = []

    # H1 : culture = E2 vs E1
    print("H1 - Transformer : E2 vs E1...")
    results.extend(
        run_sacrebleu_paired(
            "H1", "Transformer", "E2 - E1",
            "Transformer E1", t_e1,
            "Transformer E2", t_e2,
            references,
        )
    )

    print("H1 - NLLB-200 : E2 vs E1...")
    results.extend(
        run_sacrebleu_paired(
            "H1", "NLLB-200", "E2 - E1",
            "NLLB E1", n_e1,
            "NLLB E2", n_e2,
            references,
        )
    )

    # H2 : code-switching = E3 vs E2
    print("H2 - Transformer : E3 vs E2...")
    results.extend(
        run_sacrebleu_paired(
            "H2", "Transformer", "E3 - E2",
            "Transformer E2", t_e2,
            "Transformer E3", t_e3,
            references,
        )
    )

    print("H2 - NLLB-200 : E3 vs E2...")
    results.extend(
        run_sacrebleu_paired(
            "H2", "NLLB-200", "E3 - E2",
            "NLLB E2", n_e2,
            "NLLB E3", n_e3,
            references,
        )
    )

    # H3 : différence des gains
    print("H3 - comparaison des gains Transformer / NLLB-200...")
    results.extend(run_h3(t_e1, t_e3, n_e1, n_e3, references))

    # Ordre stable pour le tableau
    metric_order = {"BLEU": 0, "chrF++": 1}
    model_order = {"Transformer": 0, "NLLB-200": 1, "Transformer vs NLLB-200": 2}
    hyp_order = {"H1": 0, "H2": 1, "H3": 2}
    results.sort(
        key=lambda r: (
            hyp_order.get(r["Hypothese"], 99),
            model_order.get(r["Modele"], 99),
            metric_order.get(r["Metrique"], 99),
        )
    )

    print("Écriture du fichier Excel...")
    write_excel(results, n_sentences)

    print("\nTerminé.")
    print(f"Fichier Excel : {OUTPUT_XLSX}")
    print("\nRésumé :")
    for r in results:
        print(
            f"{r['Hypothese']:>2} | {r['Modele']:<25} | {r['Metrique']:<6} | "
            f"diff={r['Difference']:+.4f} | p={r['p_value']:.6f} | {r['Conclusion']}"
        )


if __name__ == "__main__":
    main()
