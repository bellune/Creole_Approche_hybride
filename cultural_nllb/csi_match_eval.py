import pandas as pd
from rapidfuzz import fuzz
from pathlib import Path
import argparse


def read_file(path):
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        return pd.read_json(path)
    else:
        raise ValueError(f"Format non supporté: {path.suffix}")


def csi_match_score(prediction, csi_expected):
    if pd.isna(prediction) or pd.isna(csi_expected):
        return 0.0

    prediction = str(prediction).lower().strip()
    csi_expected = str(csi_expected).lower().strip()

    if not prediction or not csi_expected:
        return 0.0

    return fuzz.partial_ratio(csi_expected, prediction)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csi_file", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--output_dir", default="csi_results")
    parser.add_argument("--id_col", default="id")
    parser.add_argument("--prediction_col", default="prediction")
    parser.add_argument("--csi_col", default="csi_translation")
    parser.add_argument("--threshold", type=float, default=80.0)

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    csi_df = read_file(args.csi_file)

    if args.id_col not in csi_df.columns:
        raise ValueError(f"Colonne id absente dans le fichier CSI: {args.id_col}")

    if args.csi_col not in csi_df.columns:
        raise ValueError(
            f"Colonne CSI absente: {args.csi_col}. "
            f"Colonnes disponibles: {csi_df.columns.tolist()}"
        )

    prediction_files = (
        list(pred_dir.glob("*.csv"))
        + list(pred_dir.glob("*.jsonl"))
        + list(pred_dir.glob("*.json"))
    )

    summary = []

    for pred_file in prediction_files:
        print(f"\nÉvaluation de: {pred_file.name}")
        pred_df = read_file(pred_file)

        if args.id_col not in pred_df.columns:
            print(f"  Ignoré: colonne id absente dans {pred_file.name}")
            continue

        if args.prediction_col not in pred_df.columns:
            print(f"  Ignoré: colonne prediction absente dans {pred_file.name}")
            print(f"  Colonnes disponibles: {pred_df.columns.tolist()}")
            continue

        df = pred_df.merge(
            csi_df[[args.id_col, args.csi_col]],
            on=args.id_col,
            how="inner"
        )

        if len(df) == 0:
            print("  Ignoré: aucun id commun trouvé.")
            continue

        df["csi_match_score"] = df.apply(
            lambda row: csi_match_score(row[args.prediction_col], row[args.csi_col]),
            axis=1
        )

        df["csi_match_success"] = df["csi_match_score"] >= args.threshold

        avg_score = df["csi_match_score"].mean()
        success_rate = df["csi_match_success"].mean() * 100

        model_name = pred_file.stem

        summary.append({
            "model": model_name,
            "prediction_file": pred_file.name,
            "number_examples": len(df),
            "average_csi_match": round(avg_score, 2),
            f"success_rate_{int(args.threshold)}": round(success_rate, 2)
        })

        output_file = output_dir / f"{model_name}_csi_details.csv"
        df.to_csv(output_file, index=False)

        print(f"  Exemples évalués: {len(df)}")
        print(f"  CSI-Match moyen: {avg_score:.2f}")
        print(f"  Success rate @{args.threshold}: {success_rate:.2f}%")
        print(f"  Détails: {output_file}")

    summary_df = pd.DataFrame(summary)
    summary_file = output_dir / "csi_match_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n===== Résumé final =====")
    print(summary_df)
    print(f"\nRésumé sauvegardé dans: {summary_file}")


if __name__ == "__main__":
    main()