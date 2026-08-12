import pandas as pd
from rapidfuzz import fuzz
from pathlib import Path
import argparse


def csi_match_score(prediction, csi_expected):
    if pd.isna(prediction) or pd.isna(csi_expected):
        return 0.0

    prediction = str(prediction).lower().strip()
    csi_expected = str(csi_expected).lower().strip()

    if not prediction or not csi_expected:
        return 0.0

    return fuzz.partial_ratio(csi_expected, prediction)


def evaluate_column(df, pred_col, csi_col, threshold):
    score_col = f"{pred_col}_csi_match_score"
    success_col = f"{pred_col}_csi_match_success"

    df[score_col] = df.apply(
        lambda row: csi_match_score(row[pred_col], row[csi_col]),
        axis=1
    )

    df[success_col] = df[score_col] >= threshold

    return {
        "prediction_column": pred_col,
        "number_examples": len(df),
        "average_csi_match": round(df[score_col].mean(), 2),
        f"success_rate_{int(threshold)}": round(df[success_col].mean() * 100, 2)
    }, df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csi_file", required=True)
    parser.add_argument("--multi_file", default=None)
    parser.add_argument("--separate_dir", default=None)
    parser.add_argument("--output_dir", default="csi_results")

    parser.add_argument("--id_col", default="id")
    parser.add_argument("--csi_col", default="csi_translation")
    parser.add_argument("--threshold", type=float, default=80.0)

    parser.add_argument(
        "--multi_prediction_cols",
        nargs="*",
        default=["baseline_prediction", "cultural_prediction", "tri_prediction"]
    )

    parser.add_argument(
        "--separate_prediction_col",
        default="prediction"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    csi_df = pd.read_csv(args.csi_file)

    if args.id_col not in csi_df.columns:
        raise ValueError(f"Colonne '{args.id_col}' absente du fichier CSI.")

    if args.csi_col not in csi_df.columns:
        raise ValueError(
            f"Colonne '{args.csi_col}' absente du fichier CSI. "
            f"Colonnes disponibles: {csi_df.columns.tolist()}"
        )

    summary = []

    # Cas 1 : fichier avec plusieurs colonnes de prédiction
    if args.multi_file:
        multi_df = pd.read_csv(args.multi_file)

        df = multi_df.merge(
            csi_df[[args.id_col, args.csi_col]],
            on=args.id_col,
            how="inner"
        )

        for pred_col in args.multi_prediction_cols:
            if pred_col not in df.columns:
                print(f"Colonne ignorée dans multi_file: {pred_col}")
                continue

            result, detail_df = evaluate_column(
                df.copy(),
                pred_col,
                args.csi_col,
                args.threshold
            )

            model_name = pred_col.replace("_prediction", "")

            result["model"] = model_name
            result["source_file"] = Path(args.multi_file).name
            result["file_type"] = "multi_column"

            summary.append(result)

            detail_df.to_csv(
                output_dir / f"{model_name}_multi_csi_details.csv",
                index=False
            )

    # Cas 2 : fichiers séparés
    if args.separate_dir:
        separate_dir = Path(args.separate_dir)

        for pred_file in separate_dir.glob("*.csv"):
            pred_df = pd.read_csv(pred_file)

            if args.id_col not in pred_df.columns:
                print(f"Fichier ignoré, pas de id: {pred_file.name}")
                continue

            if args.separate_prediction_col not in pred_df.columns:
                print(f"Fichier ignoré, pas de colonne '{args.separate_prediction_col}': {pred_file.name}")
                print(f"Colonnes disponibles: {pred_df.columns.tolist()}")
                continue

            df = pred_df.merge(
                csi_df[[args.id_col, args.csi_col]],
                on=args.id_col,
                how="inner"
            )

            if len(df) == 0:
                print(f"Fichier ignoré, aucun id commun: {pred_file.name}")
                continue

            result, detail_df = evaluate_column(
                df.copy(),
                args.separate_prediction_col,
                args.csi_col,
                args.threshold
            )

            model_name = pred_file.stem

            result["model"] = model_name
            result["source_file"] = pred_file.name
            result["file_type"] = "separate_file"

            summary.append(result)

            detail_df.to_csv(
                output_dir / f"{model_name}_csi_details.csv",
                index=False
            )

    summary_df = pd.DataFrame(summary)

    summary_file = output_dir / "csi_match_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n===== Résumé CSI-Match =====")
    print(summary_df)
    print(f"\nRésumé sauvegardé dans: {summary_file}")


if __name__ == "__main__":
    main()