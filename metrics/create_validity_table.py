from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


LANGUAGES = [
    ("en", "English"),
    ("es", "Spanish"),
    ("zh", "Chinese"),
]

TASKS = [
    ("religion", "Religion"),
    ("salary", "Salary"),
    ("politics", "Politics"),
    ("education", "Education"),
]

CONDITIONS = [
    ("visual", "Visual"),
    ("text_only", "Text-only"),
]


def latex_escape(value: object) -> str:
    """Escape characters that are special in ordinary LaTeX text."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def load_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    frame = pd.read_csv(path)
    missing = required_columns.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"{path} is missing columns: {missing_text}")
    return frame


def prepare_validity(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate validity safely in case the input contains repeated rows.

    The rate is recomputed as sum(valid) / sum(total), rather than averaging
    valid_response_rate values.
    """
    keys = ["model_name", "condition", "language", "task"]

    result = (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(total=("total", "sum"), valid=("valid", "sum"))
    )

    result["valid_pct"] = (
        100.0 * result["valid"] / result["total"].where(result["total"] != 0)
    )
    return result


def prepare_dominant_answers(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Obtain the dominant answer for every model/condition/language/task.

    Counts are summed across occupations and demographic groups. The reported
    percentage is top_count / sum(count), not an average of subgroup
    proportions.
    """
    base_keys = ["model_name", "condition", "language", "task"]
    answer_keys = base_keys + ["answer"]

    counts = (
        frame.groupby(answer_keys, as_index=False, dropna=False)["count"]
        .sum()
        .rename(columns={"count": "answer_count"})
    )

    totals = (
        counts.groupby(base_keys, as_index=False)["answer_count"]
        .sum()
        .rename(columns={"answer_count": "valid_answer_total"})
    )

    counts = counts.merge(totals, on=base_keys, how="left")
    counts["top_pct"] = (
        100.0
        * counts["answer_count"]
        / counts["valid_answer_total"].where(counts["valid_answer_total"] != 0)
    )

    # Deterministic tie-breaking: highest count, then alphabetical answer label.
    counts = counts.sort_values(
        base_keys + ["answer_count", "answer"],
        ascending=[True, True, True, True, False, True],
    )

    dominant = counts.drop_duplicates(base_keys, keep="first")
    return dominant[base_keys + ["answer", "top_pct"]]


def lookup_one(
    frame: pd.DataFrame,
    model: str,
    condition: str,
    language: str,
    task: str,
) -> pd.Series:
    mask = (
        frame["model_name"].eq(model)
        & frame["condition"].eq(condition)
        & frame["language"].eq(language)
        & frame["task"].eq(task)
    )
    matches = frame.loc[mask]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one row for "
            f"model={model!r}, condition={condition!r}, "
            f"language={language!r}, task={task!r}; found {len(matches)}."
        )
    return matches.iloc[0]


def build_rows(
    validity: pd.DataFrame,
    dominant: pd.DataFrame,
    models: list[str],
    decimals: int,
) -> list[str]:
    rows: list[str] = []
    model_span = len(CONDITIONS) * len(TASKS)

    for model_index, model in enumerate(models):
        if model_index > 0:
            rows.extend([r"\addlinespace", r"\midrule", r"\addlinespace"])

        first_model_row = True

        for condition_index, (condition_key, condition_label) in enumerate(CONDITIONS):
            if condition_index > 0:
                rows.append(r"\cmidrule(lr){2-10}")

            for task_index, (task_key, task_label) in enumerate(TASKS):
                cells: list[str] = []

                if first_model_row:
                    cells.append(
                        rf"\multirow{{{model_span}}}{{*}}{{{latex_escape(model)}}}"
                    )
                    first_model_row = False
                else:
                    cells.append("")

                if task_index == 0:
                    cells.append(
                        rf"\multirow{{{len(TASKS)}}}{{*}}{{{condition_label}}}"
                    )
                else:
                    cells.append("")

                cells.append(task_label)

                language_validities: list[float] = []

                for language_key, _ in LANGUAGES:
                    valid_row = lookup_one(
                        validity, model, condition_key, language_key, task_key
                    )
                    top_row = lookup_one(
                        dominant, model, condition_key, language_key, task_key
                    )

                    valid_pct = float(valid_row["valid_pct"])
                    top_pct = float(top_row["top_pct"])
                    answer = latex_escape(top_row["answer"])

                    language_validities.append(valid_pct)
                    cells.append(f"{valid_pct:.{decimals}f}")
                    cells.append(f"{answer} ({top_pct:.{decimals}f})")

                avg_valid = sum(language_validities) / len(language_validities)
                cells.append(f"{avg_valid:.{decimals}f}")

                rows.append(" & ".join(cells) + r" \\")

    return rows


def build_table(
    validity: pd.DataFrame,
    dominant: pd.DataFrame,
    models: list[str],
    decimals: int,
    rows_only: bool,
) -> str:
    rows = build_rows(validity, dominant, models, decimals)

    if rows_only:
        return "\n".join(rows) + "\n"

    table_lines = [
        r"% Required packages:",
        r"% \usepackage{booktabs}",
        r"% \usepackage{multirow}",
        "",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lllccccccc}",
        r"\toprule",
        r"& & &",
        r"\multicolumn{2}{c}{\textbf{English}} &",
        r"\multicolumn{2}{c}{\textbf{Spanish}} &",
        r"\multicolumn{2}{c}{\textbf{Chinese}} &",
        r"\textbf{Avg.} \\",
        r"\cmidrule(lr){4-5}",
        r"\cmidrule(lr){6-7}",
        r"\cmidrule(lr){8-9}",
        r"\textbf{Model} &",
        r"\textbf{Cond.} &",
        r"\textbf{Task} &",
        r"\textbf{Valid} &",
        r"\textbf{Top} &",
        r"\textbf{Valid} &",
        r"\textbf{Top} &",
        r"\textbf{Valid} &",
        r"\textbf{Top} &",
        r"\textbf{Valid} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Task-level valid-response rates and dominant answer "
            r"distributions by model, language, and input condition. "
            r"``Valid'' denotes the percentage of outputs containing exactly "
            r"one permitted answer. ``Top'' reports the most frequently selected "
            r"answer and its proportion among valid responses in parentheses. "
            r"``Avg.'' is the valid-response rate averaged across languages.}"
        ),
        r"\label{tab:validity_answer_distributions}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(table_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine response validity and answer distributions into LaTeX."
    )
    parser.add_argument(
        "--validity",
        type=Path,
        required=True,
        help="Path to response_validity.csv.",
    )
    parser.add_argument(
        "--distributions",
        type=Path,
        required=True,
        help="Path to answer_distributions.csv.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model to include. Repeat this option to include several models. "
            "By default, all models are included."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex file. If omitted, LaTeX is printed to stdout.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=1,
        help="Number of decimal places (default: 1).",
    )
    parser.add_argument(
        "--rows-only",
        action="store_true",
        help="Generate only the table body rows, without the table environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    validity_required = {
        "model_name",
        "language",
        "task",
        "condition",
        "total",
        "valid",
    }
    distributions_required = {
        "model_name",
        "language",
        "condition",
        "task",
        "answer",
        "count",
    }

    try:
        validity_raw = load_csv(args.validity, validity_required)
        distributions_raw = load_csv(
            args.distributions, distributions_required
        )

        validity = prepare_validity(validity_raw)
        dominant = prepare_dominant_answers(distributions_raw)

        available_models = validity["model_name"].drop_duplicates().tolist()
        distribution_models = set(dominant["model_name"].unique())

        available_models = [
            model for model in available_models if model in distribution_models
        ]

        if args.model:
            unknown = [model for model in args.model if model not in available_models]
            if unknown:
                raise ValueError(
                    "Requested model(s) not found in both CSV files: "
                    + ", ".join(unknown)
                )
            models = args.model
        else:
            models = available_models

        if not models:
            raise ValueError("No common models were found in the two CSV files.")

        latex = build_table(
            validity=validity,
            dominant=dominant,
            models=models,
            decimals=args.decimals,
            rows_only=args.rows_only,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(latex, encoding="utf-8")
            print(f"LaTeX written to: {args.output}", file=sys.stderr)
        else:
            print(latex, end="")

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())