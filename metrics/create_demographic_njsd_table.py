#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

TASK_ORDER = ["religion", "salary", "politics", "education"]
TASK_LABELS = {"religion": "Rel.", "salary": "Sal.", "politics": "Pol.", "education": "Edu."}
LANG_ORDER = ["en", "es", "zh"]
LANG_LABELS = {"en": "EN", "es": "ES", "zh": "ZH"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", required=True)
    p.add_argument("--input_csv", default="demographic_njsd_macro.csv")
    p.add_argument("--condition_visual", default="visual")
    p.add_argument("--condition_text", default="text_only")
    p.add_argument("--partition", default="intersection", choices=["gender", "race", "intersection"])
    p.add_argument("--output", default="demographic_njsd_table.tex")
    p.add_argument("--model_map", default=None)
    p.add_argument("--model_order", nargs="*", default=None)
    p.add_argument("--decimals", type=int, default=3)
    p.add_argument("--label", default="tab:demographic_njsd")
    p.add_argument(
        "--caption",
        default=(
            "Occupation-conditioned intersectional race--gender nJSD by model, "
            "language, task, and input condition. Higher values indicate stronger "
            "demographic differentiation. Bold values denote the highest value "
            "within each model, condition, and task."
        ),
    )
    return p.parse_args()


def load_model_map(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--model_map must contain a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def latex_escape(text):
    return str(text).replace("_", r"\_")


def get_value(df, model, language, condition, task):
    rows = df[
        df["model_name"].astype(str).eq(model)
        & df["language"].astype(str).str.lower().eq(language)
        & df["condition"].astype(str).str.lower().eq(condition.lower())
        & df["task"].astype(str).str.lower().eq(task)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["demographic_njsd"])


def fmt(value, decimals, bold=False):
    try:
        value = float(value)
    except Exception:
        return "--"
    if math.isnan(value):
        return "--"
    text = f"{value:.{decimals}f}"
    return rf"\textbf{{{text}}}" if bold else text


def model_order(df, requested):
    models = set(df["model_name"].dropna().astype(str))
    if not requested:
        return sorted(models)
    ordered = [m for m in requested if m in models]
    ordered += sorted(models - set(ordered))
    return ordered


def maxima(df, model, condition):
    result = {}
    for task in TASK_ORDER:
        vals = [
            get_value(df, model, language, condition, task)
            for language in LANG_ORDER
        ]
        vals = [v for v in vals if not math.isnan(v)]
        result[task] = max(vals) if vals else float("nan")
    return result


def is_max(value, maximum):
    return (
        not math.isnan(value)
        and not math.isnan(maximum)
        and math.isclose(value, maximum, rel_tol=0.0, abs_tol=1e-12)
    )


def main():
    args = parse_args()

    input_path = Path(args.metrics_dir).resolve() / args.input_csv
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)

    required = {
        "model_name", "language", "condition", "task",
        "partition", "demographic_njsd"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df[
        df["partition"].astype(str).str.lower().eq(args.partition.lower())
        & df["condition"].astype(str).str.lower().isin(
            [args.condition_visual.lower(), args.condition_text.lower()]
        )
    ].copy()

    name_map = load_model_map(args.model_map)
    models = model_order(df, args.model_order)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"& &",
        r"\multicolumn{4}{c}{\textbf{Vision--language}}",
        r"&",
        r"\multicolumn{4}{c}{\textbf{Text-only}} \\",
        r"\cmidrule(lr){3-6}",
        r"\cmidrule(lr){7-10}",
        r"\textbf{Model} & \textbf{Lang.}",
        r"& \textbf{Rel.}",
        r"& \textbf{Sal.}",
        r"& \textbf{Pol.}",
        r"& \textbf{Edu.}",
        r"& \textbf{Rel.}",
        r"& \textbf{Sal.}",
        r"& \textbf{Pol.}",
        r"& \textbf{Edu.} \\",
        r"\midrule",
    ]

    for mi, model in enumerate(models):
        shown = latex_escape(name_map.get(model, model))
        vmax = maxima(df, model, args.condition_visual)
        tmax = maxima(df, model, args.condition_text)

        for li, language in enumerate(LANG_ORDER):
            model_cell = rf"\multirow{{3}}{{*}}{{{shown}}}" if li == 0 else ""

            visual = []
            text_only = []
            for task in TASK_ORDER:
                value = get_value(df, model, language, args.condition_visual, task)
                visual.append(fmt(value, args.decimals, is_max(value, vmax[task])))
            for task in TASK_ORDER:
                value = get_value(df, model, language, args.condition_text, task)
                text_only.append(fmt(value, args.decimals, is_max(value, tmax[task])))

            lines.append(
                f"{model_cell}\n"
                f"& {LANG_LABELS[language]}\n"
                f"& {visual[0]} & {visual[1]} & {visual[2]} & {visual[3]}\n"
                f"& {text_only[0]} & {text_only[1]} & {text_only[2]} & {text_only[3]} \\\\"
            )

        if mi < len(models) - 1:
            lines += ["", r"\addlinespace", ""]

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{args.caption}}}",
        rf"\label{{{args.label}}}",
        r"\end{table*}",
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
