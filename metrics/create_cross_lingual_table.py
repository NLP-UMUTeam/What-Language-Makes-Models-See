#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

TASK_ORDER = ["religion", "salary", "politics", "education"]
TASK_LABELS = {
    "religion": "Religion",
    "salary": "Salary",
    "politics": "Politics",
    "education": "Education",
}
PAIR_ORDER = [("en", "es"), ("en", "zh"), ("es", "zh")]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_dir", required=True)
    p.add_argument("--pairwise_csv", default="cross_lingual_pairwise_consistency.csv")
    p.add_argument("--threeway_csv", default="cross_lingual_three_way_consistency.csv")
    p.add_argument("--condition", default="visual")
    p.add_argument("--output", default="cross_lingual_consistency.tex")
    p.add_argument("--model_map", default=None)
    p.add_argument("--model_order", nargs="*", default=None)
    p.add_argument("--decimals", type=int, default=2)
    p.add_argument("--percent", action="store_true")
    p.add_argument("--label", default="tab:cross_lingual_consistency")
    p.add_argument(
        "--caption",
        default=(
            "Cross-lingual consistency in the vision--language condition. "
            "We report pairwise and three-way exact agreement, pairwise "
            "Cohen's $\\kappa$, and three-language Krippendorff's $\\alpha$. "
            "Higher values indicate more stable predictions across languages."
        ),
    )
    return p.parse_args()


def load_map(path):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--model_map must contain a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def norm_pair(a, b):
    pair = (str(a).lower(), str(b).lower())
    if pair in PAIR_ORDER:
        return pair
    rev = (pair[1], pair[0])
    return rev if rev in PAIR_ORDER else pair


def fmt(value, decimals, percent=False):
    try:
        value = float(value)
    except Exception:
        return "--"
    if math.isnan(value):
        return "--"
    if percent:
        value *= 100
    return f"{value:.{decimals}f}"


def esc(text):
    return str(text).replace("_", r"\_")


def get_pair(df, model, task, pair, column):
    rows = df[
        df["model_name"].astype(str).eq(model)
        & df["task"].astype(str).str.lower().eq(task)
    ].copy()
    if rows.empty:
        return float("nan")
    rows["pair"] = rows.apply(lambda r: norm_pair(r["language_1"], r["language_2"]), axis=1)
    rows = rows[rows["pair"].map(lambda value: value == pair)]
    return float(rows.iloc[0][column]) if not rows.empty else float("nan")


def get_three(df, model, task, column):
    rows = df[
        df["model_name"].astype(str).eq(model)
        & df["task"].astype(str).str.lower().eq(task)
    ]
    return float(rows.iloc[0][column]) if not rows.empty else float("nan")


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    pairwise = pd.read_csv(metrics_dir / args.pairwise_csv)
    threeway = pd.read_csv(metrics_dir / args.threeway_csv)

    pairwise = pairwise[pairwise["condition"].astype(str).str.lower().eq(args.condition.lower())]
    threeway = threeway[threeway["condition"].astype(str).str.lower().eq(args.condition.lower())]

    models = set(pairwise["model_name"].astype(str)) | set(threeway["model_name"].astype(str))
    if args.model_order:
        model_order = [m for m in args.model_order if m in models]
        model_order += sorted(models - set(model_order))
    else:
        model_order = sorted(models)

    model_map = load_map(args.model_map)
    agreement_title = "Exact agreement (\\%)" if args.percent else "Exact agreement"

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"& &",
        rf"\multicolumn{{4}}{{c}}{{\textbf{{{agreement_title}}}}} &",
        r"\multicolumn{3}{c}{\textbf{Cohen's $\kappa$}} &",
        r"\multicolumn{1}{c}{\textbf{Kripp. $\alpha$}} \\",
        r"\cmidrule(lr){3-6}",
        r"\cmidrule(lr){7-9}",
        r"\cmidrule(lr){10-10}",
        r"\textbf{Model} &",
        r"\textbf{Task} &",
        r"\textbf{EN--ES} &",
        r"\textbf{EN--ZH} &",
        r"\textbf{ES--ZH} &",
        r"\textbf{3-way} &",
        r"\textbf{EN--ES} &",
        r"\textbf{EN--ZH} &",
        r"\textbf{ES--ZH} &",
        r"\textbf{3-lang.} \\",
        r"\midrule",
    ]

    for mi, model in enumerate(model_order):
        shown = esc(model_map.get(model, model))
        for ti, task in enumerate(TASK_ORDER):
            agr = [get_pair(pairwise, model, task, p, "exact_agreement") for p in PAIR_ORDER]
            kap = [get_pair(pairwise, model, task, p, "cohen_kappa") for p in PAIR_ORDER]
            agr3 = get_three(threeway, model, task, "three_way_exact_agreement")
            alpha = get_three(threeway, model, task, "krippendorff_alpha_nominal")

            model_cell = rf"\multirow{{4}}{{*}}{{{shown}}}" if ti == 0 else ""
            vals = [
                fmt(agr[0], args.decimals, args.percent),
                fmt(agr[1], args.decimals, args.percent),
                fmt(agr[2], args.decimals, args.percent),
                fmt(agr3, args.decimals, args.percent),
                fmt(kap[0], args.decimals),
                fmt(kap[1], args.decimals),
                fmt(kap[2], args.decimals),
                fmt(alpha, args.decimals),
            ]
            lines.append(
                f"{model_cell}\n"
                f"& {TASK_LABELS[task]}\n"
                f"& {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]}\n"
                f"& {vals[4]} & {vals[5]} & {vals[6]} & {vals[7]} \\\\"
            )
        if mi < len(model_order) - 1:
            lines += ["", r"\addlinespace", ""]

    lines += [
        "",
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