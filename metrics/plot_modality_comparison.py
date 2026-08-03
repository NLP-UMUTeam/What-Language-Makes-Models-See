#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_ORDER = ["religion", "salary", "politics", "education"]
TASK_LABELS = {
    "religion": "Religion",
    "salary": "Salary",
    "politics": "Politics",
    "education": "Education",
}
TASK_LATEX_LABELS = {
    "religion": "Rel.",
    "salary": "Sal.",
    "politics": "Pol.",
    "education": "Edu.",
}

LANG_ORDER = ["en", "es", "zh"]
LANG_LABELS = {"en": "EN", "es": "ES", "zh": "ZH"}

DEFAULT_MODALITY_COLUMNS = (
    "mean_modality_shift_njsd",
    "modality_shift_njsd",
)

DEFAULT_SUPPORT_COLUMNS = (
    "mean_visual_support",
    "visual_support",
    "mean_visual_support_for_text_only_answer",
    "visual_support_for_text_only_answer",
    "visual_support_text_only_answer",
    "mean_support",
    "support",
)

CONDITION_ALIASES = {
    "single_image": "visual",
    "image": "visual",
    "vision_language": "visual",
    "vision-language": "visual",
    "paired_images_side_by_side": "visual",
    "text": "text_only",
    "text-only": "text_only",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create revised Table 4 using modality-shift nJSD and visual "
            "support for the matched text-only answer. Visual support is "
            "computed automatically from harmonised_records.csv when a "
            "precomputed support CSV is unavailable."
        )
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="JSON manifest containing metrics_dir and model_name for each model.",
    )
    parser.add_argument(
        "--modality_shift_csv",
        default="visual_text_modality_shift_macro.csv",
        help="CSV containing modality-shift nJSD in each model metrics directory.",
    )
    parser.add_argument(
        "--harmonised_records_csv",
        default="harmonised_records.csv",
        help=(
            "CSV containing visual and text-only records. It is used to "
            "compute visual support automatically."
        ),
    )
    parser.add_argument(
        "--visual_support_csv",
        default=None,
        help=(
            "Optional precomputed support CSV. When omitted or unavailable, "
            "support is computed from harmonised_records.csv."
        ),
    )
    parser.add_argument(
        "--modality_column",
        default=None,
        help="Explicit modality-shift column name, if automatic detection fails.",
    )
    parser.add_argument(
        "--visual_support_column",
        default=None,
        help="Explicit visual-support column name, if using a precomputed CSV.",
    )
    parser.add_argument(
        "--no_write_derived_support",
        action="store_true",
        help=(
            "Do not save visual_support_text_only_answer_detail.csv and "
            "visual_support_text_only_answer_macro.csv in each metrics directory."
        ),
    )

    parser.add_argument("--languages", nargs="*", default=LANG_ORDER)
    parser.add_argument("--tasks", nargs="*", default=TASK_ORDER)

    parser.add_argument("--figure_width", type=float, default=12.0)
    parser.add_argument("--figure_height_per_row", type=float, default=0.45)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--figure_output",
        default="modality_support_comparison.pdf",
    )
    parser.add_argument(
        "--skip_figure",
        action="store_true",
        help="Generate only the LaTeX table.",
    )
    parser.add_argument("--show_values", action="store_true")
    parser.add_argument("--value_decimals", type=int, default=2)

    parser.add_argument(
        "--latex_output",
        default="modality_comparison.tex",
    )
    parser.add_argument("--latex_decimals", type=int, default=2)
    parser.add_argument(
        "--latex_caption",
        default=(
            "Cell-level comparison between the vision--language and text-only "
            "conditions. Modality shift is the normalized Jensen--Shannon "
            "divergence between the visual answer distribution and the point "
            "distribution corresponding to the text-only answer. Visual "
            "support is the proportion of matched visual instances assigned "
            "the same answer as the text-only instance. Results are "
            "macro-averaged across occupation--demographic cells."
        ),
    )
    parser.add_argument(
        "--latex_label",
        default="tab:modality_support",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("The manifest must be a non-empty JSON list.")

    required = {"metrics_dir", "model_name"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(
                f"Manifest item {index} must contain {sorted(required)}."
            )
    return data


def resolve_metrics_dir(item: dict, manifest_dir: Path) -> Path:
    metrics_dir = Path(item["metrics_dir"])
    if not metrics_dir.is_absolute():
        metrics_dir = (manifest_dir / metrics_dir).resolve()
    return metrics_dir


def canonical_condition(value: object) -> str:
    text = str(value or "").strip().lower()
    return CONDITION_ALIASES.get(text, text)


def parse_boolean_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def add_model_metadata(df: pd.DataFrame, item: dict) -> pd.DataFrame:
    result = df.copy()

    internal_name = str(item["model_name"])
    display_name = str(item.get("display_name", internal_name))
    model_size = str(item.get("size", item.get("model_size", "")))
    model_order = int(item.get("order", 9999))

    result["model_name"] = internal_name
    result["display_name"] = display_name
    result["model_size"] = model_size
    result["model_order"] = model_order

    if "language" not in result.columns:
        raise ValueError(f"Missing 'language' column for model {internal_name}.")
    if "task" not in result.columns:
        raise ValueError(f"Missing 'task' column for model {internal_name}.")

    result["language"] = result["language"].astype(str).str.strip().str.lower()
    result["task"] = result["task"].astype(str).str.strip().str.lower()
    return result


def choose_metric_column(
    df: pd.DataFrame,
    explicit_column: str | None,
    candidates: Iterable[str],
    metric_name: str,
) -> str | None:
    if explicit_column is not None:
        if explicit_column not in df.columns:
            raise ValueError(
                f"Requested {metric_name} column '{explicit_column}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit_column

    for column in candidates:
        if column in df.columns:
            return column
    return None


def derive_visual_support(
    records_path: Path,
    item: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not records_path.exists():
        raise FileNotFoundError(records_path)

    records = pd.read_csv(records_path)

    required = {
        "language",
        "condition",
        "task",
        "occupation",
        "demographic_group",
        "answer",
        "valid",
    }
    missing = sorted(required - set(records.columns))
    if missing:
        raise ValueError(
            f"{records_path} is missing required columns: {missing}. "
            f"Available columns: {list(records.columns)}"
        )

    records = records.copy()
    records["language"] = (
        records["language"].astype(str).str.strip().str.lower()
    )
    records["task"] = records["task"].astype(str).str.strip().str.lower()
    records["condition"] = records["condition"].map(canonical_condition)
    records["answer"] = (
        records["answer"].fillna("").astype(str).str.strip().str.upper()
    )
    records["valid"] = parse_boolean_series(records["valid"])

    valid_records = records[
        records["valid"]
        & records["condition"].isin({"visual", "text_only"})
        & records["answer"].ne("")
    ].copy()

    keys = [
        "language",
        "task",
        "occupation",
        "demographic_group",
    ]

    visual = valid_records[valid_records["condition"].eq("visual")]
    text_only = valid_records[valid_records["condition"].eq("text_only")]

    if visual.empty:
        raise ValueError(f"No valid visual records found in {records_path}.")
    if text_only.empty:
        raise ValueError(f"No valid text-only records found in {records_path}.")

    visual_groups = {
        key: subset
        for key, subset in visual.groupby(keys, dropna=False)
    }
    text_groups = {
        key: subset
        for key, subset in text_only.groupby(keys, dropna=False)
    }

    detail_rows: list[dict] = []
    skipped_ambiguous = 0

    for key in sorted(set(visual_groups) & set(text_groups)):
        language, task, occupation, demographic_group = key
        visual_subset = visual_groups[key]
        text_subset = text_groups[key]

        text_answers = sorted(
            answer
            for answer in text_subset["answer"].dropna().unique()
            if str(answer).strip()
        )

        if len(text_answers) != 1:
            skipped_ambiguous += 1
            warnings.warn(
                "Skipping cell with non-unique valid text-only answer: "
                f"language={language}, task={task}, occupation={occupation}, "
                f"demographic_group={demographic_group}, answers={text_answers}",
                stacklevel=2,
            )
            continue

        text_answer = text_answers[0]
        n_visual = int(len(visual_subset))
        n_matching = int((visual_subset["answer"] == text_answer).sum())
        support = n_matching / n_visual if n_visual > 0 else np.nan

        detail_rows.append(
            {
                "model_name": str(item["model_name"]),
                "language": language,
                "task": task,
                "occupation": occupation,
                "demographic_group": demographic_group,
                "text_only_answer": text_answer,
                "n_visual_valid": n_visual,
                "n_visual_matching_text_only": n_matching,
                "visual_support": support,
                "n_text_only_valid": int(len(text_subset)),
            }
        )

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        raise ValueError(
            f"No matched visual/text-only cells could be derived from {records_path}."
        )

    macro = (
        detail.groupby(
            ["model_name", "language", "task"],
            dropna=False,
        )
        .agg(
            mean_visual_support=("visual_support", "mean"),
            sd_visual_support=("visual_support", "std"),
            n_combinations=("visual_support", "count"),
            total_visual_valid=("n_visual_valid", "sum"),
            total_visual_matching_text_only=(
                "n_visual_matching_text_only",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        f"[support] {item['model_name']}: "
        f"{len(detail)} matched cells; "
        f"{skipped_ambiguous} ambiguous cells skipped."
    )
    return detail, macro


def read_model_metrics(
    item: dict,
    manifest_dir: Path,
    modality_filename: str,
    harmonised_filename: str,
    support_filename: str | None,
    write_derived_support: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_dir = resolve_metrics_dir(item, manifest_dir)
    modality_path = metrics_dir / modality_filename

    if not modality_path.exists():
        raise FileNotFoundError(modality_path)

    modality = add_model_metadata(
        pd.read_csv(modality_path),
        item,
    )

    support: pd.DataFrame | None = None

    if support_filename:
        support_path = metrics_dir / support_filename
        if support_path.exists():
            print(f"[support] Reading precomputed file: {support_path}")
            support = pd.read_csv(support_path)
        else:
            warnings.warn(
                f"Precomputed support file not found: {support_path}. "
                "Falling back to harmonised_records.csv.",
                stacklevel=2,
            )

    if support is None:
        records_path = metrics_dir / harmonised_filename
        detail, support = derive_visual_support(records_path, item)

        if write_derived_support:
            detail_path = (
                metrics_dir
                / "visual_support_text_only_answer_detail.csv"
            )
            macro_path = (
                metrics_dir
                / "visual_support_text_only_answer_macro.csv"
            )
            detail.to_csv(detail_path, index=False)
            support.to_csv(macro_path, index=False)
            print(f"[support] Wrote: {detail_path}")
            print(f"[support] Wrote: {macro_path}")

    support = add_model_metadata(support, item)
    return modality, support


def get_metric_value(
    df: pd.DataFrame,
    model: str,
    language: str,
    task: str,
    column: str,
) -> float:
    rows = df[
        df["model_name"].astype(str).eq(model)
        & df["language"].astype(str).eq(language)
        & df["task"].astype(str).eq(task)
    ]
    if rows.empty:
        return np.nan

    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    if values.empty:
        return np.nan

    if len(values) > 1:
        warnings.warn(
            f"Found {len(values)} rows for model={model}, language={language}, "
            f"task={task}, metric={column}; using their mean.",
            stacklevel=2,
        )
    return float(values.mean())


def make_matrix(
    rows: list[tuple[str, str]],
    tasks: list[str],
    df: pd.DataFrame,
    column: str,
) -> np.ndarray:
    matrix = np.full((len(rows), len(tasks)), np.nan, dtype=float)
    for row_index, (model, language) in enumerate(rows):
        for task_index, task in enumerate(tasks):
            matrix[row_index, task_index] = get_metric_value(
                df,
                model,
                language,
                task,
                column,
            )
    return matrix


def add_values(
    axis,
    matrix: np.ndarray,
    decimals: int,
) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "--" if np.isnan(value) else f"{value:.{decimals}f}"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
            )


def fmt(value: float, decimals: int) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(number):
        return "--"
    return f"{number:.{decimals}f}"


def latex_escape(text: str) -> str:
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
    return "".join(
        replacements.get(character, character)
        for character in str(text)
    )


def build_model_metadata(
    modality: pd.DataFrame,
    support: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "display_name",
        "model_size",
        "model_order",
    ]
    return (
        pd.concat(
            [
                modality[columns],
                support[columns],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["model_order", "display_name"])
        .reset_index(drop=True)
    )


def create_figure(
    modality: pd.DataFrame,
    support: pd.DataFrame,
    modality_column: str,
    support_column: str,
    model_meta: pd.DataFrame,
    languages: list[str],
    tasks: list[str],
    figure_width: float,
    figure_height_per_row: float,
    dpi: int,
    output: Path,
    show_values: bool,
    value_decimals: int,
) -> None:
    models = model_meta["model_name"].tolist()
    display_map = dict(
        zip(model_meta["model_name"], model_meta["display_name"])
    )
    size_map = dict(
        zip(model_meta["model_name"], model_meta["model_size"])
    )

    rows = [
        (model, language)
        for model in models
        for language in languages
    ]

    modality_matrix = make_matrix(
        rows,
        tasks,
        modality,
        modality_column,
    )
    support_matrix = make_matrix(
        rows,
        tasks,
        support,
        support_column,
    )

    row_labels = []
    for model, language in rows:
        display_name = display_map.get(model, model)
        size = size_map.get(model, "")
        model_label = (
            f"{display_name} {size}".strip()
            if size
            else display_name
        )
        row_labels.append(
            f"{model_label} {LANG_LABELS.get(language, language.upper())}"
        )

    task_labels = [
        TASK_LABELS.get(task, task.title())
        for task in tasks
    ]

    figure_height = max(
        4.0,
        figure_height_per_row * len(rows) + 1.5,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )

    modality_image = axes[0].imshow(
        np.ma.masked_invalid(modality_matrix),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_title("Modality shift nJSD")
    modality_colorbar = figure.colorbar(
        modality_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )
    modality_colorbar.set_label("nJSD")

    support_image = axes[1].imshow(
        np.ma.masked_invalid(support_matrix),
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    axes[1].set_title("Visual support for text-only answer")
    support_colorbar = figure.colorbar(
        support_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )
    support_colorbar.set_label("Matched visual proportion")

    for axis in axes:
        axis.set_xticks(np.arange(len(task_labels)))
        axis.set_xticklabels(task_labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(row_labels)))
        axis.set_yticklabels(row_labels)
        axis.set_xlabel("Task")
        axis.set_ylabel("Model / Language")
        axis.set_xticks(
            np.arange(-0.5, len(task_labels), 1),
            minor=True,
        )
        axis.set_yticks(
            np.arange(-0.5, len(row_labels), 1),
            minor=True,
        )
        axis.grid(
            which="minor",
            linestyle="-",
            linewidth=0.5,
        )
        axis.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

    if show_values:
        add_values(axes[0], modality_matrix, value_decimals)
        add_values(axes[1], support_matrix, value_decimals)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def create_latex_table(
    modality: pd.DataFrame,
    support: pd.DataFrame,
    modality_column: str,
    support_column: str,
    model_meta: pd.DataFrame,
    languages: list[str],
    tasks: list[str],
    decimals: int,
    caption: str,
    label: str,
    output: Path,
) -> None:
    models = model_meta["model_name"].tolist()
    display_map = dict(
        zip(model_meta["model_name"], model_meta["display_name"])
    )
    size_map = dict(
        zip(model_meta["model_name"], model_meta["model_size"])
    )

    first_metric_column = 4
    first_group_end = first_metric_column + len(tasks) - 1
    second_group_start = first_group_end + 1
    second_group_end = second_group_start + len(tasks) - 1
    tabular_spec = "lll" + ("c" * (2 * len(tasks)))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{{tabular_spec}}}",
        r"\toprule",
        r"& & &",
        rf"\multicolumn{{{len(tasks)}}}{{c}}{{\textbf{{Modality shift nJSD}}}}",
        r"&",
        rf"\multicolumn{{{len(tasks)}}}{{c}}{{\textbf{{Visual support}}}} \\",
        rf"\cmidrule(lr){{{first_metric_column}-{first_group_end}}}",
        rf"\cmidrule(lr){{{second_group_start}-{second_group_end}}}",
    ]

    header_cells = [
        r"\textbf{Model}",
        r"\textbf{Size}",
        r"\textbf{Lang.}",
    ]
    header_cells.extend(
        rf"\textbf{{{TASK_LATEX_LABELS.get(task, task.title())}}}"
        for task in tasks
    )
    header_cells.extend(
        rf"\textbf{{{TASK_LATEX_LABELS.get(task, task.title())}}}"
        for task in tasks
    )
    lines.extend(
        [
            " & ".join(header_cells) + r" \\",
            r"\midrule",
        ]
    )

    for model_index, model in enumerate(models):
        shown_name = latex_escape(display_map.get(model, model))
        shown_size = latex_escape(size_map.get(model, ""))

        for language_index, language in enumerate(languages):
            model_cell = (
                rf"\multirow{{{len(languages)}}}{{*}}{{{shown_name}}}"
                if language_index == 0
                else ""
            )
            size_cell = (
                rf"\multirow{{{len(languages)}}}{{*}}{{{shown_size}}}"
                if language_index == 0
                else ""
            )

            modality_values = [
                fmt(
                    get_metric_value(
                        modality,
                        model,
                        language,
                        task,
                        modality_column,
                    ),
                    decimals,
                )
                for task in tasks
            ]
            support_values = [
                fmt(
                    get_metric_value(
                        support,
                        model,
                        language,
                        task,
                        support_column,
                    ),
                    decimals,
                )
                for task in tasks
            ]

            row_cells = [
                model_cell,
                size_cell,
                LANG_LABELS.get(language, language.upper()),
                *modality_values,
                *support_values,
            ]
            lines.append(" & ".join(row_cells) + r" \\")

        if model_index < len(models) - 1:
            lines.extend(["", r"\addlinespace", ""])

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path)

    modality_frames: list[pd.DataFrame] = []
    support_frames: list[pd.DataFrame] = []

    for item in manifest:
        modality_frame, support_frame = read_model_metrics(
            item=item,
            manifest_dir=manifest_path.parent,
            modality_filename=args.modality_shift_csv,
            harmonised_filename=args.harmonised_records_csv,
            support_filename=args.visual_support_csv,
            write_derived_support=not args.no_write_derived_support,
        )
        modality_frames.append(modality_frame)
        support_frames.append(support_frame)

    modality = pd.concat(modality_frames, ignore_index=True)
    support = pd.concat(support_frames, ignore_index=True)

    modality_column = choose_metric_column(
        modality,
        args.modality_column,
        DEFAULT_MODALITY_COLUMNS,
        "modality-shift",
    )
    if modality_column is None:
        raise ValueError(
            "No modality-shift column found. "
            f"Available columns: {list(modality.columns)}"
        )

    support_column = choose_metric_column(
        support,
        args.visual_support_column,
        DEFAULT_SUPPORT_COLUMNS,
        "visual-support",
    )
    if support_column is None:
        raise ValueError(
            "No visual-support column found. "
            f"Available columns: {list(support.columns)}"
        )

    languages = [
        value.strip().lower()
        for value in args.languages
    ]
    tasks = [
        value.strip().lower()
        for value in args.tasks
    ]

    model_meta = build_model_metadata(
        modality,
        support,
    )

    if not args.skip_figure:
        create_figure(
            modality=modality,
            support=support,
            modality_column=modality_column,
            support_column=support_column,
            model_meta=model_meta,
            languages=languages,
            tasks=tasks,
            figure_width=args.figure_width,
            figure_height_per_row=args.figure_height_per_row,
            dpi=args.dpi,
            output=Path(args.figure_output),
            show_values=args.show_values,
            value_decimals=args.value_decimals,
        )

    create_latex_table(
        modality=modality,
        support=support,
        modality_column=modality_column,
        support_column=support_column,
        model_meta=model_meta,
        languages=languages,
        tasks=tasks,
        decimals=args.latex_decimals,
        caption=args.latex_caption,
        label=args.latex_label,
        output=Path(args.latex_output),
    )

    if not args.skip_figure:
        print(Path(args.figure_output).resolve())
    print(Path(args.latex_output).resolve())


if __name__ == "__main__":
    main()