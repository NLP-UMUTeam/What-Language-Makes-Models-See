from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    import jieba
except ImportError:
    jieba = None


LANG_ORDER = ["en", "es", "zh"]
LANG_LABELS = {"en": "English", "es": "Spanish", "zh": "Chinese"}
TASK_ORDER = ["religion", "salary", "politics", "education"]
TASK_LABELS = {
    "religion": "Religion",
    "salary": "Salary",
    "politics": "Politics",
    "education": "Education",
}

EN_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren","as","at","be","because","been","before","being","below",
    "between","both","but","by","can","cannot","could","couldn","did","didn",
    "do","does","doesn","doing","don","down","during","each","few","for","from",
    "further","had","hadn","has","hasn","have","haven","having","he","her",
    "here","hers","herself","him","himself","his","how","i","if","in","into",
    "is","isn","it","its","itself","just","me","more","most","mustn","my",
    "myself","no","nor","not","now","of","off","on","once","only","or","other",
    "our","ours","ourselves","out","over","own","same","she","should","shouldn",
    "so","some","such","than","that","the","their","theirs","them","themselves",
    "then","there","these","they","this","those","through","to","too","under",
    "until","up","very","was","wasn","we","were","weren","what","when","where",
    "which","while","who","whom","why","will","with","won","would","wouldn",
    "you","your","yours","yourself","yourselves",
    "image","person","individual","shown","shows","based","appears","appearance",
    "visible","visual","information","suggests","suggesting","indicate",
    "indicates","likely","may","might","therefore","however","without",
    "specific","given","option","choice","cannot","determine","solely"
}

ES_STOPWORDS = {
    "a","acá","ahí","al","algo","algún","alguna","algunas","alguno","algunos",
    "allá","allí","ambos","ante","antes","aquel","aquella","aquellas","aquello",
    "aquellos","aquí","arriba","así","aun","aunque","bajo","bien","cada","casi",
    "como","con","contra","cual","cuando","de","del","desde","donde","dos",
    "durante","e","el","ella","ellas","ello","ellos","en","entre","era","erais",
    "eran","eras","eres","es","esa","esas","ese","eso","esos","esta","estaba",
    "estaban","estado","estamos","estar","estas","este","esto","estos","fue",
    "fueron","ha","hace","hacia","han","hasta","hay","la","las","le","les","lo",
    "los","más","me","mi","mis","mucho","muy","nada","ni","no","nos","nuestra",
    "nuestro","o","otra","otras","otro","otros","para","pero","poco","por",
    "porque","que","qué","quien","se","sea","según","ser","si","sí","sin",
    "sobre","son","su","sus","también","tan","tanto","te","tiene","tienen",
    "todo","todos","tras","tu","tus","un","una","unas","uno","unos","y","ya",
    "imagen","persona","individuo","muestra","mostrada","mostrado","parece",
    "apariencia","visible","visibles","visual","información","sugiere","sugieren",
    "indica","indican","podría","puede","probablemente","por tanto","embargo",
    "específica","específico","opción","respuesta","determinar","basándose",
    "basado","basada","solo","únicamente"
}

ZH_STOPWORDS = {
    "的","了","和","是","在","有","也","与","及","或","而","但","并","因此","所以",
    "因为","由于","从","对","于","为","被","把","这","这个","这些","那","那个","那些",
    "其","他","她","它","他们","她们","一个","一种","一些","中","上","下","里","外",
    "很","更","最","较","比较","可能","可以","无法","没有","未","不","并不","通常",
    "一般","显示","表明","说明","看起来","似乎","根据","基于","判断","推断","选择",
    "选项","信息","视觉","图片","图中","照片","人物","此人","该人物","人"
}

TASK_BOILERPLATE = {
    "en": {
        "religion","religious","salary","income","education","educational",
        "political","politics","orientation","affiliation"
    },
    "es": {
        "religión","religiosa","religioso","salario","sueldo","ingresos",
        "educación","educativo","educativa","político","política","orientación",
        "afiliación"
    },
    "zh": {
        "宗教","信仰","宗教信仰","年薪","薪资","收入","教育","学历","政治","政治倾向",
        "倾向"
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default="hotword_tables")
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--min_frequency", type=int, default=3)
    parser.add_argument("--min_token_length", type=int, default=2)
    parser.add_argument(
        "--normalization",
        choices=["count", "per_1000"],
        default="count",
    )
    parser.add_argument("--include_task_terms", action="store_true")
    parser.add_argument("--custom_stopwords", default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("--cell_min_red", type=int, default=8)
    parser.add_argument("--cell_max_red", type=int, default=70)

    # New compact model-by-language tables
    parser.add_argument("--model_table_top_n", type=int, default=10)
    parser.add_argument("--highlight_min_models", type=int, default=3)
    parser.add_argument("--visual_condition", default="visual")
    parser.add_argument("--text_condition", default="text_only")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Manifest must be a non-empty JSON list.")
    required = {"path", "language"}
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(
                f"Manifest item {index} must contain {sorted(required)}."
            )
    return data


def normalize_language(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "english": "en", "eng": "en",
        "spanish": "es", "spa": "es",
        "chinese": "zh", "zh-cn": "zh", "cn": "zh",
    }
    return aliases.get(value, value)


def normalize_condition(value: object) -> str:
    value = str(value or "").strip().lower()
    aliases = {
        "single_image": "visual",
        "vision-language": "visual",
        "vision_language": "visual",
        "image": "visual",
        "text-only": "text_only",
        "text": "text_only",
    }
    return aliases.get(value, value or "unknown")


def detect_justification_language(text: str, prompt_language: str) -> str:
    """Detect the language actually used in a generated justification.

    This lightweight detector is designed for the three languages evaluated
    here. Chinese is identified from CJK characters. English and Spanish are
    separated using accented characters and frequent function words. When the
    evidence is insufficient, the prompt language is used as a fallback.
    """
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    if not normalized:
        return prompt_language

    cjk_count = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", normalized))
    latin_tokens = [
        token.lower()
        for token in re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", normalized)
    ]
    latin_chars = sum(char.isalpha() and ord(char) < 0x0250 for char in normalized)

    # A substantial CJK presence is sufficient for Chinese.
    if cjk_count >= 2 and cjk_count >= max(2, latin_chars * 0.20):
        return "zh"

    if latin_tokens:
        spanish_markers = {
            "el", "la", "los", "las", "una", "uno", "que", "como",
            "con", "por", "para", "del", "esta", "está", "persona",
            "imagen", "médico", "médica", "uniforme", "educación",
            "salario", "religión", "política", "puede", "parece",
        }
        english_markers = {
            "the", "a", "an", "this", "that", "with", "from", "for",
            "person", "image", "appears", "wearing", "medical", "uniform",
            "doctor", "education", "salary", "religion", "political",
            "likely", "suggests", "because", "based",
        }
        spanish_score = sum(token in spanish_markers for token in latin_tokens)
        english_score = sum(token in english_markers for token in latin_tokens)
        if re.search(r"[áéíóúüñ¿¡]", normalized.lower()):
            spanish_score += 2
        if spanish_score > english_score:
            return "es"
        if english_score > spanish_score:
            return "en"

        # Latin text under a Chinese prompt is overwhelmingly likely to be an
        # English fallback for the evaluated models.
        if prompt_language == "zh" and cjk_count == 0:
            return "en"

    return prompt_language


def load_custom_stopwords(path: str | None) -> dict[str, set[str]]:
    if path is None:
        return {"en": set(), "es": set(), "zh": set()}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {"en": set(), "es": set(), "zh": set()}
    for language in result:
        result[language] = {
            str(word).strip().lower()
            for word in data.get(language, [])
            if str(word).strip()
        }
    return result


def normalize_latin_token(token: str, language: str) -> str:
    return unicodedata.normalize("NFKC", token).lower().strip("'’-")


def tokenize_latin(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return [
        normalize_latin_token(token, language)
        for token in re.findall(
            r"[^\W\d_]+(?:['’][^\W\d_]+)?",
            text,
            flags=re.UNICODE,
        )
    ]


def tokenize_chinese(text: str) -> list[str]:
    if jieba is None:
        raise RuntimeError(
            "Chinese tokenization requires jieba. Install it with: pip install jieba"
        )
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[A-Za-z0-9_$€£¥]+", " ", text)
    return [
        token.strip()
        for token in jieba.lcut(text, cut_all=False)
        if token.strip()
    ]


def is_valid_token(token, language, stopwords, min_length):
    if not token or token in stopwords:
        return False
    if language in {"en", "es"}:
        return len(token) >= min_length and any(char.isalpha() for char in token)
    return len(token) >= 2 and not re.fullmatch(r"[\W_]+", token)


def extract_justification(record: dict) -> str:
    for key in [
        "justification", "justificación", "justificacion",
        "reasoning", "reason", "理由", "解释",
    ]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def read_records(item: dict, manifest_dir: Path) -> list[dict]:
    path = Path(item["path"])
    if not path.is_absolute():
        path = (manifest_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    language = normalize_language(item["language"])
    manifest_condition = item.get("condition")
    manifest_model = item.get("model_name")
    display_name = str(item.get("display_name", manifest_model or "unknown_model"))
    model_order = int(item.get("order", 9999))

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            justification = extract_justification(record)
            if not justification or record.get("error") not in (None, ""):
                continue

            model_name = str(
                manifest_model
                if manifest_model is not None
                else record.get("model_name", "unknown_model")
            )

            justification_language = detect_justification_language(
                justification, language
            )

            records.append(
                {
                    # ``language`` remains the prompt language so that table rows
                    # continue to represent the experimental condition.
                    "language": language,
                    "justification_language": justification_language,
                    "condition": normalize_condition(
                        manifest_condition
                        if manifest_condition is not None
                        else record.get("input_mode")
                    ),
                    "model_name": model_name,
                    "display_name": str(item.get("display_name", model_name)),
                    "model_order": model_order,
                    "task": str(record.get("task", "unknown")).lower(),
                    "justification": justification,
                    "source_file": str(path),
                    "line_number": line_number,
                }
            )
    return records


def get_stopwords(language, custom, include_task_terms):
    base = {
        "en": set(EN_STOPWORDS),
        "es": set(ES_STOPWORDS),
        "zh": set(ZH_STOPWORDS),
    }[language]
    base.update(custom.get(language, set()))
    if not include_task_terms:
        base.update(TASK_BOILERPLATE[language])
    return base


def tokenize_and_filter(text, language, stopwords, min_length):
    tokens = tokenize_chinese(text) if language == "zh" else tokenize_latin(text, language)
    return [
        token
        for token in tokens
        if is_valid_token(token, language, stopwords, min_length)
    ]


def compute_frequencies(records, custom_stopwords, include_task_terms, min_length):
    counters = {}
    totals = Counter()
    documents = Counter()

    output_language_counts = Counter()

    for _, row in records.iterrows():
        prompt_language = row["language"]
        justification_language = row.get(
            "justification_language", prompt_language
        )
        tokens = tokenize_and_filter(
            row["justification"],
            justification_language,
            get_stopwords(
                justification_language,
                custom_stopwords,
                include_task_terms,
            ),
            min_length,
        )
        key = (
            row["model_name"],
            prompt_language,
            row["condition"],
            row["task"],
        )
        output_language_counts[key + (justification_language,)] += 1
        counters.setdefault(key, Counter()).update(tokens)
        totals[key] += len(tokens)
        documents[key] += 1

    rows = []
    for key, counter in counters.items():
        model, language, condition, task = key
        for word, count in counter.most_common():
            rows.append(
                {
                    "model_name": model,
                    "language": language,
                    "condition": condition,
                    "task": task,
                    "word": word,
                    "count": count,
                    "per_1000_tokens": (
                        1000.0 * count / totals[key] if totals[key] else 0.0
                    ),
                    "total_retained_tokens": totals[key],
                    "n_justifications": documents[key],
                    # Dominant language actually used in the justifications for
                    # this model/prompt-language/condition/task cell.
                    "dominant_justification_language": max(
                        LANG_ORDER,
                        key=lambda output_language: output_language_counts.get(
                            key + (output_language,), 0
                        ),
                    ),
                }
            )
    return pd.DataFrame(rows)


def aggregate_frequency(frequencies, group_filters=None):
    frame = frequencies.copy()
    for column, value in (group_filters or {}).items():
        frame = frame[
            frame[column].astype(str).str.lower().eq(str(value).lower())
        ]
    if frame.empty:
        return frame

    grouped = (
        frame.groupby(["language", "word"], as_index=False)
        .agg(
            count=("count", "sum"),
            total_retained_tokens=("total_retained_tokens", "sum"),
            n_justifications=("n_justifications", "sum"),
        )
    )
    grouped["per_1000_tokens"] = (
        1000.0 * grouped["count"] / grouped["total_retained_tokens"]
    )
    return grouped


def latex_escape(text):
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
    return "".join(replacements.get(char, char) for char in str(text))


def heat_intensity(value, maximum, minimum_red, maximum_red):
    if maximum <= 0:
        return minimum_red
    normalized = min(max(value / maximum, 0.0), 1.0)
    return int(round(minimum_red + normalized * (maximum_red - minimum_red)))


def build_heat_table(
    aggregate, output, top_n, min_frequency, value_column,
    caption, label, cell_min_red, cell_max_red,
):
    ranked = {}
    for language in LANG_ORDER:
        subset = aggregate[
            aggregate["language"].eq(language)
            & aggregate["count"].ge(min_frequency)
        ].sort_values(
            [value_column, "word"],
            ascending=[False, True],
        ).head(top_n)

        ranked[language] = [
            (row["word"], float(row[value_column]), int(row["count"]))
            for _, row in subset.iterrows()
        ]

    maxima = {
        language: max((value for _, value, _ in ranked[language]), default=0.0)
        for language in LANG_ORDER
    }

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{clclclc}",
        r"\toprule",
        r"\textbf{Rank} &",
        r"\multicolumn{2}{c}{\textbf{English}} &",
        r"\multicolumn{2}{c}{\textbf{Spanish}} &",
        r"\multicolumn{2}{c}{\textbf{Chinese}} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"& \textbf{Term} & \textbf{Freq.} "
        r"& \textbf{Term} & \textbf{Freq.} "
        r"& \textbf{Term} & \textbf{Freq.} \\",
        r"\midrule",
    ]

    for rank in range(top_n):
        cells = []
        for language in LANG_ORDER:
            if rank >= len(ranked[language]):
                cells.extend(["--", "--"])
                continue
            word, value, count = ranked[language][rank]
            intensity = heat_intensity(
                value, maxima[language], cell_min_red, cell_max_red
            )
            displayed_value = (
                f"{count}" if value_column == "count" else f"{value:.1f}"
            )
            cells.extend(
                [
                    rf"\cellcolor{{red!{intensity}}}{latex_escape(word)}",
                    rf"\cellcolor{{red!{intensity}}}{displayed_value}",
                ]
            )

        lines.append(f"{rank + 1} & " + " & ".join(cells) + r" \\")

    metric_text = (
        "raw token counts"
        if value_column == "count"
        else "occurrences per 1,000 retained tokens"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption} Cell intensity is scaled independently within each language using {metric_text}; darker red indicates a more frequent term.}}",
        rf"\label{{{label}}}",
        r"\end{table*}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------------
# New: compact tables with languages as rows and models as columns
# -------------------------------------------------------------------------

def aggregate_by_model_language_condition(frequencies: pd.DataFrame) -> pd.DataFrame:
    if frequencies.empty:
        return frequencies
    grouped = (
        frequencies.groupby(
            ["model_name", "language", "condition", "word"],
            as_index=False,
        )
        .agg(
            count=("count", "sum"),
            total_retained_tokens=("total_retained_tokens", "sum"),
        )
    )
    grouped["per_1000_tokens"] = (
        1000.0 * grouped["count"] / grouped["total_retained_tokens"]
    )
    return grouped


def selected_model_words(
    aggregate: pd.DataFrame,
    models: list[str],
    condition: str,
    top_n: int,
    min_frequency: int,
    value_column: str,
) -> dict[tuple[str, str], list[str]]:
    selected = {}
    for language in LANG_ORDER:
        for model in models:
            subset = aggregate[
                aggregate["model_name"].astype(str).eq(model)
                & aggregate["language"].eq(language)
                & aggregate["condition"].eq(condition)
                & aggregate["count"].ge(min_frequency)
            ].sort_values(
                [value_column, "word"],
                ascending=[False, True],
            ).head(top_n)

            selected[(language, model)] = subset["word"].astype(str).tolist()
    return selected


def shared_model_words(
    selected: dict[tuple[str, str], list[str]],
    models: list[str],
    min_models: int,
) -> dict[str, set[str]]:
    shared = {}
    for language in LANG_ORDER:
        presence = Counter()
        for model in models:
            presence.update(set(selected.get((language, model), [])))
        shared[language] = {
            word for word, count in presence.items()
            if count >= min_models
        }
    return shared


def format_compact_word_list(
    words: list[tuple[str, float]],
    maximum: float,
    minimum_red: int,
    maximum_red: int,
) -> str:
    """Format terms with red intensity proportional to their frequency."""
    if not words:
        return "--"

    output = []
    for word, value in words:
        escaped = latex_escape(word)
        intensity = heat_intensity(
            value=value,
            maximum=maximum,
            minimum_red=minimum_red,
            maximum_red=maximum_red,
        )
        output.append(
            rf"\colorbox{{red!{intensity}}}{{\strut {escaped}}}"
        )
    return ", ".join(output)

def build_model_language_table(
    aggregate: pd.DataFrame,
    model_meta: pd.DataFrame,
    condition: str,
    output: Path,
    top_n: int,
    min_frequency: int,
    value_column: str,
    highlight_min_models: int,
    caption: str,
    label: str,
):
    models = model_meta["model_name"].tolist()
    display_map = dict(
        zip(model_meta["model_name"], model_meta["display_name"])
    )

    selected = selected_model_words(
        aggregate=aggregate,
        models=models,
        condition=condition,
        top_n=top_n,
        min_frequency=min_frequency,
        value_column=value_column,
    )
    shared = shared_model_words(
        selected=selected,
        models=models,
        min_models=highlight_min_models,
    )

    n_models = max(len(models), 1)
    first_width = 0.11
    model_width = 0.86 / n_models
    column_spec = (
        f"p{{{first_width:.2f}\\linewidth}}"
        + "".join(
            f"p{{{model_width:.3f}\\linewidth}}"
            for _ in models
        )
    )

    header = "Language & " + " & ".join(
        latex_escape(display_map.get(model, model))
        for model in models
    ) + r" \\"

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for index, language in enumerate(LANG_ORDER):
        cells = [
            format_compact_word_list(
                selected.get((language, model), []),
                shared[language],
            )
            for model in models
        ]
        lines.append(
            f"{LANG_LABELS[language]} & "
            + " & ".join(cells)
            + r" \\"
        )
        if index < len(LANG_ORDER) - 1:
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")



def build_language_mismatch_map(records: pd.DataFrame) -> dict[tuple[str, str, str, str], str]:
    """Return the dominant non-prompt output language for each table cell."""
    if records.empty:
        return {}

    counts = (
        records.groupby(
            [
                "model_name",
                "language",
                "condition",
                "task",
                "justification_language",
            ]
        )
        .size()
        .rename("count")
        .reset_index()
    )
    mismatch = {}
    group_columns = ["model_name", "language", "condition", "task"]
    for keys, group in counts.groupby(group_columns):
        dominant = group.sort_values(
            ["count", "justification_language"],
            ascending=[False, True],
        ).iloc[0]
        prompt_language = str(keys[1])
        output_language = str(dominant["justification_language"])
        if output_language != prompt_language:
            mismatch[tuple(str(value) for value in keys)] = output_language
    return mismatch


def build_task_language_model_table(
    frequencies: pd.DataFrame,
    model_meta: pd.DataFrame,
    condition: str,
    tasks: list[str],
    output: Path,
    top_n: int,
    min_frequency: int,
    value_column: str,
    highlight_min_models: int,
    caption: str,
    label: str,
    cell_min_red: int,
    cell_max_red: int,
    language_mismatch_map: dict[tuple[str, str, str, str], str] | None = None,
):
    """Build one table per condition, grouped by task and language.

    Red intensity is determined by term frequency. Scaling is performed
    independently within each task--language--condition row, but jointly
    across all models in that row, making model cells directly comparable.
    """
    models = model_meta["model_name"].tolist()
    display_map = dict(zip(model_meta["model_name"], model_meta["display_name"]))

    aggregate = (
        frequencies.groupby(
            ["model_name", "language", "condition", "task", "word"],
            as_index=False,
        )
        .agg(
            count=("count", "sum"),
            total_retained_tokens=("total_retained_tokens", "sum"),
        )
    )
    aggregate["per_1000_tokens"] = (
        1000.0 * aggregate["count"] / aggregate["total_retained_tokens"]
    )

    selected: dict[tuple[str, str, str], list[tuple[str, float]]] = {}
    row_maximum: dict[tuple[str, str], float] = {}

    for task in tasks:
        for language in LANG_ORDER:
            all_selected_values = []

            for model in models:
                subset = aggregate[
                    aggregate["model_name"].astype(str).eq(model)
                    & aggregate["language"].astype(str).str.lower().eq(language)
                    & aggregate["condition"].astype(str).str.lower().eq(condition)
                    & aggregate["task"].astype(str).str.lower().eq(task)
                    & aggregate["count"].ge(min_frequency)
                ].sort_values(
                    [value_column, "word"],
                    ascending=[False, True],
                ).head(top_n)

                terms = [
                    (str(row["word"]), float(row[value_column]))
                    for _, row in subset.iterrows()
                ]
                selected[(task, language, model)] = terms
                all_selected_values.extend(value for _, value in terms)

            row_maximum[(task, language)] = max(all_selected_values, default=0.0)

    n_models = max(len(models), 1)
    task_width = 0.085
    language_width = 0.10
    model_width = 0.79 / n_models
    column_spec = (
        f"p{{{task_width:.3f}\\linewidth}}"
        f"p{{{language_width:.3f}\\linewidth}}"
        + "".join(f"p{{{model_width:.3f}\\linewidth}}" for _ in models)
    )

    header = (
        "Task & Language & "
        + " & ".join(latex_escape(display_map.get(model, model)) for model in models)
        + r" \\"
    )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    last_column = 2 + len(models)
    for task_index, task in enumerate(tasks):
        for language_index, language in enumerate(LANG_ORDER):
            task_cell = (
                rf"\multirow{{3}}{{*}}{{{TASK_LABELS.get(task, task)}}}"
                if language_index == 0 else ""
            )
            maximum = row_maximum.get((task, language), 0.0)
            cells = []
            for model in models:
                cell = format_compact_word_list(
                    selected.get((task, language, model), []),
                    maximum=maximum,
                    minimum_red=cell_min_red,
                    maximum_red=cell_max_red,
                )
                mismatch_language = (language_mismatch_map or {}).get(
                    (str(model), language, condition, task)
                )
                if mismatch_language is not None and cell != "--":
                    cell += r"\textsuperscript{\dagger}"
                cells.append(cell)
            lines.append(
                f"{task_cell} & {LANG_LABELS[language]} & "
                + " & ".join(cells)
                + r" \\"
            )
            if language_index < 2:
                lines.append(rf"\cmidrule(lr){{2-{last_column}}}")
        if task_index < len(tasks) - 1:
            lines.append(r"\midrule")

    metric_text = (
        "raw term frequency"
        if value_column == "count"
        else "frequency per 1,000 retained tokens"
    )
    mismatch_note = ""
    relevant_mismatches = {
        key: output_language
        for key, output_language in (language_mismatch_map or {}).items()
        if key[2] == condition and key[3] in tasks
    }
    if relevant_mismatches:
        mismatch_note = (
            r" \textsuperscript{\dagger}The generated justification was "
            r"predominantly in a language different from the prompt language; "
            r"the lexical cues are reported in their original generated language "
            r"without translation."
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption} Red intensity is proportional to {metric_text} and is scaled jointly across models within each task--language row.{mismatch_note}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    all_records = []
    for item in manifest:
        all_records.extend(read_records(item, manifest_path.parent))

    records = pd.DataFrame(all_records)
    if records.empty:
        raise RuntimeError("No valid justifications were found.")

    if args.conditions:
        allowed = {normalize_condition(value) for value in args.conditions}
        records = records[records["condition"].isin(allowed)]

    tasks = [task.lower() for task in args.tasks]
    records = records[records["task"].isin(tasks)]

    custom_stopwords = load_custom_stopwords(args.custom_stopwords)
    frequencies = compute_frequencies(
        records=records,
        custom_stopwords=custom_stopwords,
        include_task_terms=args.include_task_terms,
        min_length=args.min_token_length,
    )
    frequencies.to_csv(
        output_dir / "hotword_frequencies.csv",
        index=False,
    )

    value_column = (
        "count"
        if args.normalization == "count"
        else "per_1000_tokens"
    )

    overall = aggregate_frequency(frequencies)
    build_heat_table(
        aggregate=overall,
        output=output_dir / "hotwords_overall.tex",
        top_n=args.top_n,
        min_frequency=args.min_frequency,
        value_column=value_column,
        caption=(
            "Most frequent content words in model justifications across all "
            "tasks, models, and input conditions."
        ),
        label="tab:appendix_hotwords_overall",
        cell_min_red=args.cell_min_red,
        cell_max_red=args.cell_max_red,
    )

    generated = [
        r"\section{Justification Hotword Analysis}",
        r"\label{appendix:hotwords}",
        (
            "We remove language-specific stopwords, punctuation, numerical "
            "tokens, generic response boilerplate, and task-label terms before "
            "counting content words. Chinese text is segmented using Jieba. "
            "Red intensity is normalised independently within each language."
        ),
        "",
        r"\subsection{Overall Frequencies}",
        r"\input{hotword_tables/hotwords_overall}",
        "",
    ]

    for task in tasks:
        aggregate = aggregate_frequency(frequencies, {"task": task})
        filename = f"hotwords_by_task_{task}.tex"
        build_heat_table(
            aggregate=aggregate,
            output=output_dir / filename,
            top_n=args.top_n,
            min_frequency=args.min_frequency,
            value_column=value_column,
            caption=(
                f"Most frequent content words in justifications for the "
                f"{TASK_LABELS.get(task, task)} task."
            ),
            label=f"tab:appendix_hotwords_{task}",
            cell_min_red=args.cell_min_red,
            cell_max_red=args.cell_max_red,
        )
        generated.extend(
            [
                rf"\subsection{{{TASK_LABELS.get(task, task)} Justifications}}",
                rf"\input{{hotword_tables/{filename[:-4]}}}",
                "",
            ]
        )

    for condition in sorted(records["condition"].unique()):
        aggregate = aggregate_frequency(frequencies, {"condition": condition})
        safe_condition = re.sub(r"[^a-z0-9_]+", "_", condition.lower())
        filename = f"hotwords_by_condition_{safe_condition}.tex"
        condition_label = (
            "Vision--language"
            if condition == "visual"
            else "Text-only"
            if condition == "text_only"
            else condition
        )
        build_heat_table(
            aggregate=aggregate,
            output=output_dir / filename,
            top_n=args.top_n,
            min_frequency=args.min_frequency,
            value_column=value_column,
            caption=(
                f"Most frequent content words in {condition_label} "
                f"justifications."
            ),
            label=f"tab:appendix_hotwords_condition_{safe_condition}",
            cell_min_red=args.cell_min_red,
            cell_max_red=args.cell_max_red,
        )
        generated.extend(
            [
                rf"\subsection{{{condition_label} Justifications}}",
                rf"\input{{hotword_tables/{filename[:-4]}}}",
                "",
            ]
        )

    # Two compact tables total: one visual and one text-only.
    # Each table contains all tasks and languages.
    model_meta = (
        records[["model_name", "display_name", "model_order"]]
        .drop_duplicates()
        .sort_values(["model_order", "display_name"])
        .reset_index(drop=True)
    )

    visual_condition = normalize_condition(args.visual_condition)
    text_condition = normalize_condition(args.text_condition)
    language_mismatch_map = build_language_mismatch_map(records)

    build_task_language_model_table(
        frequencies=frequencies,
        model_meta=model_meta,
        condition=visual_condition,
        tasks=tasks,
        output=output_dir / "hotwords_visual_tasks_table.tex",
        top_n=args.model_table_top_n,
        min_frequency=args.min_frequency,
        value_column=value_column,
        highlight_min_models=args.highlight_min_models,
        caption=(
            "Frequent lexical cues in vision--language justifications by task "
            "and language. Terms are selected independently for each model, "
            "task, and language after removing stopwords, punctuation, generic "
            "response terms, and task labels. Terms appearing among the "
            f"selected cues for at least {args.highlight_min_models} models "
            "within the same task and language are highlighted."
        ),
        label="tab:lexical-visual-task-rationales",
        cell_min_red=args.cell_min_red,
        cell_max_red=args.cell_max_red,
        language_mismatch_map=language_mismatch_map,
    )

    build_task_language_model_table(
        frequencies=frequencies,
        model_meta=model_meta,
        condition=text_condition,
        tasks=tasks,
        output=output_dir / "hotwords_text_only_tasks_table.tex",
        top_n=args.model_table_top_n,
        min_frequency=args.min_frequency,
        value_column=value_column,
        highlight_min_models=args.highlight_min_models,
        caption=(
            "Frequent lexical cues in text-only justifications by task and "
            "language. Terms are selected independently for each model, task, "
            "and language after removing stopwords, punctuation, generic "
            "response terms, and task labels. Terms appearing among the "
            f"selected cues for at least {args.highlight_min_models} models "
            "within the same task and language are highlighted."
        ),
        label="tab:lexical-text-task-rationales",
        cell_min_red=args.cell_min_red,
        cell_max_red=args.cell_max_red,
        language_mismatch_map=language_mismatch_map,
    )

    generated.extend([
        r"\subsection{Model-Specific Lexical Cues by Task}",
        r"\input{hotword_tables/hotwords_visual_tasks_table}",
        "",
        r"\input{hotword_tables/hotwords_text_only_tasks_table}",
        "",
    ])

    (output_dir / "hotwords_appendix_master.tex").write_text(
        "\n".join(generated) + "\n",
        encoding="utf-8",
    )

    summary = (
        records.groupby(
            [
                "language",
                "justification_language",
                "condition",
                "task",
                "model_name",
            ]
        )
        .size()
        .rename("n_justifications")
        .reset_index()
    )
    summary.to_csv(
        output_dir / "hotword_input_summary.csv",
        index=False,
    )

    print(f"Hotword tables written to: {output_dir}")
    print(output_dir / "hotwords_visual_tasks_table.tex")
    print(output_dir / "hotwords_text_only_tasks_table.tex")


if __name__ == "__main__":
    main()