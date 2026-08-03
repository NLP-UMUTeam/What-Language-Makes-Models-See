# What Language Makes Models See: Cross-Lingual Bias in Sensitive Judgments by Vision-Language Models

Code release for a multilingual audit of sensitive-attribute judgments in vision--language models. The evaluation uses semantically aligned prompts in English, Spanish, and Chinese for four tasks: religion, salary, political orientation, and education. It supports both vision--language inference and a cell-level text-only diagnostic.


## Repository structure

```text
.
├── src/
│   ├── vlm_base.py                 # VLM loading and visual inference
│   └── vlm_base_text_only.py       # VLM loading with text-only support
├── scripts/
│   ├── zero_shot_en.py
│   ├── zero_shot_es.py
│   ├── zero_shot_cn.py
│   ├── zero_shot_en_text_only.py
│   ├── zero_shot_es_text_only.py
│   └── zero_shot_cn_text_only.py
├── prompts/
│   ├── prompt_en.txt                # Aligned task questions and answer spaces
│   ├── prompt_es.txt 
│   ├── prompt_cn.txt 
├── data/                            # Dataset location (not redistributed)
└── results/                         # Generated JSONL outputs
```

## Evaluation design

The scripts preserve the same images, task definitions, answer-option order, category boundaries, prompt structure, and decoding configuration across languages. Generation is deterministic by default (`do_sample=False`, `temperature=0.0`).

The text-only condition evaluates each unique occupation--demographic cell once. Repeating the same description for every image would duplicate the same deterministic prediction and would artificially weight cells according to their image count.

## Installation

Python 3.12 or newer is recommended.

```bash
python -m venv .venv 
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Some model families require recent development versions of `transformers`, model-specific dependencies, gated-model access, or multiple GPUs. Install `flash-attn` only when supported by the local CUDA environment.

## Dataset layout

The scripts expect the VisBias images in the following hierarchy:

```text
data/visbias/
├── CEO/
│   ├── ME_female/
│   ├── ME_male/
│   ├── asian_female/
│   └── ...
├── basketball_player/
├── cook/
├── doctor/
├── firefighter/
├── lawyer/
└── nurse/
```

The repository does not redistribute the benchmark images. Obtain them from the original VisBias release and follow its license and terms in the official repository (https://github.com/limenlp/VisBias/tree/main/source).

## Running vision--language inference

Run commands from the repository root. Example for English:

```bash
python scripts/zero_shot_en.py \
  --source_dir data/visbias \
  --output_jsonl results/qwen3_vl_32b_visual_en.jsonl \
  --model_name Qwen3-VL-32B-Instruct \
  --tasks religion salary politics education \
  --max_new_tokens 2048
```

Spanish and Chinese use the corresponding scripts:

```bash
python scripts/zero_shot_es.py ...
python scripts/zero_shot_cn.py ...
```

Use `--resume` to continue an interrupted JSONL run. Do not pass `--do_sample` for the deterministic setting used in the paper.

## Running the text-only diagnostic

```bash
python scripts/zero_shot_en_text_only.py \
  --source_dir data/visbias \
  --output_jsonl results/qwen3_vl_32b_text_only_en.jsonl \
  --model_name Qwen3-VL-32B-Instruct \
  --tasks religion salary politics education \
  --text_only \
  --max_new_tokens 2048
```

Equivalent scripts are provided for Spanish and Chinese. Spanish occupation names are gender-inflected when masculine and feminine forms differ, such as *director ejecutivo* and *directora ejecutiva*.

## Model paths

The model registry in `src/vlm_base.py` and `src/vlm_base_text_only.py` uses public-style model identifiers. Replace an identifier with a local checkpoint path when running on an offline cluster. Some experimental identifiers may need to be adapted to the exact checkpoint used in your environment.

## Output format

Each JSONL row stores, among other fields:

- model and decoding configuration;
- language-specific question and complete prompt;
- task, occupation, and demographic group;
- parsed answer and justification;
- raw model response;
- input condition and source metadata;
- inference errors and elapsed time.

## Reproducibility notes

- Greedy deterministic decoding is used in the reported experiments.
- The same task and option order is preserved across languages.
- The random seed controls data traversal and paired-image selection where applicable.
- Invalid or malformed responses remain recorded in the output JSONL.
- Model versions, GPU architecture, CUDA, PyTorch, and `transformers` versions should be logged for exact replication.

## Ethical use

The tasks intentionally force models to assign sensitive attributes that generally cannot be inferred reliably from appearance. Outputs must be interpreted as model associations under a controlled audit, not as valid assessments of real people. Do not use this code to profile individuals or support consequential decisions.
