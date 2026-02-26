import argparse
import json
import os
import pickle
import re
from pathlib import Path

import anthropic
import numpy as np
import pandas as pd
import requests
import torch
from mammo_metrics import is_mammo_dataset
from model_factory import create_embedding_backend
from openai import OpenAI
from prompts.gpt4_prompt import (
    create_CELEBA_prompts,
    create_Metashift_prompts,
    create_NIH_prompts,
    create_RSNA_prompts,
    create_Waterbirds_prompts,
)
from utils import seed_all


def config():
    parser = argparse.ArgumentParser(
        description="Discovering Error Slices via LLM  using LLM-generated hypotheses and CLIP."
    )
    parser.add_argument(
        "--dataset",
        default="Waterbirds",
        type=str,
        help="Dataset name (e.g., NIH, RSNA, Waterbirds, CelebA, MetaShift).",
    )
    parser.add_argument(
        "--clip_check_pt",
        default="",
        type=str,
        help="Path to the pretrained CLIP checkpoint (optional).",
    )
    parser.add_argument(
        "--LLM",
        default="gpt-4o",
        type=str,
        help="Which LLM to use (e.g., gpt-4o, gpt-4o-azure-api, claude, llama, gemini, gemini-vertex).",
    )
    parser.add_argument(
        "--key",
        default="",
        type=str,
        help="API key for the selected LLM (OpenAI, Claude, Gemini, etc).",
    )
    parser.add_argument(
        "--clip_vision_encoder",
        default="swin-tiny-cxr-clip",
        type=str,
        help="CLIP vision encoder architecture (e.g., RN50, ViT-B/32, swin-tiny-cxr-clip).",
    )
    parser.add_argument(
        "--class_label",
        default="",
        type=str,
        help="Target class label for error slice analysis (e.g., 'dog', 'cat', 'blonde').",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to use for inference (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--prediction_col",
        default="out_put_predict",
        type=str,
        help="Column name in CSV with model predictions to evaluate.",
    )
    parser.add_argument(
        "--top50-err-text",
        default="./Ladder/out/NIH_Cxrclip/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip/pneumothorax_error_top_50_sent_diff_emb.txt",
        type=str,
        help="Path to the file containing top-K error slice sentences.",
    )
    parser.add_argument(
        "--save_path",
        metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32",
        help="Directory to save error slice outputs and logs (supports {seed} formatting).",
    )
    parser.add_argument(
        "--clf_results_csv",
        metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_additional_info.csv",
        help="Path to classifier outputs with ground truth and predictions.",
    )
    parser.add_argument(
        "--clf_image_emb_path",
        metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy",
        help="Fallback path to NumPy file containing image embeddings.",
    )
    parser.add_argument(
        "--image_emb_path",
        metavar="DIR",
        default="",
        help="Explicit image embedding path template (preferred for llava_mammo backend).",
    )

    parser.add_argument(
        "--aligner_path",
        metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/aligner/aligner_50.pth",
        help="Path to trained linear aligner (classifier to CLIP space projection).",
    )
    parser.add_argument(
        "--tokenizers",
        default="",
        type=str,
        help="Path to tokenizer (required for CXR-CLIP or Mammo-CLIP).",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Path to local cache for pretrained models or tokenizers.",
    )
    parser.add_argument(
        "--azure_api_version",
        default="",
        type=str,
        help="API version for Azure OpenAI deployment (required if using gpt-4o-azure-api).",
    )
    parser.add_argument("--azure_endpoint", default="", type=str, help="Azure OpenAI endpoint URL")
    parser.add_argument(
        "--azure_deployment_name",
        default="",
        type=str,
        help="Name of your Azure deployment for the GPT model.",
    )
    parser.add_argument(
        "--append_birads_to_passed_hypothesis",
        action="store_true",
        help="Append BI-RADS score (from passed slice samples) to hypothesis labels in logs/output columns.",
    )
    parser.add_argument(
        "--birads_formats",
        nargs="+",
        default=["category"],
        metavar="FORMAT",
        help=(
            "One or more BI-RADS suffix formats to apply when "
            "--append_birads_to_passed_hypothesis is set. "
            "Multiple formats produce a cross-product with BI-RADS levels. "
            "Built-in presets: "
            "  category       →  ', BI-RADS category {level}'  (default) "
            "  category_desc  →  ', BI-RADS category {level} ({desc})' "
            "  short          →  ', BI-RADS {level}' "
            "  parens         →  ' (BI-RADS {level})' "
            "  parens_desc    →  ' (BI-RADS {level}: {desc})' "
            "You may also supply a raw template string that uses {level} and/or {desc} "
            'as placeholders, e.g. " [{level}]" or ", assessment {level} ({desc})".'
        ),
    )
    parser.add_argument(
        "--birads_expansion_mode",
        default="all",
        choices=["all", "mode"],
        help=(
            "Controls how BI-RADS levels are appended when "
            "--append_birads_to_passed_hypothesis is set. "
            "'all' (default): expand each verified hypothesis into one variant per "
            "BI-RADS level (1-5). "
            "'mode': append only the single modal BI-RADS level observed in the "
            "passed slice for that hypothesis, producing one variant per hypothesis."
        ),
    )
    parser.add_argument(
        "--backend",
        default="legacy",
        choices=["legacy", "llava_mammo"],
        help="Embedding backend. legacy uses CLIP+aligner; llava_mammo uses Llava embeddings directly.",
    )
    parser.add_argument(
        "--llava_model_path",
        default="",
        type=str,
        help="Local path to unpacked Llava-Mammo model or adapter directory (required for llava_mammo backend).",
    )
    parser.add_argument(
        "--llava_base_model_id",
        default="llava-hf/llava-v1.6-vicuna-7b-hf",
        type=str,
        help="Base Llava model ID used when llava_model_path points to an adapter-only checkpoint.",
    )
    parser.add_argument(
        "--llava_processor_id",
        default="llava-hf/llava-v1.6-vicuna-7b-hf",
        type=str,
        help="Processor ID for Llava input preprocessing.",
    )
    parser.add_argument(
        "--llava_text_batch_size",
        default=32,
        type=int,
        help="Micro-batch size for Llava text embedding extraction.",
    )
    parser.add_argument(
        "--llava_text_max_length",
        default=256,
        type=int,
        help="Max token length for Llava text embedding extraction.",
    )
    parser.add_argument("--seed", default="0", type=int, help="Random seed for reproducibility.")
    return parser.parse_args()


def get_hypothesis_from_GPT(key, prompt, LLM="gpt-4o"):
    """
    Generates hypotheses and prompts using OpenAI GPT models via OpenAI Python SDK.

    Args:
        key (str): OpenAI API key.
        prompt (str): User-defined prompt for hypothesis generation.
        LLM (str): Model to use (e.g., "gpt-4o", "gpt-4-turbo").

    Returns:
        tuple: (hypothesis_dict, prompt_dict) extracted from the model's response.
    """
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        # model="gpt-4-turbo",
        model=LLM,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Help me with my problem!"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        temperature=0.0,
    )
    python_code = response.choices[0].message.content
    clean_python_code = python_code.strip("`").split("python\n")[1].strip()
    namespace = {}
    exec(clean_python_code, {}, namespace)

    hypothesis_dict = namespace.get("hypothesis_dict")
    prompt_dict = namespace.get("prompt_dict")

    print("Hypothesis Dictionary:", hypothesis_dict)
    print("Prompt Dictionary:", prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_claude(key, prompt):
    """
    Generates hypotheses and prompts using Anthropic Claude models.

    Args:
        key (str): API key for Anthropic Claude.
        prompt (str): User-defined prompt for Claude.

    Returns:
        tuple: (hypothesis_dict, prompt_dict) from Claude's response.
    """
    client = anthropic.Anthropic(api_key=key)

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    data = response.json()
    python_code = json.loads(data)["content"][0]["text"]

    clean_python_code = python_code.strip("`").split("python\n")[1]
    local_vars = {}
    exec(clean_python_code, {}, local_vars)

    hypothesis_dict = local_vars["hypothesis_dict"]
    prompt_dict = local_vars["prompt_dict"]

    print(hypothesis_dict)
    print("\n")
    print(prompt_dict)

    return hypothesis_dict, prompt_dict


def get_hypothesis_from_llama(key, prompt):
    """
    Generates hypotheses using Llama API (e.g., llama3-70b via LlamaAPI).

    Args:
        key (str): Llama API key.
        prompt (str): Prompt to send to the LLM.

    Returns:
        tuple: (hypothesis_dict, prompt_dict) as parsed from response.
    """
    from llamaapi import LlamaAPI

    llama = LlamaAPI(key)

    # Build the API request
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "function_call": "get_current_weather",
        "max_tokens": 2048,
    }

    # Execute the Request
    response = llama.run(api_request_json)
    data = response.json()

    python_code = data["choices"][0]["message"]["content"]
    python_code = re.sub(r"```python|```", "", python_code).strip()

    print(python_code)
    local_vars = {}
    exec(python_code, {}, local_vars)
    hypothesis_dict = local_vars.get("hypothesis_dict")
    prompt_dict = local_vars.get("prompt_dict")
    # Now the dictionaries hypothesis_dict and prompt_dict are available
    print("hypothesis_dict:")
    print(hypothesis_dict)
    print("\nprompt_dict:")
    print(prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_gemini(key, prompt):
    """
    Uses Gemini API (via google.generativeai) to extract hypothesis and prompt dictionaries.

    Args:
        key (str): API key for Gemini.
        prompt (str): Prompt text for Gemini model.

    Returns:
        tuple: (hypothesis_dict, prompt_dict)
    """
    import google.generativeai as genai

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    # print(response.text)

    clean_python_code = response.text.strip("`").split("python\n")[1]
    clean_python_code = clean_python_code.split("```")[0]  # Remove the ending ```

    local_vars = {}
    exec(clean_python_code, {}, local_vars)
    hypothesis_dict = local_vars.get("hypothesis_dict")
    prompt_dict = local_vars.get("prompt_dict")
    # Now the dictionaries hypothesis_dict and prompt_dict are available
    print("hypothesis_dict:")
    print(hypothesis_dict)
    print("\nprompt_dict:")
    print(prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_gemini_vertex(key, prompt):
    """
    Uses Google Vertex AI to call Gemini-1.5-flash and extract hypotheses.

    Args:
        key (str): Google Cloud project ID for VertexAI.
        prompt (str): Prompt string to generate hypotheses.

    Returns:
        tuple: (hypothesis_dict, prompt_dict)
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel, SafetySetting

    vertexai.init(project=key, location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-flash-002",
    )
    chat = model.start_chat()
    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 1,
        "top_p": 0.95,
    }
    safety_settings = [
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
        SafetySetting(
            category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=SafetySetting.HarmBlockThreshold.OFF,
        ),
    ]
    response = chat.send_message(
        [prompt], generation_config=generation_config, safety_settings=safety_settings
    )
    print(response)
    clean_python_code = response.text.strip("`").split("python\n")[1]
    clean_python_code = clean_python_code.split("```")[0]

    local_vars = {}
    exec(clean_python_code, {}, local_vars)
    hypothesis_dict = local_vars.get("hypothesis_dict")
    prompt_dict = local_vars.get("prompt_dict")
    print("hypothesis_dict:")
    print(hypothesis_dict)
    print("\nprompt_dict:")
    print(prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_GPT_azure_api(key, prompt, azure_params=None):
    """
    Calls Azure-hosted GPT model using REST API and extracts hypothesis/prompt dictionaries.

    Args:
        key (str): Azure OpenAI API key.
        prompt (str): Prompt string for the chat completion.
        azure_params (dict): Dictionary containing 'azure_endpoint', 'azure_deployment_name', and 'azure_api_version'.

    Returns:
        tuple: (hypothesis_dict, prompt_dict)
    """
    endpoint = azure_params["azure_endpoint"]
    deployment_name = azure_params["azure_deployment_name"]
    api_version = azure_params["azure_api_version"]

    headers = {"Content-Type": "application/json", "api-key": key}

    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Help me with my problem!"},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 50,
        "temperature": 0.7,
    }

    url = (
        f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    )

    # Make the API request
    response = requests.post(url, headers=headers, json=data)
    python_code = response.choices[0].message.content
    clean_python_code = python_code.strip("`").split("python\n")[1].strip()
    namespace = {}
    exec(clean_python_code, {}, namespace)

    hypothesis_dict = namespace.get("hypothesis_dict")
    prompt_dict = namespace.get("prompt_dict")

    print("Hypothesis Dictionary:", hypothesis_dict)
    print("Prompt Dictionary:", prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_LLM(LLM, key, prompt, hypothesis_dict_file, prompt_dict_file, azure_params):
    """
    Unified interface for getting hypothesis and prompt dicts from various LLM providers.

    Args:
        LLM (str): LLM provider ("gpt-4o", "claude", "gemini", etc).
        key (str): API key (or project ID for Vertex).
        prompt (str): The text prompt to be processed.
        hypothesis_dict_file (str): Path to cache file for saving/loading hypothesis dict.
        prompt_dict_file (str): Path to cache file for saving/loading prompt dict.
        azure_params (dict): Azure-specific parameters (optional).

    Returns:
        tuple: (hypothesis_dict, prompt_dict)
    """
    hypothesis_dict, prompt_dict = {}, {}

    if LLM.lower() == "gpt-4o" or LLM.lower() == "gpt-4-turbo" or LLM.lower() == "o1-preview":
        hypothesis_dict, prompt_dict = get_hypothesis_from_GPT(key, prompt, LLM=LLM)
    if LLM.lower() == "gpt-4o-azure-api":
        hypothesis_dict, prompt_dict = get_hypothesis_from_GPT_azure_api(key, prompt, azure_params)
    elif LLM.lower() == "claude":
        hypothesis_dict, prompt_dict = get_hypothesis_from_claude(key, prompt)
    elif LLM.lower() == "llama":
        hypothesis_dict, prompt_dict = get_hypothesis_from_llama(key, prompt)
    elif LLM.lower() == "gemini":
        hypothesis_dict, prompt_dict = get_hypothesis_from_gemini(key, prompt)
    elif LLM.lower() == "gemini-vertex":
        hypothesis_dict, prompt_dict = get_hypothesis_from_gemini_vertex(key, prompt)

    pickle.dump(hypothesis_dict, open(hypothesis_dict_file, "wb"))
    pickle.dump(prompt_dict, open(prompt_dict_file, "wb"))
    return hypothesis_dict, prompt_dict


def get_prompt_embedding(hyp_sent_list, embedding_backend, dataset_type="medical"):
    """
    Converts a list of hypothesis sentences into CLIP text embeddings.

    Args:
        hyp_sent_list (list of str): List of hypothesis prompts.
        embedding_backend: Embedding backend object.
        dataset_type (str): Either "medical" or "vision"; determines tokenizer and projection usage.

    Returns:
        torch.Tensor or np.ndarray: Normalized embeddings for the input hypotheses.
    """
    attr_embs = []
    for prompt in hyp_sent_list:
        if isinstance(prompt, list):
            prompt_texts = prompt
        else:
            prompt_texts = [prompt]
        text_emb = embedding_backend.encode_texts(prompt_texts, dataset_type=dataset_type)
        text_emb = text_emb.mean(dim=0, keepdim=True)
        text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
        attr_embs.append(text_emb)
    return torch.cat(attr_embs, dim=0)


def _extract_birads_numeric(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric
    extracted = series.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def _find_birads_column(df):
    preferred = [
        "breast_birads",
        "BIRADS",
        "birads",
        "birads_assessment",
        "birads_category",
        "assessment",
    ]
    lower_map = {str(col).lower(): col for col in df.columns}

    def _is_valid_birads_column(col_name):
        parsed = _extract_birads_numeric(df[col_name])
        return parsed.notna().any()

    for name in preferred:
        if name in df.columns:
            if _is_valid_birads_column(name):
                return name
        key = str(name).lower()
        if key in lower_map:
            candidate = lower_map[key]
            if _is_valid_birads_column(candidate):
                return candidate
    for col in df.columns:
        if "birads" in str(col).lower():
            if _is_valid_birads_column(col):
                return col
    return None


def _birads_label_from_passed_slice(series):
    birads_values = _extract_birads_numeric(series).dropna()
    if birads_values.empty:
        return "NA"
    mode_values = birads_values.mode(dropna=True)
    if mode_values.empty:
        return "NA"
    birads_value = float(mode_values.iloc[0])
    if birads_value.is_integer():
        return str(int(birads_value))
    return f"{birads_value:.2f}"


# Broad regex that strips any BI-RADS suffix produced by _expand_prompt_dict_with_birads,
# regardless of which format was used.  Matches both comma-prefixed forms
# ("…, BI-RADS …") and parenthesised forms ("… (BI-RADS …)").
_ANY_BIRADS_SUFFIX_RE = re.compile(
    r"(?:,\s*BI-RADS\b[^)]*\)?|(?<=\s)\(BI-RADS\b[^)]*\))\s*$",
    re.IGNORECASE,
)


def _has_birads_suffix(text):
    return bool(_ANY_BIRADS_SUFFIX_RE.search(str(text)))


BIRADS_DESCRIPTIONS = {
    1: "negative",
    2: "benign",
    3: "probably benign",
    4: "suspicious",
    5: "highly suggestive of malignancy",
}

# Named format presets.  Each value is a suffix template; use {level} for the
# BI-RADS level number and {desc} for the plain-English description.
# You can also pass a raw template string directly to _expand_prompt_dict_with_birads.
BIRADS_FORMAT_PRESETS = {
    "category": ", BI-RADS category {level}",
    "category_desc": ", BI-RADS category {level} ({desc})",
    "short": ", BI-RADS {level}",
    "parens": " (BI-RADS {level})",
    "parens_desc": " (BI-RADS {level}: {desc})",
}


def _resolve_birads_format(fmt: str) -> str:
    """Return the suffix template for *fmt*.

    If *fmt* is a key in BIRADS_FORMAT_PRESETS the corresponding template is
    returned; otherwise *fmt* is treated as a raw template string and returned
    unchanged.
    """
    return BIRADS_FORMAT_PRESETS.get(fmt, fmt)


def _expand_prompt_dict_with_birads(
    prompt_dict,
    birads_levels=(1, 2, 3, 4, 5),
    birads_formats=("category",),
):
    """
    Expands each hypothesis into one variant per (BI-RADS level × format).
    Handles prompt_dict values that are either a single string or a list of strings.
    Handles stale pickles by stripping any existing BI-RADS suffix before re-expanding.

    Args:
        birads_levels: Iterable of integer BI-RADS levels to include (default 1-5).
        birads_formats: Sequence of format names from BIRADS_FORMAT_PRESETS **or** raw
            template strings using ``{level}`` and ``{desc}`` placeholders.
            Multiple formats produce a cross-product with *birads_levels*.

            Built-in presets
            ----------------
            ``"category"``       →  ``, BI-RADS category {level}``       (default)
            ``"category_desc"``  →  ``, BI-RADS category {level} ({desc})``
            ``"short"``          →  ``, BI-RADS {level}``
            ``"parens"``         →  `` (BI-RADS {level})``
            ``"parens_desc"``    →  `` (BI-RADS {level}: {desc})``

            Custom example::

                birads_formats=[", assessment category {level} — {desc}"]
    """

    def _strip_sent(s):
        return _ANY_BIRADS_SUFFIX_RE.sub("", str(s)).rstrip()

    def _strip_value(value):
        if isinstance(value, list):
            return [_strip_sent(s) for s in value]
        return _strip_sent(value)

    def _append_birads(value, suffix):
        if isinstance(value, list):
            return [s + suffix for s in value]
        return value + suffix

    # Deduplicate to base hypotheses, stripping any stale BI-RADS suffix
    base_entries = {}
    for hyp, sentence in prompt_dict.items():
        base_hyp = _ANY_BIRADS_SUFFIX_RE.sub("", hyp).rstrip()
        if base_hyp not in base_entries:
            base_entries[base_hyp] = _strip_value(sentence)

    # Resolve format names → template strings once
    templates = [_resolve_birads_format(fmt) for fmt in birads_formats]

    expanded = {}
    for base_hyp, base_sentence in base_entries.items():
        for level in birads_levels:
            desc = BIRADS_DESCRIPTIONS[level]
            for template in templates:
                suffix = template.format(level=level, desc=desc)
                key = f"{base_hyp}{suffix}"
                expanded[key] = _append_birads(base_sentence, suffix)

    return expanded


def _expand_prompt_dict_with_mode_birads(
    verified_hypotheses,
    df,
    percentile,
    birads_col,
    class_label,
    birads_formats=("category",),
):
    """
    For each verified hypothesis, appends only the single modal BI-RADS level
    observed in the *passed* slice (rows where similarity score >= threshold).
    Produces one variant per hypothesis (× number of formats) rather than one
    per level × format as in :func:`_expand_prompt_dict_with_birads`.

    Args:
        verified_hypotheses: Dict mapping hypothesis key → sentence(s), as
            built inside :func:`discover_slices`.  Keys must correspond to
            columns already present in *df* (added during the slice loop).
        df: The working dataframe, which must already contain a column for
            every hypothesis in *verified_hypotheses*.
        percentile: Same percentile used in the slice loop.
        birads_col: Name of the BI-RADS column in *df* (from
            :func:`_find_birads_column`).  If ``None`` the function returns
            an empty dict.
        class_label: Integer class label used to filter positive-class rows.
        birads_formats: Sequence of format names / raw templates forwarded to
            :func:`_resolve_birads_format`.
    """
    if not birads_col:
        return {}

    templates = [_resolve_birads_format(fmt) for fmt in birads_formats]
    pt = df[df["out_put_GT"] == class_label]

    def _strip_sent(s):
        return _ANY_BIRADS_SUFFIX_RE.sub("", str(s)).rstrip()

    def _strip_value(value):
        if isinstance(value, list):
            return [_strip_sent(s) for s in value]
        return _strip_sent(value)

    # Deduplicate base hypotheses (strip any stale suffix from the key)
    base_entries = {}
    for hyp, sentence in verified_hypotheses.items():
        base_hyp = _ANY_BIRADS_SUFFIX_RE.sub("", hyp).rstrip()
        if base_hyp not in base_entries:
            base_entries[base_hyp] = _strip_value(sentence)

    expanded = {}
    for base_hyp, base_sentence in base_entries.items():
        if base_hyp not in df.columns:
            print(
                f"Warning: column '{base_hyp}' not found in df; skipping mode BI-RADS for this hypothesis."
            )
            continue

        th = np.percentile(df[base_hyp].values, percentile)
        passed_slice = pt[pt[base_hyp] >= th]
        mode_label = _birads_label_from_passed_slice(passed_slice[birads_col])

        if mode_label == "NA":
            print(f"Warning: no valid BI-RADS value in passed slice for '{base_hyp}'; skipping.")
            continue

        try:
            level = int(float(mode_label))
        except (ValueError, TypeError):
            level = mode_label

        desc = BIRADS_DESCRIPTIONS.get(level, "")
        print(f"Mode BI-RADS for '{base_hyp}': {level} ({desc})")

        for template in templates:
            suffix = template.format(level=level, desc=desc)
            key = f"{base_hyp}{suffix}"
            if isinstance(base_sentence, list):
                expanded[key] = [s + suffix for s in base_sentence]
            else:
                expanded[key] = base_sentence + suffix

    return expanded


def discover_slices(
    df,
    pred_col,
    prompt_dict,
    embedding_backend,
    image_emb_path,
    aligner_path,
    save_path,
    save_file,
    dataset_type="medical",
    percentile=75,
    class_label=1,
    out_file=None,
    append_birads_to_passed_hypothesis=False,
    birads_prompt_dict=None,
    birads_formats=("category",),
    birads_expansion_mode="all",
    backend="legacy",
):
    """
    Discovers data slices aligned with specific hypotheses by comparing aligned image and prompt embeddings.
    When append_birads_to_passed_hypothesis is True, each verified hypothesis is expanded into five
    variants (BI-RADS 1-5) after slice analysis confirms it is a meaningful error driver
    (acc_failed < acc_passed). This ensures BI-RADS expansion is only applied to hypotheses that
    have demonstrated a real performance gap.

    Args:
        df (pd.DataFrame): Input data with predictions and ground truth.
        pred_col (str): Column name for predictions.
        prompt_dict (dict): Dictionary mapping hypothesis names to prompts.
        embedding_backend: Loaded embedding backend.
        image_emb_path (str): Path to image embeddings (.npy).
        aligner_path (str): Path to the learned linear aligner (.pth).
        save_path (Path): Directory to save the output CSV.
        save_file (str): Output filename.
        dataset_type (str): "medical" or "vision" to adapt prompt processing.
        percentile (float): Threshold percentile to define slices.
        class_label (int): Class of interest (e.g., 1 for positive).
        out_file (str, optional): Path to write logs.
        append_birads_to_passed_hypothesis (bool): When True, expand only verified hypotheses
            (acc_failed < acc_passed) into five BI-RADS-qualified variants after slice analysis.
        birads_formats: Sequence of format names or raw templates forwarded to
            _expand_prompt_dict_with_birads.  See BIRADS_FORMAT_PRESETS for built-in
            options (default: ``("category",)``).
        birads_expansion_mode: ``"all"`` (default) expands each verified hypothesis
            into one variant per BI-RADS level 1-5.  ``"mode"`` appends only the
            single modal BI-RADS level observed in the passed slice for that
            hypothesis, producing one variant per hypothesis (requires a BI-RADS
            column to be present in *df*).

    Returns:
        dict: Updated prompt dictionary (BI-RADS-expanded verified hypotheses when
            append_birads_to_passed_hypothesis is True, otherwise all processed hypotheses).
    """

    hyp_sent_list = []
    hyp_list = []

    for key, value in prompt_dict.items():
        hyp_list.append(key)
        hyp_sent_list.append(value)

    attr_embs = get_prompt_embedding(hyp_sent_list, embedding_backend, dataset_type=dataset_type).float()
    if attr_embs.ndim == 1:
        attr_embs = attr_embs.unsqueeze(0)
    print(f"attr_embs: {attr_embs.size()}")

    print(df.shape)
    df_indx = df.index.tolist()
    img_emb = np.load(image_emb_path)
    print(img_emb.shape)
    img_emb = img_emb[df_indx]

    img_emb_tensor = torch.from_numpy(img_emb).float()
    if backend == "legacy":
        aligner = torch.load(aligner_path)
        W = aligner["W"]
        b = aligner["b"]
        img_emb_clip_tensor = img_emb_tensor @ W.T + b
    else:
        print("Skipping aligner projection because backend=llava_mammo.")
        img_emb_clip_tensor = img_emb_tensor

    sim_device = attr_embs.device
    sim_score = torch.matmul(img_emb_clip_tensor.to(sim_device).float(), attr_embs.to(sim_device).float().T)
    print(f"img_emb_clip_tensor: {img_emb_clip_tensor.size()}")
    print(f"sim_score size: {sim_score.size()}")

    acc = []
    updated_prompt_dict = {}
    verified_hypotheses = {}
    for idx, hyp in enumerate(hyp_list):
        print("==============================================")
        df[hyp] = sim_score[:, idx].cpu().numpy()
        pt = df[df["out_put_GT"] == class_label]
        print(f"total shape: {pt.shape}")
        th = np.percentile(df[hyp].values, percentile)
        err_slice = pt[pt[hyp] < th]
        gt = err_slice["out_put_GT"].values
        pred = err_slice[pred_col].values
        acc_failed = np.mean(gt == pred)
        print(
            f"Accuracy on the error slice (where attribute absent, the hypothesis failed): {acc_failed}"
        )
        print(
            f"Shape of the error slice (where attribute absent, the hypothesis failed): {err_slice.shape}"
        )

        err_slice = pt[pt[hyp] >= th]
        gt = err_slice["out_put_GT"].values
        pred = err_slice[pred_col].values
        acc_passed = np.mean(gt == pred)
        print(
            f"Accuracy on the bias aligned slice (where attribute present, the hypothesis passed): {acc_passed}"
        )
        print(
            f"Shape of the bias aligned slice (where attribute present, the hypothesis passed): {err_slice.shape}"
        )

        is_verified = acc_failed < acc_passed
        print(f"Hypothesis verified (acc_failed < acc_passed): {is_verified}")
        print(idx, hyp)
        updated_prompt_dict[hyp] = prompt_dict[hyp]
        if is_verified:
            verified_hypotheses[hyp] = prompt_dict[hyp]
        acc.append(acc_failed)

        df[f"{hyp}_bin"] = (df[hyp].values >= th).astype(int)
        print("==============================================")

        if out_file:
            with open(out_file, "a") as f:
                print("==============================================", file=f)
                print(idx, hyp, file=f)
                print(
                    f"Accuracy on the error slice (where attribute absent, the hypothesis failed): {acc_failed}",
                    file=f,
                )
                print(
                    f"Accuracy on the bias aligned slice (where attribute present, the hypothesis passed): {acc_passed}",
                    file=f,
                )
                print(f"Hypothesis verified (acc_failed < acc_passed): {is_verified}", file=f)
                print("==============================================", file=f)

    if append_birads_to_passed_hypothesis:
        # Use a pre-computed BI-RADS dict (e.g. from the train run) when available so that
        # valid/test splits get columns for the exact same hypotheses.  Only fall back to
        # re-verifying when no pre-computed dict has been supplied.
        effective_birads_dict = birads_prompt_dict
        if effective_birads_dict is None and verified_hypotheses:
            if birads_expansion_mode == "mode":
                birads_col = _find_birads_column(df)
                if birads_col:
                    print(f"Mode BI-RADS expansion: using column '{birads_col}'.")
                else:
                    print(
                        "Warning: birads_expansion_mode='mode' but no BI-RADS column found in df; "
                        "falling back to all-levels expansion."
                    )
                effective_birads_dict = (
                    _expand_prompt_dict_with_mode_birads(
                        verified_hypotheses,
                        df=df,
                        percentile=percentile,
                        birads_col=birads_col,
                        class_label=class_label,
                        birads_formats=birads_formats,
                    )
                    if birads_col
                    else _expand_prompt_dict_with_birads(
                        verified_hypotheses, birads_formats=birads_formats
                    )
                )
            else:
                effective_birads_dict = _expand_prompt_dict_with_birads(
                    verified_hypotheses, birads_formats=birads_formats
                )
            print(
                f"Expanded {len(verified_hypotheses)} verified hypotheses to "
                f"{len(effective_birads_dict)} BI-RADS variants "
                f"(mode: {birads_expansion_mode}, formats: {list(birads_formats)})."
            )
        elif effective_birads_dict is not None:
            print(
                f"Using {len(effective_birads_dict)} pre-computed BI-RADS variants for this split."
            )

        if effective_birads_dict:
            updated_prompt_dict = effective_birads_dict
            # Compute similarity scores for BI-RADS variants and add columns to df
            # so downstream scripts (mitigate_error_slices.py) can use them as column names.
            birads_hyp_list = list(effective_birads_dict.keys())
            birads_sent_list = list(effective_birads_dict.values())
            birads_attr_embs = get_prompt_embedding(
                birads_sent_list, embedding_backend, dataset_type=dataset_type
            )
            birads_sim_score = torch.matmul(
                img_emb_clip_tensor.to(sim_device).float(),
                birads_attr_embs.to(sim_device).float().T,
            )
            for bidx, birads_hyp in enumerate(birads_hyp_list):
                df[birads_hyp] = birads_sim_score[:, bidx].cpu().numpy()
                th = np.percentile(df[birads_hyp].values, percentile)
                df[f"{birads_hyp}_bin"] = (df[birads_hyp].values >= th).astype(int)
        else:
            print(
                "No verified hypotheses found and no pre-computed BI-RADS dict supplied; skipping BI-RADS expansion."
            )

    df.to_csv(save_path / save_file, index=False)
    print(f"Dataframe saved successfully at: {save_path / save_file}!")
    return updated_prompt_dict


# def discover_slices(
#         df, pred_col, prompt_dict, clip_model, clf_image_emb_path, aligner_path, save_path, save_file,
#         dataset_type="medical", percentile=75, class_label=1, out_file=None,
#         append_birads_to_passed_hypothesis=False):
#     """
#         Discovers data slices aligned with specific hypotheses by comparing aligned image and prompt embeddings.

#         Args:
#             df (pd.DataFrame): Input data with predictions and ground truth.
#             pred_col (str): Column name for predictions.
#             prompt_dict (dict): Dictionary mapping hypothesis names to prompts.
#             clip_model (dict): Loaded CLIP model and tokenizer.
#             clf_image_emb_path (str): Path to classifier embeddings (.npy).
#             aligner_path (str): Path to the learned linear aligner (.pth).
#             save_path (Path): Directory to save the output CSV.
#             save_file (str): Output filename.
#             dataset_type (str): "medical" or "vision" to adapt prompt processing.
#             percentile (float): Threshold percentile to define slices.
#             class_label (int): Class of interest (e.g., 1 for positive).
#             out_file (str, optional): Path to write logs.

#         Returns:
#             None
#     """
#     hyp_sent_list = []
#     hyp_list = []

#     for key, value in prompt_dict.items():
#         hyp_list.append(key)
#         hyp_sent_list.append(value)

#     attr_embs = get_prompt_embedding(hyp_sent_list, clip_model, dataset_type=dataset_type)
#     attr_embs = torch.tensor(attr_embs)
#     print(f"attr_embs: {attr_embs.size()}")

#     print(df.shape)
#     df_indx = df.index.tolist()
#     img_emb_clf = np.load(clf_image_emb_path)
#     print(img_emb_clf.shape)
#     img_emb_clf = img_emb_clf[df_indx]

#     aligner = torch.load(aligner_path)
#     W = aligner["W"]
#     b = aligner["b"]

#     img_emb_clf_tensor = torch.from_numpy(img_emb_clf)
#     img_emb_clip_tensor = img_emb_clf_tensor @ W.T + b
#     # print(type(img_emb_clip_tensor), type(attr_embs))
#     sim_score = torch.matmul(img_emb_clip_tensor.to("cuda").float(), attr_embs.to("cuda").float().T)
#     print(f"img_emb_clip_tensor: {img_emb_clip_tensor.size()}")
#     print(f"sim_score size: {sim_score.size()}")
#     birads_col = None
#     if append_birads_to_passed_hypothesis:
#         birads_col = _find_birads_column(df)
#         if birads_col:
#             print(f"Appending BI-RADS score from column: {birads_col}")
#         else:
#             print("Warning: --append_birads_to_passed_hypothesis enabled but no BI-RADS column was found.")

#     acc = []
#     updated_prompt_dict = {}
#     for idx, hyp in enumerate(hyp_list):
#         print("==============================================")
#         output_hyp = hyp
#         df[hyp] = sim_score[:, idx].cpu().numpy()
#         pt = df[df["out_put_GT"] == class_label]
#         print(f"total shape: {pt.shape}")
#         th = np.percentile(df[hyp].values, percentile)
#         err_slice = pt[pt[hyp] < th]
#         gt = err_slice["out_put_GT"].values
#         pred = err_slice[pred_col].values
#         acc_failed = np.mean(gt == pred)
#         print(f"Accuracy on the error slice (where attribute absent, the hypothesis failed): {acc_failed}")
#         print(f"Shape of the error slice (where attribute absent, the hypothesis failed): {err_slice.shape}")

#         err_slice = pt[pt[hyp] >= th]
#         gt = err_slice["out_put_GT"].values
#         pred = err_slice[pred_col].values
#         acc_passed = np.mean(gt == pred)
#         print(
#             f"Accuracy on the bias aligned slice (where attribute present, , the hypothesis passed): {acc_passed}")
#         print(
#             f"Shape of the bias aligned slice (where attribute present, , the hypothesis passed): {err_slice.shape}")

#         if append_birads_to_passed_hypothesis and birads_col and not _has_birads_suffix(hyp):
#             birads_score = _birads_label_from_passed_slice(err_slice[birads_col])
#             output_hyp = f"{hyp} with BI-RADS score of {birads_score}"
#             df.rename(columns={hyp: output_hyp}, inplace=True)
#         print(idx, output_hyp)
#         updated_prompt_dict[output_hyp] = prompt_dict[hyp]
#         acc.append(acc_failed)

#         df[f"{output_hyp}_bin"] = (df[output_hyp].values >= th).astype(int)
#         print("==============================================")

#         if out_file:
#             with open(out_file, 'a') as f:
#                 print("==============================================", file=f)
#                 print(idx, output_hyp, file=f)
#                 print(
#                     f"Accuracy on the error slice (where attribute absent, the hypothesis failed): {acc_failed}",
#                     file=f)
#                 print(
#                     f"Accuracy on the bias aligned slice (where attribute present, the hypothesis passed): {acc_passed}",
#                     file=f)
#                 print("==============================================", file=f)

#     df.to_csv(save_path / save_file, index=False)
#     print(f"Dataframe saved successfully at: {save_path / save_file}!")
#     return updated_prompt_dict


def validate_error_slices_via_LLM(
    LLM,
    key,
    save_path,
    clf_results_csv,
    image_emb_path,
    aligner_path,
    prompt,
    embedding_backend,
    prediction_col,
    dataset_type="medical",
    mode="valid",
    class_label="",
    percentile=75,
    out_file=None,
    azure_params=None,
    append_birads_to_passed_hypothesis=False,
    birads_formats=("category",),
    birads_expansion_mode="all",
    backend="legacy",
):
    """
    Validates automatically discovered error slices by using a prompt-based LLM to generate hypotheses.

    Args:
        LLM (str): Name of the LLM ("gpt-4o", "claude", "gemini", etc.).
        key (str): API key for the LLM.
        save_path (Path): Path to save generated outputs and logs.
        clf_results_csv (str): Path to the classifier result CSV.
        image_emb_path (str): Path to image embeddings.
        aligner_path (str): Path to the saved aligner weights.
        prompt (str): Prompt text to send to the LLM.
        embedding_backend: The embedding backend object.
        prediction_col (str): Column name for predicted probabilities or labels.
        dataset_type (str): "medical" or "vision".
        mode (str): Dataset split ("train", "valid", or "test").
        class_label (str): Class label of interest ("pneumothorax", "mass", etc.).
        percentile (int): Threshold percentile for slice creation.
        out_file (str, optional): Path to a text file to write evaluation logs.
        azure_params (dict, optional): Configs for Azure OpenAI API if using it.

    Returns:
        None
    """
    df = pd.read_csv(clf_results_csv)
    if prediction_col == "out_put_predict":
        df["Predictions_bin"] = (df[prediction_col] >= 0.5).astype(int)
        pred_col = "Predictions_bin"
    else:
        pred_col = prediction_col
    print(f"Prediction column: {pred_col}")
    print(f"\ndf: {df.shape}")

    print(
        "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Prompt start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )
    print(prompt)
    with open(save_path / "prompt.txt", "w") as file:
        file.write(prompt)
    print(
        "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Prompt End >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    )

    hypothesis_dict_file = save_path / f"{class_label}_hypothesis_dict.pkl"
    # base_prompt_dict_file caches the raw LLM output and is never overwritten with BI-RADS
    # expansions, so valid/test runs always load clean base hypotheses.
    base_prompt_dict_file = save_path / f"{class_label}_base_prompt_dict.pkl"
    # prompt_dict_file is the final output consumed by downstream scripts (mitigate, evaluate).
    # When append_birads_to_passed_hypothesis is True it will contain the BI-RADS expansion;
    # otherwise it is written once with the base hypotheses.
    prompt_dict_file = save_path / f"{class_label}_prompt_dict.pkl"
    if hypothesis_dict_file.exists() and base_prompt_dict_file.exists():
        hypothesis_dict = pickle.load(open(hypothesis_dict_file, "rb"))
        prompt_dict = pickle.load(open(base_prompt_dict_file, "rb"))
    else:
        hypothesis_dict, prompt_dict = get_hypothesis_from_LLM(
            LLM, key, prompt, hypothesis_dict_file, base_prompt_dict_file, azure_params
        )
        # Also seed prompt_dict_file with base hypotheses so downstream scripts have
        # something to read even when BI-RADS expansion is disabled.
        pickle.dump(prompt_dict, open(prompt_dict_file, "wb"))

    print("<<<<<====================================================>>>>")
    print("Hypothesis Dictionary:")
    print(hypothesis_dict)
    print("\nPrompt Dictionary:")
    print(prompt_dict)
    print("<<<<<====================================================>>>>")

    if out_file:
        with open(out_file, "w") as f:
            print("Hypothesis Dictionary:", file=f)
            print(hypothesis_dict, file=f)
            print("\nPrompt Dictionary:", file=f)
            print(prompt_dict, file=f)

    if (
        class_label.lower() == "landbirds"
        or class_label.lower() == "dog"
        or class_label.lower() == "urban"
    ):
        class_idx = 0
    else:
        class_idx = 1
    print(f"class_label (class_idx): {class_label} ({class_idx})")
    birads_prompt_dict_file = save_path / f"{class_label}_birads_prompt_dict.pkl"
    existing_birads_prompt_dict = None
    if append_birads_to_passed_hypothesis and birads_prompt_dict_file.exists():
        existing_birads_prompt_dict = pickle.load(open(birads_prompt_dict_file, "rb"))
        print(f"Loaded existing BI-RADS prompt dictionary from {birads_prompt_dict_file}")

    updated_prompt_dict = discover_slices(
        df,
        pred_col,
        prompt_dict,
        embedding_backend,
        image_emb_path,
        aligner_path,
        save_path,
        save_file=f"{mode}_{class_label}_dataframe_mitigation.csv",
        dataset_type=dataset_type,
        percentile=percentile,
        class_label=class_idx,
        out_file=out_file,
        append_birads_to_passed_hypothesis=append_birads_to_passed_hypothesis,
        birads_prompt_dict=existing_birads_prompt_dict,
        birads_formats=birads_formats,
        birads_expansion_mode=birads_expansion_mode,
        backend=backend,
    )
    if (
        append_birads_to_passed_hypothesis
        and updated_prompt_dict
        and any(_has_birads_suffix(k) for k in updated_prompt_dict)
        and not birads_prompt_dict_file.exists()
    ):
        # Save the BI-RADS expansion to a separate file for downstream scripts.
        # - Only saved on the first (train) run; valid/test runs skip because the file already exists.
        # - prompt_dict_file is intentionally never overwritten so valid/test runs always
        #   load the original base hypotheses and avoid the stale-pickle bug.
        pickle.dump(updated_prompt_dict, open(birads_prompt_dict_file, "wb"))
        print(f"Saved BI-RADS prompt dictionary to: {birads_prompt_dict_file}")


def validate_error_slices_via_sent(
    LLM,
    key,
    dataset,
    save_path,
    clf_results_csv,
    image_emb_path,
    aligner_path,
    top50_err_text,
    embedding_backend,
    class_label,
    prediction_col,
    mode="test",
    out_file=None,
    azure_params=None,
    append_birads_to_passed_hypothesis=False,
    birads_formats=("category",),
    birads_expansion_mode="all",
    backend="legacy",
):
    """
    Wrapper function that reads top-50 failure text, constructs dataset-specific prompt,
    and triggers error slice validation via LLM.

    Args:
        LLM (str): LLM name to use for hypothesis generation.
        key (str): API key for the LLM.
        dataset (str): Name of the dataset ("nih", "rsna", "celeba", etc.).
        save_path (Path): Directory where outputs are saved.
        clf_results_csv (str): Path to classifier results CSV.
        image_emb_path (str): Path to image embeddings.
        aligner_path (str): Path to linear aligner weights.
        top50_err_text (str): Text file with top-50 error samples.
        embedding_backend: Loaded embedding backend.
        class_label (str): Class label to analyze.
        prediction_col (str): Name of prediction column.
        mode (str): Dataset split (e.g., "train", "valid", "test").
        out_file (str, optional): Path to output log file.
        azure_params (dict, optional): Azure config parameters.

    Returns:
        None
    """
    with open(top50_err_text) as file:
        content = file.read()
    if dataset.lower() == "nih":
        prompt = create_NIH_prompts(content)
        validate_error_slices_via_LLM(
            LLM,
            key,
            save_path,
            clf_results_csv,
            image_emb_path,
            aligner_path,
            prompt,
            embedding_backend,
            prediction_col,
            dataset_type="medical",
            mode=mode,
            class_label=class_label,
            percentile=55,
            out_file=out_file,
            azure_params=azure_params,
            append_birads_to_passed_hypothesis=append_birads_to_passed_hypothesis,
            birads_formats=birads_formats,
            birads_expansion_mode=birads_expansion_mode,
            backend=backend,
        )
    elif is_mammo_dataset(dataset):
        prompt = create_RSNA_prompts(content)
        validate_error_slices_via_LLM(
            LLM,
            key,
            save_path,
            clf_results_csv,
            image_emb_path,
            aligner_path,
            prompt,
            embedding_backend,
            prediction_col,
            dataset_type="medical",
            mode=mode,
            class_label=class_label,
            percentile=40,
            out_file=out_file,
            append_birads_to_passed_hypothesis=append_birads_to_passed_hypothesis,
            birads_formats=birads_formats,
            birads_expansion_mode=birads_expansion_mode,
            backend=backend,
        )
    elif dataset.lower() == "celeba":
        prompt = create_CELEBA_prompts(content)
        validate_error_slices_via_LLM(
            LLM,
            key,
            save_path,
            clf_results_csv,
            image_emb_path,
            aligner_path,
            prompt,
            embedding_backend,
            prediction_col,
            dataset_type="vision",
            mode=mode,
            class_label=class_label,
            percentile=50,
            out_file=out_file,
            append_birads_to_passed_hypothesis=append_birads_to_passed_hypothesis,
            birads_formats=birads_formats,
            birads_expansion_mode=birads_expansion_mode,
            backend=backend,
        )
    elif dataset.lower() == "waterbirds":
        prompt = create_Waterbirds_prompts(content)
        validate_error_slices_via_LLM(
            LLM,
            key,
            save_path,
            clf_results_csv,
            image_emb_path,
            aligner_path,
            prompt,
            embedding_backend,
            prediction_col,
            dataset_type="vision",
            mode=mode,
            class_label=class_label,
            percentile=55,
            out_file=out_file,
            append_birads_to_passed_hypothesis=append_birads_to_passed_hypothesis,
            birads_formats=birads_formats,
            birads_expansion_mode=birads_expansion_mode,
            backend=backend,
        )
    elif dataset.lower() == "metashift":
        cat_prompt, dog_prompt = create_Metashift_prompts(content)
        prompt = None
        if class_label.lower() == "cat":
            prompt = cat_prompt
        elif class_label.lower() == "dog":
            prompt = dog_prompt

        validate_error_slices_via_LLM(
            LLM,
            key,
            save_path,
            clf_results_csv,
            image_emb_path,
            aligner_path,
            prompt,
            embedding_backend,
            prediction_col,
            dataset_type="vision",
            mode=mode,
            class_label=class_label,
            percentile=55,
            out_file=out_file,
            append_birads_to_passed_hypothesis=append_birads_to_passed_hypothesis,
            birads_formats=birads_formats,
            birads_expansion_mode=birads_expansion_mode,
            backend=backend,
        )


def main(args):
    seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.backend == "llava_mammo" and not is_mammo_dataset(args.dataset):
        raise ValueError(
            "llava_mammo backend in validate_error_slices_w_LLM.py is supported only for mammography datasets."
        )
    if args.backend == "legacy" and not args.aligner_path:
        raise ValueError("--aligner_path is required when --backend legacy is selected.")

    args.aligner_path = args.aligner_path.format(args.seed) if args.aligner_path else ""
    args.top50_err_text = args.top50_err_text.format(args.seed)
    args.save_path = Path(args.save_path.format(args.seed))
    if Path(args.save_path).exists():
        for pattern in [
            f"{args.class_label}_*_dict.pkl",
            f"*_{args.class_label}_dataframe_mitigation.csv",
            "prompt.txt",
            f"ladder_validate_slices_w_LLM-{args.class_label}.txt",
        ]:
            for old_file in args.save_path.glob(pattern):
                old_file.unlink()
                print(f"Deleted old save: {old_file}")
    args.save_path.mkdir(parents=True, exist_ok=True)
    out_file = args.save_path / f"ladder_validate_slices_w_LLM-{args.class_label}.txt"

    print("\n")
    print(f"Embedding backend: {args.backend}")
    print(args.save_path)

    embedding_backend = create_embedding_backend(args)
    azure_params = {
        "azure_api_version": args.azure_api_version,
        "azure_endpoint": args.azure_endpoint,
        "azure_deployment_name": args.azure_deployment_name,
    }
    print("####################" * 10)
    if args.prediction_col == "out_put_predict":
        clf_results_csv = args.clf_results_csv.format(args.seed, "train")
        image_emb_path = (
            args.image_emb_path.format(args.seed, "train")
            if args.image_emb_path
            else args.clf_image_emb_path.format(args.seed, "train")
        )
        print("\n")
        print(args.save_path)
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Train <<<<<======================================="
        )
        validate_error_slices_via_sent(
            args.LLM,
            args.key,
            args.dataset,
            args.save_path,
            clf_results_csv,
            image_emb_path,
            args.aligner_path,
            args.top50_err_text,
            embedding_backend,
            args.class_label,
            args.prediction_col,
            mode="train",
            azure_params=azure_params,
            append_birads_to_passed_hypothesis=args.append_birads_to_passed_hypothesis,
            birads_formats=args.birads_formats,
            birads_expansion_mode=args.birads_expansion_mode,
            backend=args.backend,
        )

        clf_results_csv = args.clf_results_csv.format(args.seed, "valid")
        image_emb_path = (
            args.image_emb_path.format(args.seed, "valid")
            if args.image_emb_path
            else args.clf_image_emb_path.format(args.seed, "valid")
        )
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Valid <<<<<======================================="
        )
        validate_error_slices_via_sent(
            args.LLM,
            args.key,
            args.dataset,
            args.save_path,
            clf_results_csv,
            image_emb_path,
            args.aligner_path,
            args.top50_err_text,
            embedding_backend,
            args.class_label,
            args.prediction_col,
            mode="valid",
            azure_params=azure_params,
            append_birads_to_passed_hypothesis=args.append_birads_to_passed_hypothesis,
            birads_formats=args.birads_formats,
            birads_expansion_mode=args.birads_expansion_mode,
            backend=args.backend,
        )

        clf_results_csv = args.clf_results_csv.format(args.seed, "test")
        image_emb_path = (
            args.image_emb_path.format(args.seed, "test")
            if args.image_emb_path
            else args.clf_image_emb_path.format(args.seed, "test")
        )
        print("\n")
        print(args.save_path)
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Test <<<<<======================================="
        )
        validate_error_slices_via_sent(
            args.LLM,
            args.key,
            args.dataset,
            args.save_path,
            clf_results_csv,
            image_emb_path,
            args.aligner_path,
            args.top50_err_text,
            embedding_backend,
            args.class_label,
            args.prediction_col,
            mode="test",
            out_file=out_file,
            azure_params=azure_params,
            append_birads_to_passed_hypothesis=args.append_birads_to_passed_hypothesis,
            birads_formats=args.birads_formats,
            birads_expansion_mode=args.birads_expansion_mode,
            backend=args.backend,
        )

    else:
        clf_results_csv = args.clf_results_csv.format(args.seed, "test")
        image_emb_path = (
            args.image_emb_path.format(args.seed, "test")
            if args.image_emb_path
            else args.clf_image_emb_path.format(args.seed, "test")
        )
        print("\n")
        print(args.save_path)
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Test <<<<<======================================="
        )
        validate_error_slices_via_sent(
            args.LLM,
            args.key,
            args.dataset,
            args.save_path,
            clf_results_csv,
            image_emb_path,
            args.aligner_path,
            args.top50_err_text,
            embedding_backend,
            args.class_label,
            args.prediction_col,
            mode="test",
            out_file=out_file,
            azure_params=azure_params,
            append_birads_to_passed_hypothesis=args.append_birads_to_passed_hypothesis,
            birads_formats=args.birads_formats,
            birads_expansion_mode=args.birads_expansion_mode,
            backend=args.backend,
        )

    print("Completed")
    print(
        f"Check logs for test set: {args.save_path / f'ladder_validate_slices_w_LLM-{args.class_label}.txt'}"
    )


if __name__ == "__main__":
    _args = config()
    main(_args)
