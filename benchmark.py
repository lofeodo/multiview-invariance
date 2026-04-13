#!/usr/bin/env python3
"""
benchmark.py — Benchmark a VLM on the multiview-invariance dataset.

Each query shows the model a rendered scene image (view_N.png) with a
colored arrow overlaid. The model must predict spatial relations from the
arrow's imagined perspective. Ground truth is the arrow-viewpoint spatial
relations stored in the dataset index.

Supported models:
    chatgpt  — OpenAI API (gpt-4o by default)
    gemini   — Google Generative AI API (gemini-1.5-flash by default)
    llava    — LLaVA loaded locally via transformers
    qwen     — Qwen-VL-Chat loaded locally via transformers

Results are saved to:
    results/<model>_<timestamp>/
        config.json        — run configuration
        predictions.jsonl  — per-query predictions and ground truth
        metrics.json       — all computed metrics

Usage:
    python benchmark.py --model chatgpt --api_key sk-...
    python benchmark.py --model gemini  --api_key AI...
    python benchmark.py --model llava   --n_viewpoints 100
    python benchmark.py --model qwen    --axes 0 1
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import TypeAdapter, ValidationError
from tqdm import tqdm

# Validates that a JSON object is exactly dict[str, bool] (pydantic coerces
# bare "true"/"false" strings to bool automatically)
_BOOL_DICT = TypeAdapter(dict[str, bool])

from dataset import MultiviewDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AXIS_NAMES: dict[int, str] = {0: "lateral", 1: "depth", 2: "vertical"}

# Canonical ground-truth key pairs (positive, negative) per axis
AXIS_KEYS: dict[int, tuple[str, str]] = {
    0: ("A_left_of_B",     "A_right_of_B"),
    1: ("A_in_front_of_B", "A_behind_B"),
    2: ("A_above_B",       "A_below_B"),
}

# Fragments used when building prompt JSON keys like "{A_color}_left_of_{B_color}"
AXIS_PROMPT_POS: dict[int, str] = {0: "left_of",     1: "in_front_of", 2: "above"}
AXIS_PROMPT_NEG: dict[int, str] = {0: "right_of",    1: "behind",      2: "below"}

DIFFICULTY_BINS: list[tuple[str, float, float]] = [
    ("aligned",   0,    30),
    ("slight",   30,    60),
    ("moderate", 60,   120),
    ("strong",  120,   150),
    ("extreme", 150,   181),   # 181 to include exactly 180°
]


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------

def _load_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


class ModelAdapter:
    def query(self, image_path: Path, prompt: str) -> str:
        raise NotImplementedError


class ChatGPTAdapter(ModelAdapter):
    def __init__(self, api_key: str, model_id: str = "gpt-4o") -> None:
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai package not installed. Run: pip install openai")
        self._client = OpenAI(api_key=api_key)
        self._model_id = model_id

    def query(self, image_path: Path, prompt: str) -> str:
        b64 = _load_b64(image_path)
        resp = self._client.chat.completions.create(
            model=self._model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            max_tokens=512,
        )
        return resp.choices[0].message.content


class GeminiAdapter(ModelAdapter):
    def __init__(self, api_key: str, model_id: str = "gemini-1.5-flash") -> None:
        try:
            import google.generativeai as genai
        except ImportError:
            sys.exit("google-generativeai package not installed. Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_id)

    def query(self, image_path: Path, prompt: str) -> str:
        resp = self._model.generate_content([prompt, Image.open(image_path)])
        return resp.text


class LLaVAAdapter(ModelAdapter):
    def __init__(self, model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf") -> None:
        try:
            import torch
            from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        except ImportError:
            sys.exit("transformers/torch not installed. Run: pip install transformers torch")
        import torch
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._processor = LlavaNextProcessor.from_pretrained(model_id)
        self._model_obj = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto",
        )

    def query(self, image_path: Path, prompt: str) -> str:
        import torch
        img = Image.open(image_path).convert("RGB")
        conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(conv, add_generation_prompt=True)
        inputs = self._processor(img, text, return_tensors="pt").to(self._model_obj.device)
        with torch.no_grad():
            out = self._model_obj.generate(**inputs, max_new_tokens=256)
        decoded = self._processor.decode(out[0], skip_special_tokens=True)
        # Strip the prompt portion that precedes the model's reply
        marker = "[/INST]"
        idx = decoded.rfind(marker)
        return decoded[idx + len(marker):].strip() if idx != -1 else decoded.strip()


class QwenAdapter(ModelAdapter):
    def __init__(self, model_id: str = "Qwen/Qwen-VL-Chat") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            sys.exit("transformers/torch not installed. Run: pip install transformers torch")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["visual"],
        )
        self._model_obj = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, quantization_config=bnb_config,
            fp16=True,
        ).eval()

    def query(self, image_path: Path, prompt: str) -> str:
        qry = self._tokenizer.from_list_format([
            {"image": str(image_path.resolve())},
            {"text": prompt},
        ])
        resp, _ = self._model_obj.chat(self._tokenizer, query=qry, history=None)
        return resp


def make_adapter(model_name: str, api_key: str | None, model_id: str | None) -> ModelAdapter:
    defaults = {
        "chatgpt": "gpt-4o",
        "gemini":  "gemini-1.5-flash",
        "llava":   "llava-hf/llava-v1.6-mistral-7b-hf",
        "qwen":    "Qwen/Qwen-VL-Chat",
    }
    mid = model_id or defaults.get(model_name, "")
    if model_name == "chatgpt":
        if not api_key:
            sys.exit("--api_key is required for --model chatgpt")
        return ChatGPTAdapter(api_key, mid)
    if model_name == "gemini":
        if not api_key:
            sys.exit("--api_key is required for --model gemini")
        return GeminiAdapter(api_key, mid)
    if model_name == "llava":
        return LLaVAAdapter(mid)
    if model_name == "qwen":
        return QwenAdapter(mid)
    sys.exit(f"Unknown model {model_name!r}. Choose from: chatgpt, gemini, llava, qwen")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(group: dict, axes: list[int], attempt: int = 0) -> str:
    a_color   = group["object_A"]["color"]
    b_color   = group["object_B"]["color"]
    a_label   = group["object_A"]["label"]
    b_label   = group["object_B"]["label"]
    arr_color = group["reference_object_arrow"]["color"]

    kv_lines: list[str] = []
    for ax in axes:
        kv_lines.append(f'  "{a_color}_{AXIS_PROMPT_POS[ax]}_{b_color}": <true or false>')
        kv_lines.append(f'  "{a_color}_{AXIS_PROMPT_NEG[ax]}_{b_color}": <true or false>')
    json_template = "{\n" + ",\n".join(kv_lines) + "\n}"

    retry_prefix = (
        "IMPORTANT: Your previous response could not be parsed. "
        "You MUST output ONLY the raw JSON object below — "
        "no explanation, no markdown fences, no preamble, no trailing text.\n\n"
    ) if attempt > 0 else ""

    return (
        f"{retry_prefix}"
        f"You are standing at the {arr_color} arrow in this image, "
        f"looking in the direction it points.\n\n"
        f"From that perspective, judge the spatial relations of the "
        f"{a_color} object ({a_label}) relative to the {b_color} object ({b_label}).\n\n"
        f"Rules: for each opposite pair (left/right, in_front/behind, above/below), "
        f"at most one can be true — they cannot both be true at the same time.\n\n"
        f"Output ONLY the JSON object below. "
        f"Replace every <true or false> with true or false (lowercase, no quotes). "
        f"Do not write anything before or after the JSON object.\n\n"
        f"{json_template}"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(
    raw: str,
    group: dict,
    axes: list[int],
) -> tuple[dict[str, bool] | None, str | None]:
    """Extract and validate the model's JSON from *raw*.

    Returns ``(predicted_dict, None)`` on success or ``(None, error_msg)`` on failure.
    The returned dict uses canonical keys like ``"A_left_of_B"``.
    """
    a_color = group["object_A"]["color"]
    b_color = group["object_B"]["color"]

    # Map prompt keys → canonical keys
    pk_to_ck: dict[str, str] = {}
    for ax in axes:
        pos_k, neg_k = AXIS_KEYS[ax]
        pk_to_ck[f"{a_color}_{AXIS_PROMPT_POS[ax]}_{b_color}"] = pos_k
        pk_to_ck[f"{a_color}_{AXIS_PROMPT_NEG[ax]}_{b_color}"] = neg_k

    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return None, f"no JSON object found in response: {raw[:200]!r}"

    try:
        data = _BOOL_DICT.validate_python(json.loads(m.group()))
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"
    except ValidationError as exc:
        return None, f"schema validation error: {exc}"

    expected = set(pk_to_ck)
    missing  = expected - data.keys()
    extra    = data.keys() - expected
    if missing:
        return None, f"missing keys: {sorted(missing)}"
    if extra:
        return None, f"unexpected keys: {sorted(extra)}"

    return {ck: data[pk] for pk, ck in pk_to_ck.items()}, None


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------

def is_structurally_invalid(pred: dict[str, bool], axes: list[int]) -> bool:
    """Return True if any axis has both directions true, or all values are false."""
    for ax in axes:
        pos_k, neg_k = AXIS_KEYS[ax]
        if pred.get(pos_k, False) and pred.get(neg_k, False):
            return True
    all_vals = [pred.get(k, False) for ax in axes for k in AXIS_KEYS[ax]]
    return bool(all_vals) and not any(all_vals)


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def decode_label(pos_val: bool, neg_val: bool) -> str:
    """Map (positive_direction, negative_direction) booleans to a category string."""
    if pos_val and neg_val:
        return "invalid"
    if pos_val:
        return "positive"   # left / in_front_of / above
    if neg_val:
        return "negative"   # right / behind / below
    return "neither"


def get_difficulty_bin(yaw: float) -> str:
    yaw = abs(yaw)
    for name, lo, hi in DIFFICULTY_BINS:
        if lo <= yaw < hi:
            return name
    return "extreme"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_group_consistency(records: list[dict], axes: list[int]) -> dict:
    """Compute cross-viewpoint consistency metrics at the group level.

    Both viewpoints in a group share the same ground truth, so this measures
    whether the model produces invariant predictions regardless of camera angle.

    Requires records to have ``group_id``, ``predicted`` (or None), and
    ``is_invalid`` fields.
    """
    # Group records by group_id, keeping only groups with exactly 2 parsed,
    # structurally valid predictions (one per viewpoint).
    from collections import defaultdict
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r["predicted"] is not None and not r["is_invalid"]:
            by_group[r["group_id"]].append(r)

    complete_pairs = {gid: recs for gid, recs in by_group.items() if len(recs) == 2}
    n_pairs = len(complete_pairs)

    if n_pairs == 0:
        return {
            "n_complete_pairs": 0,
            "note": "No groups with two clean predictions available.",
        }

    n_consistent         = 0   # identical prediction on all active axes
    n_consistent_correct = 0   # identical AND both correct on all axes
    n_consistent_wrong   = 0   # identical AND both wrong on at least one axis
    n_inconsistent       = 0   # at least one axis flips between viewpoints

    per_axis_consistent  = {ax: 0 for ax in axes}  # axis-level agreement count
    per_axis_flip        = {ax: 0 for ax in axes}  # axis-level flip count

    for recs in complete_pairs.values():
        r0, r1 = recs[0], recs[1]

        ax_agrees  = {}
        ax_correct = {}  # correct on BOTH viewpoints
        for ax in axes:
            lbl0_pred = _decoded(r0, ax, "predicted")
            lbl1_pred = _decoded(r1, ax, "predicted")
            lbl_gt    = _decoded(r0, ax, "ground_truth")   # same for both

            agrees = lbl0_pred == lbl1_pred
            ax_agrees[ax]  = agrees
            ax_correct[ax] = agrees and (lbl0_pred == lbl_gt)

            if agrees:
                per_axis_consistent[ax] += 1
            else:
                per_axis_flip[ax] += 1

        fully_consistent = all(ax_agrees.values())
        if fully_consistent:
            n_consistent += 1
            if all(ax_correct.values()):
                n_consistent_correct += 1
            else:
                n_consistent_wrong += 1
        else:
            n_inconsistent += 1

    def _r(x: int) -> float:
        return round(x / n_pairs, 6) if n_pairs else 0.0

    per_axis_out: dict[str, dict] = {}
    for ax in axes:
        per_axis_out[AXIS_NAMES[ax]] = {
            "consistency_rate": round(per_axis_consistent[ax] / n_pairs, 6) if n_pairs else 0.0,
            "flip_rate":        round(per_axis_flip[ax]        / n_pairs, 6) if n_pairs else 0.0,
        }

    return {
        "n_complete_pairs":        n_pairs,
        "consistent_rate":         _r(n_consistent),
        "consistent_correct_rate": _r(n_consistent_correct),
        "consistent_wrong_rate":   _r(n_consistent_wrong),
        "inconsistent_rate":       _r(n_inconsistent),
        "per_axis":                per_axis_out,
    }

def _f1_macro(y_true: list[str], y_pred: list[str]) -> float:
    classes = ["positive", "negative", "neither"]
    f1s: list[float] = []
    for cls in classes:
        tp = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        fp = sum(t != cls and p == cls for t, p in zip(y_true, y_pred))
        fn = sum(t == cls and p != cls for t, p in zip(y_true, y_pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _decoded(record: dict, ax: int, field: str) -> str:
    pos_k, neg_k = AXIS_KEYS[ax]
    d = record[field]
    return decode_label(d.get(pos_k, False), d.get(neg_k, False))


def compute_metrics(
    records: list[dict],
    axes: list[int],
    _include_bins: bool = True,
) -> dict:
    """Compute all benchmark metrics over *records*.

    Each record must have fields: ``predicted`` (dict or None), ``ground_truth``
    (dict), ``is_invalid`` (bool), ``difficulty_bin`` (str).
    """
    n_total  = len(records)
    parsed   = [r for r in records if r["predicted"] is not None]
    n_parsed = len(parsed)
    n_struct_invalid = sum(1 for r in parsed if r["is_invalid"])
    clean    = [r for r in parsed if not r["is_invalid"]]
    n_clean  = len(clean)

    # ── Axis metrics ──────────────────────────────────────────────────────
    axis_metrics: dict[str, dict] = {}
    for ax in axes:
        gt_lbls   = [_decoded(r, ax, "ground_truth") for r in clean]
        pred_lbls = [_decoded(r, ax, "predicted")    for r in clean]
        acc = sum(g == p for g, p in zip(gt_lbls, pred_lbls)) / n_clean if n_clean else 0.0
        axis_metrics[AXIS_NAMES[ax]] = {
            "accuracy":  round(acc, 6),
            "macro_f1":  round(_f1_macro(gt_lbls, pred_lbls), 6),
        }

    # ── Joint metrics ─────────────────────────────────────────────────────
    partial_counts: dict[str, int] = defaultdict(int)
    n_exact = 0
    for r in clean:
        ax_correct = {ax: (_decoded(r, ax, "ground_truth") == _decoded(r, ax, "predicted"))
                      for ax in axes}
        if all(ax_correct.values()):
            n_exact += 1
        correct_names = [AXIS_NAMES[ax] for ax in axes if ax_correct[ax]]
        n_c = len(correct_names)
        if n_c == len(axes):
            lbl = "all_axes_correct"
        elif n_c == 0:
            lbl = "all_wrong"
        elif n_c == 1:
            lbl = f"{correct_names[0]}_only"
        else:
            lbl = "_and_".join(correct_names)
        partial_counts[lbl] += 1

    partial_breakdown = {
        lbl: {"count": cnt, "fraction": round(cnt / n_clean, 6) if n_clean else 0.0}
        for lbl, cnt in sorted(partial_counts.items(), key=lambda kv: -kv[1])
    }

    joint_metrics = {
        "exact_match_accuracy": round(n_exact / n_clean, 6) if n_clean else 0.0,
        "partial_correctness":  partial_breakdown,
    }

    # ── Confusion matrices ────────────────────────────────────────────────
    # Rows = ground truth label, columns = predicted label.
    # "parsed" includes structurally invalid predictions so "invalid" column
    # can be non-zero.
    col_labels = ["positive", "negative", "neither", "invalid"]
    row_labels  = ["positive", "negative", "neither"]
    conf_matrices: dict[str, dict] = {}
    for ax in axes:
        matrix: dict[str, dict[str, int]] = {r: {c: 0 for c in col_labels} for r in row_labels}
        for rec in parsed:
            gt_lbl   = _decoded(rec, ax, "ground_truth")
            pred_lbl = _decoded(rec, ax, "predicted")
            if gt_lbl in row_labels:
                matrix[gt_lbl][pred_lbl] = matrix[gt_lbl].get(pred_lbl, 0) + 1
        conf_matrices[AXIS_NAMES[ax]] = matrix

    # ── Directional bias ──────────────────────────────────────────────────
    def _bias(dicts: list[dict]) -> dict[str, float]:
        total = len(dicts)
        return {
            k: round(sum(d.get(k, False) for d in dicts) / total, 6) if total else 0.0
            for ax in axes for k in AXIS_KEYS[ax]
        }

    directional_bias = {
        "dataset": _bias([r["ground_truth"] for r in parsed]),
        "model":   _bias([r["predicted"]    for r in parsed]),
    }

    # ── Assemble ──────────────────────────────────────────────────────────
    result: dict = {
        "n_total":               n_total,
        "n_parse_success":       n_parsed,
        "parse_error_rate":      round(1 - n_parsed / n_total, 6) if n_total else 0.0,
        "n_structurally_invalid": n_struct_invalid,
        "invalid_rate":          round(n_struct_invalid / n_parsed, 6) if n_parsed else 0.0,
        "n_clean":               n_clean,
        "axis_metrics":          axis_metrics,
        "joint_metrics":         joint_metrics,
        "viewpoint_consistency": compute_group_consistency(records, axes),
        "confusion_matrices":    conf_matrices,
        "directional_bias":      directional_bias,
    }

    # ── Difficulty bins (top-level only to avoid recursion) ───────────────
    if _include_bins:
        bins: dict[str, dict] = {}
        for bname, _, _ in DIFFICULTY_BINS:
            bin_recs = [r for r in records if r.get("difficulty_bin") == bname]
            if bin_recs:
                bins[bname] = compute_metrics(bin_recs, axes, _include_bins=False)
        result["difficulty_bins"] = bins

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a VLM on the multiview-invariance dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True, choices=["chatgpt", "gemini", "llava", "qwen"],
        help="Model to benchmark",
    )
    parser.add_argument(
        "--api_key", default=None,
        help="API key (required for chatgpt and gemini)",
    )
    parser.add_argument(
        "--model_id", default=None,
        help=(
            "Override the default model version "
            "(e.g. gpt-4o-mini, gemini-1.5-pro, llava-hf/llava-v1.6-34b-hf)"
        ),
    )
    parser.add_argument(
        "--n_viewpoints", type=int, default=500,
        help=(
            "Total number of viewpoint queries. Must be even because viewpoints "
            "are selected in pairs (view_0 + view_1 per group). "
            "Odd values are rounded down automatically."
        ),
    )
    parser.add_argument(
        "--axes", type=int, nargs="+", default=[0, 1, 2], choices=[0, 1, 2],
        metavar="AXIS",
        help="Axes to evaluate: 0=lateral (left/right), 1=depth (front/behind), 2=vertical (above/below)",
    )
    parser.add_argument(
        "--dataset_dir", default="dataset",
        help="Path to the dataset index directory produced by build_dataset_index.py",
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Root directory for benchmark outputs",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for group shuffling",
    )
    args = parser.parse_args()

    # Enforce even n_viewpoints
    if args.n_viewpoints % 2 != 0:
        args.n_viewpoints -= 1
        print(f"[warn] --n_viewpoints adjusted to {args.n_viewpoints} (must be even)")

    axes = sorted(set(args.axes))

    # ── Load dataset ──────────────────────────────────────────────────────
    print(f"Loading dataset from {args.dataset_dir!r} …")
    ds = MultiviewDataset(args.dataset_dir)
    ds.shuffle_groups(seed=args.seed)

    n_groups_needed = args.n_viewpoints // 2
    if len(ds.groups) < n_groups_needed:
        print(
            f"[warn] Only {len(ds.groups)} groups available "
            f"(need {n_groups_needed}); using all groups."
        )
        n_groups_needed = len(ds.groups)

    selected_groups     = ds.groups[:n_groups_needed]
    selected_group_ids  = {g["group_id"] for g in selected_groups}
    group_by_id         = {g["group_id"]: g for g in ds.groups}

    # Collect examples for selected groups, sorted by group then viewpoint index
    examples = sorted(
        (ex for ex in ds.examples if ex["group_id"] in selected_group_ids),
        key=lambda e: (e["group_id"], e["viewpoint_index"]),
    )
    print(
        f"Selected {n_groups_needed} groups → {len(examples)} viewpoint queries "
        f"| axes: {[AXIS_NAMES[ax] for ax in axes]}"
    )

    # ── Output directory ──────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{args.model}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps({
        "model":        args.model,
        "model_id":     args.model_id,
        "n_viewpoints": len(examples),
        "n_groups":     n_groups_needed,
        "axes":         axes,
        "axis_names":   [AXIS_NAMES[ax] for ax in axes],
        "dataset_dir":  args.dataset_dir,
        "seed":         args.seed,
        "timestamp":    ts,
    }, indent=2))

    # ── Load model ────────────────────────────────────────────────────────
    print(f"Loading model {args.model!r} …")
    adapter = make_adapter(args.model, args.api_key, args.model_id)

    # ── Run queries ───────────────────────────────────────────────────────
    repo_root  = Path(__file__).parent
    records: list[dict] = []
    pred_path  = out_dir / "predictions.jsonl"

    MAX_RETRIES = 3

    with open(pred_path, "w", encoding="utf-8") as pred_f:
        for ex in tqdm(examples, desc="Querying"):
            group        = group_by_id[ex["group_id"]]
            image_path   = repo_root / ex["image_path"]
            ground_truth = group["reference_object_arrow"]["spatial_relations_from_arrow"]

            raw_response = ""
            predicted    = None
            parse_error  = None

            for attempt in range(MAX_RETRIES):
                prompt = build_prompt(group, axes, attempt=attempt)
                try:
                    raw_response = adapter.query(image_path, prompt)
                except Exception as exc:
                    raw_response = f"[API_ERROR] {exc}"
                    break
                predicted, parse_error = parse_response(raw_response, group, axes)
                if predicted is not None:
                    break
            is_invalid  = predicted is not None and is_structurally_invalid(predicted, axes)
            diff_bin    = get_difficulty_bin(ex["yaw_to_arrow"])

            record: dict = {
                "example_id":      ex["example_id"],
                "group_id":        ex["group_id"],
                "scene_id":        ex["scene_id"],
                "viewpoint_index": ex["viewpoint_index"],
                "yaw_to_arrow":    ex["yaw_to_arrow"],
                "difficulty_bin":  diff_bin,
                "ground_truth":    ground_truth,
                "predicted":       predicted,
                "is_invalid":      is_invalid,
                "parse_error":     parse_error,
                "raw_response":    raw_response,
            }
            records.append(record)
            pred_f.write(json.dumps(record) + "\n")
            pred_f.flush()

    # ── Compute & save metrics ────────────────────────────────────────────
    print("Computing metrics …")
    metrics = compute_metrics(records, axes)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ── Summary ───────────────────────────────────────────────────────────
    sep = "=" * 56
    print(f"\n{sep}")
    print(f"Results saved to : {out_dir}")
    print(f"{sep}")
    print(f"Total queries    : {metrics['n_total']}")
    parse_errs = metrics['n_total'] - metrics['n_parse_success']
    print(f"Parse errors     : {parse_errs}  ({metrics['parse_error_rate']:.1%})")
    print(f"Struct. invalid  : {metrics['n_structurally_invalid']}  ({metrics['invalid_rate']:.1%} of parsed)")
    print(f"Clean queries    : {metrics['n_clean']}")
    print(f"{sep}")
    em = metrics["joint_metrics"]["exact_match_accuracy"]
    print(f"Exact match acc  : {em:.1%}")
    print(f"{'Axis':<12}  {'Accuracy':>10}  {'Macro-F1':>10}")
    print("-" * 36)
    for ax_name, ax_m in metrics["axis_metrics"].items():
        print(f"{ax_name:<12}  {ax_m['accuracy']:>10.1%}  {ax_m['macro_f1']:>10.3f}")
    print(f"{sep}")
    print("Partial correctness:")
    for lbl, info in metrics["joint_metrics"]["partial_correctness"].items():
        print(f"  {lbl:<32}  {info['count']:>5}  ({info['fraction']:.1%})")
    vc = metrics.get("viewpoint_consistency", {})
    if vc.get("n_complete_pairs", 0) > 0:
        print(f"{sep}")
        print(f"Viewpoint consistency  (N={vc['n_complete_pairs']} groups)")
        print(f"  consistent           : {vc['consistent_rate']:.1%}  "
              f"(correct {vc['consistent_correct_rate']:.1%}  |  wrong {vc['consistent_wrong_rate']:.1%})")
        print(f"  inconsistent (flip)  : {vc['inconsistent_rate']:.1%}")
        print(f"  {'Axis':<12}  {'Consistent':>12}  {'Flip rate':>10}")
        print("  " + "-" * 36)
        for ax_name, ax_vc in vc.get("per_axis", {}).items():
            print(f"  {ax_name:<12}  {ax_vc['consistency_rate']:>12.1%}  {ax_vc['flip_rate']:>10.1%}")
    if "difficulty_bins" in metrics:
        print(f"{sep}")
        print(f"{'Difficulty bin':<14}  {'N':>6}  {'Exact':>8}  {'Lat acc':>8}  {'Dep acc':>8}  {'Vert acc':>8}")
        print("-" * 60)
        for bname, bm in metrics["difficulty_bins"].items():
            lat  = bm["axis_metrics"].get("lateral",  {}).get("accuracy", float("nan"))
            dep  = bm["axis_metrics"].get("depth",    {}).get("accuracy", float("nan"))
            vert = bm["axis_metrics"].get("vertical", {}).get("accuracy", float("nan"))
            ex_  = bm["joint_metrics"]["exact_match_accuracy"]
            print(
                f"{bname:<14}  {bm['n_clean']:>6}  {ex_:>8.1%}  "
                f"{lat:>8.1%}  {dep:>8.1%}  {vert:>8.1%}"
            )
    print(sep)


if __name__ == "__main__":
    main()
