"""
Benchmark arrow-frame spatial reasoning on outputs/*/metadata.json.

For each object-pair group, the script makes one model call per regular view
(`view_0`, `view_1`, ...). It never sends `view_arrow` to the model. Instead,
it uses the arrow-frame labels from metadata.json as the benchmark ground truth
and then aggregates the requested evaluation metrics across scenes.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_OUTPUT_DIR = Path("results/chatgpt_arrow_frame_benchmark")
DEFAULT_SYSTEM_PROMPT_FILE = Path("prompts/arrow_frame_system_prompt.txt")
RESPONSES_API_URL = "https://api.openai.com/v1/responses"
VALID_IMAGE_DETAILS = {"auto", "low", "high", "original"}
ALLOWED_LATERAL = ("left", "right", "neither")
ALLOWED_DEPTH = ("in_front", "behind", "neither")
INVALID_LABEL = "__invalid__"


# Current OpenAI model-page pricing, used only for approximate cost reporting.
MODEL_PRICING_USD_PER_1M: dict[str, dict[str, float]] = {
    "gpt-5.4": {"input": 3.0, "cached_input": 0.3, "output": 12.0},
}


class BenchmarkAPIError(RuntimeError):
    """Raised when the OpenAI Responses API returns an error."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def load_dotenv(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file."""
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate arrow-frame spatial reasoning on outputs/*/metadata.json "
            "with an OpenAI frontier GPT model."
        )
    )
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--results-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-md", default=None)
    parser.add_argument(
        "--system-prompt-file",
        default=str(DEFAULT_SYSTEM_PROMPT_FILE),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--detail",
        choices=sorted(VALID_IMAGE_DETAILS),
        default="original",
    )
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="none",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--scene-id", action="append", default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0)
    parser.add_argument("--sleep-between-requests", type=float, default=0.0)
    parser.add_argument("--input-price-per-1m", type=float, default=None)
    parser.add_argument("--cached-input-price-per-1m", type=float, default=None)
    parser.add_argument("--output-price-per-1m", type=float, default=None)
    return parser.parse_args()


def load_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def rel_to_project_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def bool_pair_to_label(
    *,
    left: Any,
    right: Any,
    negative_label: str,
    positive_labels: tuple[str, str],
    axis_name: str,
) -> str:
    left_bool = bool(left)
    right_bool = bool(right)
    if left_bool and right_bool:
        raise ValueError(f"Invalid {axis_name} relation: both directions are true")
    if left_bool:
        return positive_labels[0]
    if right_bool:
        return positive_labels[1]
    return negative_label


def labels_from_spatial_relations(relations: dict[str, Any]) -> dict[str, str]:
    return {
        "lateral": bool_pair_to_label(
            left=relations.get("A_left_of_B"),
            right=relations.get("A_right_of_B"),
            negative_label="neither",
            positive_labels=("left", "right"),
            axis_name="lateral",
        ),
        "depth": bool_pair_to_label(
            left=relations.get("A_in_front_of_B"),
            right=relations.get("A_behind_B"),
            negative_label="neither",
            positive_labels=("in_front", "behind"),
            axis_name="depth",
        ),
    }


def load_benchmark_groups(
    outputs_dir: Path,
    *,
    scene_ids: set[str] | None,
    max_groups: int | None,
    project_root: Path,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    metadata_paths = sorted(outputs_dir.glob("*/metadata.json"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No metadata.json files found under {outputs_dir.resolve()}"
        )

    for metadata_path in metadata_paths:
        scene_dir = metadata_path.parent
        scene_id = scene_dir.name
        if scene_ids is not None and scene_id not in scene_ids:
            continue

        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        for raw_group in meta.get("viewpoint_groups", []):
            arrow = raw_group.get("reference_object_arrow") or {}
            arrow_gt_relations = arrow.get("spatial_relations_from_arrow")
            if not arrow_gt_relations:
                continue

            regular_views = []
            for viewpoint in sorted(
                raw_group.get("viewpoints", []),
                key=lambda item: item["viewpoint_index"],
            ):
                image_path_abs = (scene_dir / viewpoint["image_path"]).resolve()
                regular_views.append(
                    {
                        "viewpoint_index": viewpoint["viewpoint_index"],
                        "image_path_abs": image_path_abs,
                        "image_path": rel_to_project_path(image_path_abs, project_root),
                        "raw_view_labels": labels_from_spatial_relations(
                            viewpoint["spatial_relations"]
                        ),
                        "spatial_relations": viewpoint["spatial_relations"],
                        "angular_sep_from_view0_deg": viewpoint.get(
                            "angular_sep_from_view0_deg"
                        ),
                        "yaw_to_arrow": viewpoint.get("yaw_to_arrow"),
                    }
                )

            if not regular_views:
                continue

            arrow_image_path_abs = None
            arrow_image_path = None
            if arrow.get("image_path"):
                arrow_image_path_abs = (scene_dir / arrow["image_path"]).resolve()
                arrow_image_path = rel_to_project_path(
                    arrow_image_path_abs, project_root
                )

            groups.append(
                {
                    "group_id": f"{scene_id}__{raw_group['pair_id']}",
                    "scene_id": scene_id,
                    "pair_id": raw_group["pair_id"],
                    "object_A": raw_group["object_A"],
                    "object_B": raw_group["object_B"],
                    "reference_object_arrow": {
                        "color": arrow.get("color"),
                        "image_path": arrow_image_path,
                        "spatial_relations_from_arrow": arrow_gt_relations,
                    },
                    "ground_truth_arrow_frame": labels_from_spatial_relations(
                        arrow_gt_relations
                    ),
                    "regular_views": regular_views,
                    "viewpoint_angular_separation_degrees": raw_group.get(
                        "viewpoint_angular_separation_degrees"
                    ),
                }
            )
            if max_groups is not None and len(groups) >= max_groups:
                return groups

    return groups


def normalize_enum_label(value: Any, allowed: tuple[str, ...]) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in allowed else None


def canonical_joint_label(lateral: str, depth: str) -> str:
    return f"{lateral}|{depth}"


def local_image_to_data_url(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def input_image_item(path: Path, detail: str) -> dict[str, Any]:
    return {
        "type": "input_image",
        "image_url": local_image_to_data_url(path),
        "detail": detail,
    }


def build_user_content(
    group: dict[str, Any],
    view: dict[str, Any],
    *,
    detail: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    object_a = group["object_A"]
    object_b = group["object_B"]
    arrow = group["reference_object_arrow"]
    arrow_color = arrow.get("color", "arrow")

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": "\n".join(
                [
                    "You are evaluating spatial relations in a rendered 3D scene.",
                    f"Object A is the {object_a['color']} {object_a['label']}.",
                    f"Object B is the {object_b['color']} {object_b['label']}.",
                    f"The reference arrow is {arrow_color}.",
                    "Answer only about Object A relative to Object B.",
                    "Use the visible arrow in the image as the reference frame.",
                ]
            ),
        }
    ]
    request_image_paths: list[str] = []

    content.append(
        {
            "type": "input_text",
            "text": (
                f"Target image `view_{view['viewpoint_index']}`: answer for this "
                "single regular viewpoint. Do not answer in the current camera's "
                "frame. Instead, imagine you are the arrow, standing at its base "
                "and facing the direction it points."
            ),
        }
    )
    content.append(input_image_item(view["image_path_abs"], detail))
    request_image_paths.append(view["image_path"])

    content.append(
        {
            "type": "input_text",
            "text": (
                f"For `view_{view['viewpoint_index']}`, what is the relation of the "
                f"{object_a['color']} {object_a['label']} (Object A) relative to the "
                f"{object_b['color']} {object_b['label']} (Object B) in the arrow's "
                "reference frame?\n"
                f"- `lateral_relation` must be one of: {', '.join(ALLOWED_LATERAL)}\n"
                f"- `depth_relation` must be one of: {', '.join(ALLOWED_DEPTH)}\n"
                "Use `neither` when the relation along that axis is neither direction.\n"
                "Return only the requested JSON object."
            ),
        }
    )
    return content, request_image_paths


def build_json_schema_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "arrow_frame_relation",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "lateral_relation": {
                    "type": "string",
                    "enum": list(ALLOWED_LATERAL),
                },
                "depth_relation": {
                    "type": "string",
                    "enum": list(ALLOWED_DEPTH),
                },
            },
            "required": ["lateral_relation", "depth_relation"],
        },
    }


def build_json_object_instructions() -> str:
    return (
        "Return exactly one JSON object with this shape:\n"
        '{\n'
        '  "lateral_relation": "left|right|neither",\n'
        '  "depth_relation": "in_front|behind|neither"\n'
        "}\n"
        "Do not include any explanation."
    )


def build_request_payload(
    *,
    model: str,
    system_prompt: str,
    content: list[dict[str, Any]],
    max_output_tokens: int | None,
    reasoning_effort: str,
    use_json_schema: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": (
                build_json_schema_format()
                if use_json_schema
                else {"type": "json_object"}
            )
        },
    }
    if max_output_tokens is not None:
        payload["max_output_tokens"] = max_output_tokens
    if reasoning_effort != "none":
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def should_retry_api_error(status_code: int | None) -> bool:
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise BenchmarkAPIError(
            f"HTTP {exc.code}: {body}",
            status_code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise BenchmarkAPIError(str(exc), status_code=None) from exc


def request_model_response(
    *,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> dict[str, Any]:
    for attempt in range(max_retries + 1):
        try:
            return post_json(
                url=RESPONSES_API_URL,
                payload=payload,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
            )
        except BenchmarkAPIError as exc:
            is_last_attempt = attempt >= max_retries
            if is_last_attempt or not should_retry_api_error(exc.status_code):
                raise
            sleep_seconds = retry_backoff_seconds * (2**attempt)
            print(
                f"  transient API error ({exc}); retrying in {sleep_seconds:.1f}s..."
            )
            time.sleep(sleep_seconds)
    raise AssertionError("unreachable")


def extract_response_text(response_json: dict[str, Any]) -> str:
    output_text = response_json.get("output_text")
    if isinstance(output_text, str):
        return output_text.strip()

    texts: list[str] = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(
                content.get("text"), str
            ):
                texts.append(content["text"])
            if content.get("type") == "refusal" and isinstance(
                content.get("refusal"), str
            ):
                texts.append(content["refusal"])
    return "\n".join(texts).strip()


def parse_prediction(response_text: str) -> tuple[dict[str, str] | None, str | None]:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}"

    if not isinstance(payload, dict):
        return None, "Response JSON is not an object"

    lateral = normalize_enum_label(payload.get("lateral_relation"), ALLOWED_LATERAL)
    depth = normalize_enum_label(payload.get("depth_relation"), ALLOWED_DEPTH)
    if lateral is None or depth is None:
        return None, "Missing or invalid lateral_relation/depth_relation labels"

    return {
        "lateral_relation": lateral,
        "depth_relation": depth,
    }, None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def usage_counts(usage: dict[str, Any] | None) -> dict[str, int | None]:
    usage = usage or {}
    input_details = usage.get("input_tokens_details") or {}
    return {
        "input_tokens": safe_int(usage.get("input_tokens")),
        "output_tokens": safe_int(usage.get("output_tokens")),
        "total_tokens": safe_int(usage.get("total_tokens")),
        "cached_input_tokens": (
            safe_int(input_details.get("cached_tokens"))
            or safe_int(usage.get("cached_input_tokens"))
            or 0
        ),
    }


def canonicalize_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    lowered = model_name.lower()
    for prefix in sorted(MODEL_PRICING_USD_PER_1M, key=len, reverse=True):
        if lowered == prefix or lowered.startswith(prefix + "-"):
            return prefix
    return lowered


def estimate_cost_usd(
    *,
    model_name: str | None,
    usage: dict[str, Any] | None,
    cli_prices: dict[str, float | None],
) -> float | None:
    counts = usage_counts(usage)
    input_tokens = counts["input_tokens"]
    output_tokens = counts["output_tokens"]
    cached_input_tokens = counts["cached_input_tokens"] or 0
    if input_tokens is None or output_tokens is None:
        return None

    input_price = cli_prices["input"]
    cached_input_price = cli_prices["cached_input"]
    output_price = cli_prices["output"]

    if input_price is None or cached_input_price is None or output_price is None:
        pricing_key = canonicalize_model_name(model_name)
        pricing = MODEL_PRICING_USD_PER_1M.get(pricing_key or "")
        if pricing is None:
            return None
        input_price = pricing["input"] if input_price is None else input_price
        cached_input_price = (
            pricing["cached_input"]
            if cached_input_price is None
            else cached_input_price
        )
        output_price = pricing["output"] if output_price is None else output_price

    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached_input_tokens * input_price
        + cached_input_tokens * cached_input_price
        + output_tokens * output_price
    ) / 1_000_000.0


def make_error_view_result(
    *,
    view: dict[str, Any],
    request_image_paths: list[str],
    arrow_view_image_available: bool,
    request_format: str,
    latency_seconds: float,
    error_type: str,
    error_message: str,
) -> dict[str, Any]:
    return {
        "viewpoint_index": view["viewpoint_index"],
        "image_path": view["image_path"],
        "request_image_paths": request_image_paths,
        "arrow_view_image_available": arrow_view_image_available,
        "raw_view_labels": view["raw_view_labels"],
        "spatial_relations": view["spatial_relations"],
        "angular_sep_from_view0_deg": view.get("angular_sep_from_view0_deg"),
        "yaw_to_arrow": view.get("yaw_to_arrow"),
        "request_format": request_format,
        "latency_seconds": latency_seconds,
        "response_id": None,
        "response_status": error_type,
        "response_model": None,
        "usage": None,
        "usage_counts": usage_counts(None),
        "estimated_cost_usd": None,
        "response_text": "",
        "parsed_prediction": None,
        "parse_error": error_message,
        "valid_prediction": False,
        "correct": {"lateral": False, "depth": False, "joint": False},
    }


def call_model_for_view(
    *,
    group: dict[str, Any],
    view: dict[str, Any],
    system_prompt: str,
    api_key: str,
    model: str,
    detail: str,
    max_output_tokens: int | None,
    reasoning_effort: str,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    cli_prices: dict[str, float | None],
) -> dict[str, Any]:
    content, request_image_paths = build_user_content(
        group,
        view,
        detail=detail,
    )
    request_payload = build_request_payload(
        model=model,
        system_prompt=system_prompt,
        content=content,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        use_json_schema=True,
    )

    start = time.perf_counter()
    request_format = "json_schema"
    try:
        response_json = request_model_response(
            api_key=api_key,
            payload=request_payload,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    except BenchmarkAPIError as exc:
        message = str(exc)
        if exc.status_code == 400:
            fallback_content = list(content)
            fallback_content.append(
                {"type": "input_text", "text": build_json_object_instructions()}
            )
            fallback_payload = build_request_payload(
                model=model,
                system_prompt=system_prompt,
                content=fallback_content,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                use_json_schema=False,
            )
            try:
                response_json = request_model_response(
                    api_key=api_key,
                    payload=fallback_payload,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                request_format = "json_object_fallback"
            except BenchmarkAPIError:
                return make_error_view_result(
                    view=view,
                    request_image_paths=request_image_paths,
                    arrow_view_image_available=bool(
                        group["reference_object_arrow"].get("image_path")
                    ),
                    request_format=request_format,
                    latency_seconds=time.perf_counter() - start,
                    error_type="request_error",
                    error_message=message,
                )
        else:
            return make_error_view_result(
                view=view,
                request_image_paths=request_image_paths,
                arrow_view_image_available=bool(
                    group["reference_object_arrow"].get("image_path")
                ),
                request_format=request_format,
                latency_seconds=time.perf_counter() - start,
                error_type="request_error",
                error_message=message,
            )

    latency_seconds = time.perf_counter() - start
    response_text = extract_response_text(response_json)
    parsed_prediction, parse_error = parse_prediction(response_text)
    usage = response_json.get("usage")
    response_model = response_json.get("model", model)
    estimated_cost = estimate_cost_usd(
        model_name=response_model,
        usage=usage,
        cli_prices=cli_prices,
    )

    ground_truth = group["ground_truth_arrow_frame"]
    correct_lateral = (
        parsed_prediction is not None
        and parsed_prediction["lateral_relation"] == ground_truth["lateral"]
    )
    correct_depth = (
        parsed_prediction is not None
        and parsed_prediction["depth_relation"] == ground_truth["depth"]
    )

    return {
        "viewpoint_index": view["viewpoint_index"],
        "image_path": view["image_path"],
        "request_image_paths": request_image_paths,
        "arrow_view_image_available": bool(
            group["reference_object_arrow"].get("image_path")
        ),
        "raw_view_labels": view["raw_view_labels"],
        "spatial_relations": view["spatial_relations"],
        "angular_sep_from_view0_deg": view.get("angular_sep_from_view0_deg"),
        "yaw_to_arrow": view.get("yaw_to_arrow"),
        "request_format": request_format,
        "latency_seconds": latency_seconds,
        "response_id": response_json.get("id"),
        "response_status": response_json.get("status"),
        "response_model": response_model,
        "usage": usage,
        "usage_counts": usage_counts(usage),
        "estimated_cost_usd": estimated_cost,
        "response_text": response_text,
        "parsed_prediction": parsed_prediction,
        "parse_error": parse_error,
        "valid_prediction": parsed_prediction is not None,
        "correct": {
            "lateral": bool(correct_lateral),
            "depth": bool(correct_depth),
            "joint": bool(correct_lateral and correct_depth),
        },
    }


def finalize_group_record(group_record: dict[str, Any]) -> dict[str, Any]:
    predicted_pairs = []
    valid_predictions = []
    raw_lateral = []
    raw_depth = []
    raw_joint = []
    total_latency = 0.0
    total_estimated_cost = 0.0
    cost_available_for_all_views = True
    total_usage = Counter()

    for view in group_record["views"]:
        total_latency += float(view.get("latency_seconds") or 0.0)

        estimated_cost = view.get("estimated_cost_usd")
        if estimated_cost is None:
            cost_available_for_all_views = False
        else:
            total_estimated_cost += float(estimated_cost)

        usage = view.get("usage_counts") or {}
        for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens"):
            if usage.get(key) is not None:
                total_usage[key] += int(usage[key])

        raw_view = view["raw_view_labels"]
        raw_lateral.append(raw_view["lateral"])
        raw_depth.append(raw_view["depth"])
        raw_joint.append(canonical_joint_label(raw_view["lateral"], raw_view["depth"]))

        if view["valid_prediction"]:
            prediction = view["parsed_prediction"]
            predicted_pairs.append(
                canonical_joint_label(
                    prediction["lateral_relation"], prediction["depth_relation"]
                )
            )
            valid_predictions.append(prediction)

    complete_valid = len(valid_predictions) == len(group_record["views"])
    consistent = complete_valid and len(set(predicted_pairs)) == 1
    any_joint_correct = any(view["correct"]["joint"] for view in group_record["views"])
    all_joint_correct = (
        complete_valid and all(view["correct"]["joint"] for view in group_record["views"])
    )

    group_record["group_metrics"] = {
        "n_regular_views": len(group_record["views"]),
        "all_views_valid": complete_valid,
        "consistent_predictions": consistent,
        "any_joint_correct": any_joint_correct,
        "all_views_joint_correct": all_joint_correct,
        "consistency_when_correct": bool(
            complete_valid and any_joint_correct and consistent and all_joint_correct
        ),
        "raw_relation_flip": {
            "lateral": len(set(raw_lateral)) > 1,
            "depth": len(set(raw_depth)) > 1,
            "joint": len(set(raw_joint)) > 1,
        },
        "predicted_relation_consistent": {
            "lateral": bool(
                complete_valid
                and len({pred["lateral_relation"] for pred in valid_predictions}) == 1
            ),
            "depth": bool(
                complete_valid
                and len({pred["depth_relation"] for pred in valid_predictions}) == 1
            ),
            "joint": consistent,
        },
        "latency_seconds": total_latency,
        "estimated_cost_usd": (
            total_estimated_cost if cost_available_for_all_views else None
        ),
        "usage_counts": {
            "input_tokens": total_usage.get("input_tokens"),
            "output_tokens": total_usage.get("output_tokens"),
            "total_tokens": total_usage.get("total_tokens"),
            "cached_input_tokens": total_usage.get("cached_input_tokens"),
        },
    }
    return group_record


def load_existing_results(results_jsonl: Path) -> list[dict[str, Any]]:
    if not results_jsonl.exists():
        return []
    records = []
    with results_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def dedupe_records_by_group(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        group_id = record.get("group_id")
        if isinstance(group_id, str):
            latest[group_id] = record
    return [latest[group_id] for group_id in sorted(latest)]


def rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def average(total: float, count: int) -> float | None:
    return None if count == 0 else total / count


def make_bucket_stats() -> dict[str, Any]:
    return {
        "samples": 0,
        "valid_predictions": 0,
        "lateral_correct": 0,
        "depth_correct": 0,
        "joint_correct": 0,
        "latency_seconds_total": 0.0,
        "estimated_cost_usd_total": 0.0,
        "cost_count": 0,
    }


def add_sample_to_bucket(bucket: dict[str, Any], view: dict[str, Any]) -> None:
    bucket["samples"] += 1
    bucket["latency_seconds_total"] += float(view.get("latency_seconds") or 0.0)
    if view["valid_prediction"]:
        bucket["valid_predictions"] += 1
    if view["correct"]["lateral"]:
        bucket["lateral_correct"] += 1
    if view["correct"]["depth"]:
        bucket["depth_correct"] += 1
    if view["correct"]["joint"]:
        bucket["joint_correct"] += 1
    if view.get("estimated_cost_usd") is not None:
        bucket["estimated_cost_usd_total"] += float(view["estimated_cost_usd"])
        bucket["cost_count"] += 1


def finalize_bucket_stats(bucket: dict[str, Any]) -> dict[str, Any]:
    samples = bucket["samples"]
    valid_predictions = bucket["valid_predictions"]
    return {
        "samples": samples,
        "valid_predictions": valid_predictions,
        "valid_prediction_rate": rate(valid_predictions, samples),
        "lateral_accuracy": rate(bucket["lateral_correct"], samples),
        "depth_accuracy": rate(bucket["depth_correct"], samples),
        "joint_exact_match_accuracy": rate(bucket["joint_correct"], samples),
        "lateral_accuracy_given_valid": rate(bucket["lateral_correct"], valid_predictions),
        "depth_accuracy_given_valid": rate(bucket["depth_correct"], valid_predictions),
        "joint_accuracy_given_valid": rate(bucket["joint_correct"], valid_predictions),
        "avg_latency_seconds": average(bucket["latency_seconds_total"], samples),
        "avg_estimated_cost_usd": average(
            bucket["estimated_cost_usd_total"], bucket["cost_count"]
        ),
    }


def viewpoint_separation_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    degrees = float(value)
    if degrees < 30:
        return "<30"
    if degrees < 60:
        return "30-60"
    if degrees < 90:
        return "60-90"
    return ">=90"


def abs_yaw_bucket(value: Any) -> str:
    if value is None:
        return "unknown"
    degrees = abs(float(value))
    if degrees < 15:
        return "<15"
    if degrees < 30:
        return "15-30"
    if degrees < 60:
        return "30-60"
    if degrees < 90:
        return "60-90"
    return ">=90"


def object_pair_bucket(group_record: dict[str, Any]) -> str:
    return f"{group_record['object_A']['label']} -> {group_record['object_B']['label']}"


def arrow_view_availability_bucket(view: dict[str, Any]) -> str:
    return "present" if view["arrow_view_image_available"] else "absent"


def init_confusion_matrix(truth_labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    predicted_labels = list(truth_labels) + [INVALID_LABEL]
    return {
        truth: {pred: 0 for pred in predicted_labels}
        for truth in truth_labels
    }


def summarize_results(
    *,
    benchmark_groups: list[dict[str, Any]],
    group_records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    records = dedupe_records_by_group(group_records)
    record_by_group = {record["group_id"]: record for record in records}

    total_dataset_groups = len(benchmark_groups)
    total_dataset_samples = sum(len(group["regular_views"]) for group in benchmark_groups)

    lateral_matrix = init_confusion_matrix(ALLOWED_LATERAL)
    depth_matrix = init_confusion_matrix(ALLOWED_DEPTH)

    by_separation: dict[str, dict[str, Any]] = defaultdict(make_bucket_stats)
    by_abs_yaw: dict[str, dict[str, Any]] = defaultdict(make_bucket_stats)
    by_object_pair: dict[str, dict[str, Any]] = defaultdict(make_bucket_stats)
    by_arrow_view_availability: dict[str, dict[str, Any]] = defaultdict(make_bucket_stats)

    sample_totals = {
        "samples": 0,
        "valid_predictions": 0,
        "lateral_correct": 0,
        "depth_correct": 0,
        "joint_correct": 0,
        "latency_seconds_total": 0.0,
        "estimated_cost_usd_total": 0.0,
        "cost_count": 0,
        "input_tokens_total": 0,
        "output_tokens_total": 0,
        "total_tokens_total": 0,
        "token_count_samples": 0,
    }
    group_totals = {
        "processed_groups": len(records),
        "groups_all_views_valid": 0,
        "consistent_groups": 0,
        "consistency_when_correct_numerator": 0,
        "consistency_when_correct_denominator": 0,
        "flip_robust_lateral_numerator": 0,
        "flip_robust_lateral_denominator": 0,
        "flip_robust_depth_numerator": 0,
        "flip_robust_depth_denominator": 0,
        "flip_robust_joint_numerator": 0,
        "flip_robust_joint_denominator": 0,
        "latency_seconds_total": 0.0,
        "estimated_cost_usd_total": 0.0,
        "cost_count": 0,
        "input_tokens_total": 0,
        "output_tokens_total": 0,
        "total_tokens_total": 0,
        "token_count_groups": 0,
    }

    for benchmark_group in benchmark_groups:
        record = record_by_group.get(benchmark_group["group_id"])
        if record is None:
            continue

        group_metrics = record.get("group_metrics") or {}
        group_totals["latency_seconds_total"] += float(
            group_metrics.get("latency_seconds") or 0.0
        )
        if group_metrics.get("estimated_cost_usd") is not None:
            group_totals["estimated_cost_usd_total"] += float(
                group_metrics["estimated_cost_usd"]
            )
            group_totals["cost_count"] += 1

        group_usage = group_metrics.get("usage_counts") or {}
        if group_usage.get("total_tokens") is not None:
            group_totals["token_count_groups"] += 1
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = group_usage.get(key)
                if value is not None:
                    group_totals[f"{key}_total"] += int(value)

        if group_metrics.get("all_views_valid"):
            group_totals["groups_all_views_valid"] += 1
            if group_metrics.get("consistent_predictions"):
                group_totals["consistent_groups"] += 1
            if group_metrics.get("any_joint_correct"):
                group_totals["consistency_when_correct_denominator"] += 1
                if group_metrics.get("consistency_when_correct"):
                    group_totals["consistency_when_correct_numerator"] += 1
            if group_metrics.get("raw_relation_flip", {}).get("lateral"):
                group_totals["flip_robust_lateral_denominator"] += 1
                if group_metrics.get("predicted_relation_consistent", {}).get("lateral"):
                    group_totals["flip_robust_lateral_numerator"] += 1
            if group_metrics.get("raw_relation_flip", {}).get("depth"):
                group_totals["flip_robust_depth_denominator"] += 1
                if group_metrics.get("predicted_relation_consistent", {}).get("depth"):
                    group_totals["flip_robust_depth_numerator"] += 1
            if group_metrics.get("raw_relation_flip", {}).get("joint"):
                group_totals["flip_robust_joint_denominator"] += 1
                if group_metrics.get("predicted_relation_consistent", {}).get("joint"):
                    group_totals["flip_robust_joint_numerator"] += 1

        for view in record.get("views", []):
            sample_totals["samples"] += 1
            sample_totals["latency_seconds_total"] += float(
                view.get("latency_seconds") or 0.0
            )
            if view.get("estimated_cost_usd") is not None:
                sample_totals["estimated_cost_usd_total"] += float(
                    view["estimated_cost_usd"]
                )
                sample_totals["cost_count"] += 1

            usage = view.get("usage_counts") or {}
            if usage.get("total_tokens") is not None:
                sample_totals["token_count_samples"] += 1
                for key in ("input_tokens", "output_tokens", "total_tokens"):
                    value = usage.get(key)
                    if value is not None:
                        sample_totals[f"{key}_total"] += int(value)

            if view["valid_prediction"]:
                sample_totals["valid_predictions"] += 1
            if view["correct"]["lateral"]:
                sample_totals["lateral_correct"] += 1
            if view["correct"]["depth"]:
                sample_totals["depth_correct"] += 1
            if view["correct"]["joint"]:
                sample_totals["joint_correct"] += 1

            truth = record["ground_truth_arrow_frame"]
            predicted_lateral = INVALID_LABEL
            predicted_depth = INVALID_LABEL
            if view["valid_prediction"]:
                predicted_lateral = view["parsed_prediction"]["lateral_relation"]
                predicted_depth = view["parsed_prediction"]["depth_relation"]
            lateral_matrix[truth["lateral"]][predicted_lateral] += 1
            depth_matrix[truth["depth"]][predicted_depth] += 1

            add_sample_to_bucket(
                by_separation[
                    viewpoint_separation_bucket(
                        record.get("viewpoint_angular_separation_degrees")
                    )
                ],
                view,
            )
            add_sample_to_bucket(by_abs_yaw[abs_yaw_bucket(view.get("yaw_to_arrow"))], view)
            add_sample_to_bucket(by_object_pair[object_pair_bucket(record)], view)
            add_sample_to_bucket(
                by_arrow_view_availability[arrow_view_availability_bucket(view)],
                view,
            )

    sample_valid = sample_totals["valid_predictions"]
    processed_groups = group_totals["processed_groups"]

    return {
        "config": config,
        "dataset": {
            "eligible_groups": total_dataset_groups,
            "eligible_regular_view_samples": total_dataset_samples,
            "processed_groups": processed_groups,
            "processed_regular_view_samples": sample_totals["samples"],
        },
        "parse_and_valid_response_rate": {
            "sample_valid_predictions": sample_valid,
            "sample_valid_prediction_rate": rate(sample_valid, sample_totals["samples"]),
            "groups_all_views_valid": group_totals["groups_all_views_valid"],
            "group_all_views_valid_rate": rate(
                group_totals["groups_all_views_valid"], processed_groups
            ),
        },
        "accuracy": {
            "lateral_accuracy": rate(
                sample_totals["lateral_correct"], sample_totals["samples"]
            ),
            "depth_accuracy": rate(
                sample_totals["depth_correct"], sample_totals["samples"]
            ),
            "joint_exact_match_accuracy": rate(
                sample_totals["joint_correct"], sample_totals["samples"]
            ),
            "lateral_accuracy_given_valid": rate(
                sample_totals["lateral_correct"], sample_valid
            ),
            "depth_accuracy_given_valid": rate(
                sample_totals["depth_correct"], sample_valid
            ),
            "joint_accuracy_given_valid": rate(
                sample_totals["joint_correct"], sample_valid
            ),
        },
        "group_metrics": {
            "consistency": {
                "numerator": group_totals["consistent_groups"],
                "denominator": group_totals["groups_all_views_valid"],
                "rate": rate(
                    group_totals["consistent_groups"],
                    group_totals["groups_all_views_valid"],
                ),
            },
            "consistency_when_correct": {
                "numerator": group_totals["consistency_when_correct_numerator"],
                "denominator": group_totals["consistency_when_correct_denominator"],
                "rate": rate(
                    group_totals["consistency_when_correct_numerator"],
                    group_totals["consistency_when_correct_denominator"],
                ),
            },
            "flip_robustness": {
                "lateral": {
                    "numerator": group_totals["flip_robust_lateral_numerator"],
                    "denominator": group_totals["flip_robust_lateral_denominator"],
                    "rate": rate(
                        group_totals["flip_robust_lateral_numerator"],
                        group_totals["flip_robust_lateral_denominator"],
                    ),
                },
                "depth": {
                    "numerator": group_totals["flip_robust_depth_numerator"],
                    "denominator": group_totals["flip_robust_depth_denominator"],
                    "rate": rate(
                        group_totals["flip_robust_depth_numerator"],
                        group_totals["flip_robust_depth_denominator"],
                    ),
                },
                "joint": {
                    "numerator": group_totals["flip_robust_joint_numerator"],
                    "denominator": group_totals["flip_robust_joint_denominator"],
                    "rate": rate(
                        group_totals["flip_robust_joint_numerator"],
                        group_totals["flip_robust_joint_denominator"],
                    ),
                },
            },
        },
        "confusion_matrices": {"lateral": lateral_matrix, "depth": depth_matrix},
        "accuracy_by_difficulty_bucket": {
            "viewpoint_angular_separation_degrees": {
                key: finalize_bucket_stats(value)
                for key, value in sorted(by_separation.items())
            },
            "abs_yaw_to_arrow": {
                key: finalize_bucket_stats(value)
                for key, value in sorted(by_abs_yaw.items())
            },
            "object_pair_type": {
                key: finalize_bucket_stats(value)
                for key, value in sorted(by_object_pair.items())
            },
            "arrow_view_image_available": {
                key: finalize_bucket_stats(value)
                for key, value in sorted(by_arrow_view_availability.items())
            },
        },
        "cost_and_latency": {
            "avg_latency_seconds_per_sample": average(
                sample_totals["latency_seconds_total"], sample_totals["samples"]
            ),
            "avg_latency_seconds_per_group": average(
                group_totals["latency_seconds_total"], processed_groups
            ),
            "avg_estimated_cost_usd_per_sample": average(
                sample_totals["estimated_cost_usd_total"], sample_totals["cost_count"]
            ),
            "avg_estimated_cost_usd_per_group": average(
                group_totals["estimated_cost_usd_total"], group_totals["cost_count"]
            ),
            "total_estimated_cost_usd": (
                group_totals["estimated_cost_usd_total"]
                if group_totals["cost_count"] > 0
                else None
            ),
            "avg_input_tokens_per_sample": average(
                sample_totals["input_tokens_total"],
                sample_totals["token_count_samples"],
            ),
            "avg_output_tokens_per_sample": average(
                sample_totals["output_tokens_total"],
                sample_totals["token_count_samples"],
            ),
            "avg_total_tokens_per_sample": average(
                sample_totals["total_tokens_total"],
                sample_totals["token_count_samples"],
            ),
            "avg_input_tokens_per_group": average(
                group_totals["input_tokens_total"],
                group_totals["token_count_groups"],
            ),
            "avg_output_tokens_per_group": average(
                group_totals["output_tokens_total"],
                group_totals["token_count_groups"],
            ),
            "avg_total_tokens_per_group": average(
                group_totals["total_tokens_total"],
                group_totals["token_count_groups"],
            ),
        },
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.2f}%"


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}s"


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Arrow-Frame Benchmark Summary", ""]
    dataset = summary["dataset"]
    parse_stats = summary["parse_and_valid_response_rate"]
    accuracy = summary["accuracy"]
    group_metrics = summary["group_metrics"]
    cost = summary["cost_and_latency"]

    lines.extend(
        [
            "## Coverage",
            f"- Eligible groups: {dataset['eligible_groups']}",
            f"- Eligible regular-view samples: {dataset['eligible_regular_view_samples']}",
            f"- Processed groups: {dataset['processed_groups']}",
            f"- Processed regular-view samples: {dataset['processed_regular_view_samples']}",
            "",
            "## Response Quality",
            (
                "- Sample valid-response rate: "
                f"{format_metric(parse_stats['sample_valid_prediction_rate'])} "
                f"({parse_stats['sample_valid_predictions']}/"
                f"{dataset['processed_regular_view_samples']})"
            ),
            (
                "- Group all-views-valid rate: "
                f"{format_metric(parse_stats['group_all_views_valid_rate'])} "
                f"({parse_stats['groups_all_views_valid']}/"
                f"{dataset['processed_groups']})"
            ),
            "",
            "## Accuracy",
            f"- Lateral accuracy: {format_metric(accuracy['lateral_accuracy'])}",
            f"- Depth accuracy: {format_metric(accuracy['depth_accuracy'])}",
            (
                "- Joint exact-match accuracy: "
                f"{format_metric(accuracy['joint_exact_match_accuracy'])}"
            ),
            "",
            "## Group Metrics",
            (
                "- Consistency: "
                f"{format_metric(group_metrics['consistency']['rate'])} "
                f"({group_metrics['consistency']['numerator']}/"
                f"{group_metrics['consistency']['denominator']})"
            ),
            (
                "- Consistency-when-correct: "
                f"{format_metric(group_metrics['consistency_when_correct']['rate'])} "
                f"({group_metrics['consistency_when_correct']['numerator']}/"
                f"{group_metrics['consistency_when_correct']['denominator']})"
            ),
            (
                "- Flip robustness (lateral/depth/joint): "
                f"{format_metric(group_metrics['flip_robustness']['lateral']['rate'])} / "
                f"{format_metric(group_metrics['flip_robustness']['depth']['rate'])} / "
                f"{format_metric(group_metrics['flip_robustness']['joint']['rate'])}"
            ),
            "",
            "## Cost And Latency",
            (
                "- Avg latency per sample/group: "
                f"{format_seconds(cost['avg_latency_seconds_per_sample'])} / "
                f"{format_seconds(cost['avg_latency_seconds_per_group'])}"
            ),
            (
                "- Avg estimated cost per sample/group: "
                f"${format_float(cost['avg_estimated_cost_usd_per_sample'], 4)} / "
                f"${format_float(cost['avg_estimated_cost_usd_per_group'], 4)}"
            ),
            (
                "- Avg tokens per sample (input/output/total): "
                f"{format_float(cost['avg_input_tokens_per_sample'], 1)} / "
                f"{format_float(cost['avg_output_tokens_per_sample'], 1)} / "
                f"{format_float(cost['avg_total_tokens_per_sample'], 1)}"
            ),
            (
                "- Avg tokens per group (input/output/total): "
                f"{format_float(cost['avg_input_tokens_per_group'], 1)} / "
                f"{format_float(cost['avg_output_tokens_per_group'], 1)} / "
                f"{format_float(cost['avg_total_tokens_per_group'], 1)}"
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def make_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "detail": args.detail,
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "system_prompt_file": str(Path(args.system_prompt_file)),
        "send_arrow_view_image_to_model": False,
        "outputs_dir": str(Path(args.outputs_dir)),
        "scene_ids": args.scene_id,
        "max_groups": args.max_groups,
    }


def main() -> None:
    load_dotenv(Path(".env"))
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    output_dir = Path(args.output_dir)
    results_jsonl = (
        Path(args.results_jsonl) if args.results_jsonl else output_dir / "group_results.jsonl"
    )
    summary_json = Path(args.summary_json) if args.summary_json else output_dir / "summary.json"
    summary_md = Path(args.summary_md) if args.summary_md else output_dir / "summary.md"
    system_prompt_file = Path(args.system_prompt_file)
    run_config = make_run_config(args)

    benchmark_groups = load_benchmark_groups(
        outputs_dir,
        scene_ids=set(args.scene_id) if args.scene_id else None,
        max_groups=args.max_groups,
        project_root=Path.cwd(),
    )
    if not benchmark_groups:
        raise SystemExit("No eligible groups found in the selected outputs directory.")

    print(
        f"Loaded {len(benchmark_groups)} group(s) from {outputs_dir} covering "
        f"{sum(len(g['regular_views']) for g in benchmark_groups)} regular views."
    )

    if args.overwrite and results_jsonl.exists():
        results_jsonl.unlink()

    if args.summarize_only:
        summary = summarize_results(
            benchmark_groups=benchmark_groups,
            group_records=load_existing_results(results_jsonl),
            config=run_config,
        )
        write_json(summary_json, summary)
        write_markdown(summary_md, render_summary_markdown(summary))
        print(f"Summary written to {summary_json}")
        print(f"Markdown summary written to {summary_md}")
        return

    system_prompt = load_text_file(system_prompt_file)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    existing_records = load_existing_results(results_jsonl) if results_jsonl.exists() else []
    processed_group_ids = {
        record["group_id"]
        for record in dedupe_records_by_group(existing_records)
        if isinstance(record.get("group_id"), str)
    }
    cli_prices = {
        "input": args.input_price_per_1m,
        "cached_input": args.cached_input_price_per_1m,
        "output": args.output_price_per_1m,
    }

    new_records: list[dict[str, Any]] = []
    for index, group in enumerate(benchmark_groups, start=1):
        if group["group_id"] in processed_group_ids:
            print(f"[{index}/{len(benchmark_groups)}] Skipping {group['group_id']} (already processed)")
            continue

        print(
            f"[{index}/{len(benchmark_groups)}] Processing {group['group_id']} "
            f"with {len(group['regular_views'])} regular views..."
        )

        record = {
            "group_id": group["group_id"],
            "scene_id": group["scene_id"],
            "pair_id": group["pair_id"],
            "object_A": group["object_A"],
            "object_B": group["object_B"],
            "reference_object_arrow": group["reference_object_arrow"],
            "ground_truth_arrow_frame": group["ground_truth_arrow_frame"],
            "viewpoint_angular_separation_degrees": group[
                "viewpoint_angular_separation_degrees"
            ],
            "run_config": {
                **run_config,
                "arrow_view_image_sent_to_model_for_this_group": False,
            },
            "views": [],
        }

        for view in group["regular_views"]:
            record["views"].append(
                call_model_for_view(
                    group=group,
                    view=view,
                    system_prompt=system_prompt,
                    api_key=api_key,
                    model=args.model,
                    detail=args.detail,
                    max_output_tokens=args.max_output_tokens,
                    reasoning_effort=args.reasoning_effort,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                    retry_backoff_seconds=args.retry_backoff_seconds,
                    cli_prices=cli_prices,
                )
            )
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)

        finalized_record = finalize_group_record(record)
        write_jsonl_record(results_jsonl, finalized_record)
        new_records.append(finalized_record)

    summary = summarize_results(
        benchmark_groups=benchmark_groups,
        group_records=existing_records + new_records,
        config=run_config,
    )
    write_json(summary_json, summary)
    write_markdown(summary_md, render_summary_markdown(summary))

    print(f"Raw group results: {results_jsonl}")
    print(f"Summary JSON:      {summary_json}")
    print(f"Summary Markdown:  {summary_md}")


if __name__ == "__main__":
    main()
