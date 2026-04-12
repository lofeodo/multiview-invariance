"""
chatgpt_api.py

Small OpenAI/ChatGPT API client for multimodal prompts.

This module is intentionally narrow in scope for the first integration pass:
it can send a text prompt plus one or more images to the OpenAI Responses API
and return the model's text reply.

Usage examples
--------------
    from chatgpt_api import ChatGPTVisionClient

    client = ChatGPTVisionClient(model="gpt-4.1-mini")
    result = client.prompt_with_images(
        prompt="Describe the spatial relation between the highlighted objects.",
        image_sources=[
            "outputs/scene0000_00/images/objA_3_objB_7_view_0.png",
            "outputs/scene0000_00/images/objA_3_objB_7_view_1.png",
        ],
    )
    print(result.text)

CLI:
    python chatgpt_api.py \
        --prompt "What changed between these views?" \
        --images img1.png img2.png
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_MODEL = "gpt-4.1-mini"
VALID_IMAGE_DETAILS = {"auto", "low", "high"}
URL_PREFIXES = ("http://", "https://", "data:")


@dataclass
class ChatGPTResponse:
    """Convenience wrapper around a multimodal model response."""

    text: str
    model: str
    response_id: str | None
    usage: dict[str, Any] | None
    raw_response: Any


class ChatGPTVisionClient:
    """Thin wrapper over the OpenAI Responses API for text + image prompts."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.model = model
        client_kwargs: dict[str, Any] = {}
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self.client = OpenAI(**client_kwargs)

    def prompt_with_images(
        self,
        prompt: str,
        image_sources: list[str | Path],
        instructions: str | None = None,
        detail: str = "auto",
        max_output_tokens: int | None = None,
    ) -> ChatGPTResponse:
        """Send one prompt plus one or more images and return the text reply.

        Parameters
        ----------
        prompt:
            The user text prompt to send alongside the images.
        image_sources:
            Paths or URLs for images. Local images are base64-encoded into
            data URLs so they can be sent directly in the API request.
        instructions:
            Optional higher-priority instructions to insert into the model
            context, typically loaded from a separate system-prompt file.
        detail:
            Vision detail level: "auto", "low", or "high".
        max_output_tokens:
            Optional cap on generated output tokens.
        """
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not image_sources:
            raise ValueError("image_sources must contain at least one image")
        if detail not in VALID_IMAGE_DETAILS:
            raise ValueError(
                f"detail must be one of {sorted(VALID_IMAGE_DETAILS)}, got {detail!r}"
            )

        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for source in image_sources:
            content.append(self._image_input_item(source, detail))

        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
        }
        if instructions is not None and instructions.strip():
            request["instructions"] = instructions.strip()
        if max_output_tokens is not None:
            request["max_output_tokens"] = max_output_tokens

        response = self.client.responses.create(**request)
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return ChatGPTResponse(
            text=response.output_text,
            model=getattr(response, "model", self.model),
            response_id=getattr(response, "id", None),
            usage=usage,
            raw_response=response,
        )

    def _image_input_item(
        self,
        source: str | Path,
        detail: str,
    ) -> dict[str, str]:
        source_str = str(source)
        if source_str.startswith(URL_PREFIXES):
            image_url = source_str
        else:
            image_url = _local_image_to_data_url(Path(source_str))
        return {
            "type": "input_image",
            "image_url": image_url,
            "detail": detail,
        }


def _local_image_to_data_url(path: Path) -> str:
    """Encode a local image file as a base64 data URL."""
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(
            f"Unsupported image type for {path}. Expected a standard image file."
        )
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a text prompt plus images to the OpenAI Responses API."
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt to send with the images.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more local image paths or remote image URLs.",
    )
    parser.add_argument(
        "--instructions-file",
        default=None,
        help="Optional path to a text file containing higher-priority instructions.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--detail",
        choices=sorted(VALID_IMAGE_DETAILS),
        default="auto",
        help="Image detail level sent to the model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional output token cap.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key override. Otherwise uses OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--print-usage",
        action="store_true",
        help="Print response usage metadata after the answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY or pass --api-key."
        )

    client = ChatGPTVisionClient(model=args.model, api_key=api_key)
    instructions = None
    if args.instructions_file is not None:
        instructions = Path(args.instructions_file).read_text(encoding="utf-8").strip()
    result = client.prompt_with_images(
        prompt=args.prompt,
        image_sources=args.images,
        instructions=instructions,
        detail=args.detail,
        max_output_tokens=args.max_output_tokens,
    )
    print(result.text)
    if args.print_usage and result.usage is not None:
        print("\nUsage:", result.usage)


if __name__ == "__main__":
    main()
