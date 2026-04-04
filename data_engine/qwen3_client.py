"""
Drop-in adapter for the Qwen3-VL custom endpoint.

Wraps the custom /predict endpoint to mimic the OpenAI client interface
used by sim2real compose scripts (client.chat.completions.create).

Usage:
    from qwen3_client import Qwen3Client
    client = Qwen3Client("https://52a1-49-50-129-163.ngrok-free.app")
    # Use exactly like OpenAI client in compose scripts
"""

import base64
import re
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


class _Message:
    def __init__(self, content, refusal=None):
        self.content = content
        self.refusal = refusal


class _Choice:
    def __init__(self, content):
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, endpoint_url, timeout=120):
        self.endpoint_url = endpoint_url
        self.timeout = timeout

    def create(self, model=None, messages=None, max_completion_tokens=2048, **kwargs):
        prompt_parts = []
        image_b64 = None

        for msg in (messages or []):
            content = msg.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        prompt_parts.append(item["text"])
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            image_b64 = url.split(",", 1)[1]
                        else:
                            image_b64 = url

        payload = {
            "prompt": "\n".join(prompt_parts),
            "max_new_tokens": max_completion_tokens,
        }
        if image_b64:
            payload["image_base64"] = image_b64

        resp = requests.post(self.endpoint_url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        text = resp.json().get("response", "")
        return _Response(text)


class _Chat:
    def __init__(self, endpoint_url, timeout=120):
        self.completions = _Completions(endpoint_url, timeout)


class Qwen3Client:
    """Drop-in replacement for OpenAI client targeting Qwen3-VL /predict endpoint."""

    def __init__(self, base_url, timeout=120):
        endpoint = f"{base_url.rstrip('/')}/predict"
        self.chat = _Chat(endpoint, timeout)
