"""Minimal OpenRouter chat-completions client with vision (image) inputs.

The API key comes from the OPENROUTER_API_KEY environment variable (or is
passed explicitly). When no key is available the client reports
`available == False` and the pipeline falls back to the pose/flow-only
emission terms — nothing VLM-related runs.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

API_URL = "https://openrouter.ai/api/v1/chat/completions"
# cheap vision models per the proposal (Gemini Flash / Qwen2.5-VL)
DEFAULT_MODEL = "google/gemini-2.5-flash"

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class VlmError(RuntimeError):
    pass


def image_content(jpeg_b64: str) -> dict:
    """OpenAI-style image part for a base64 JPEG."""
    return {"type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}}


def text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def parse_json_reply(reply: str) -> dict:
    """Extract the JSON object from a model reply (raw JSON or fenced block)."""
    reply = reply.strip()
    m = _JSON_BLOCK.search(reply)
    if m:
        reply = m.group(1)
    else:
        # tolerate leading/trailing prose around a single top-level object
        i, j = reply.find("{"), reply.rfind("}")
        if i >= 0 and j > i:
            reply = reply[i:j + 1]
    try:
        return json.loads(reply)
    except json.JSONDecodeError as e:
        raise VlmError(f"VLM reply is not valid JSON: {e}\n---\n{reply[:500]}") from e


class OpenRouterClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 timeout: float = 180.0, max_retries: int = 3,
                 temperature: float = 0.0):
        if api_key is None and "OPENROUTER_API_KEY" not in os.environ:
            from src.io.env import load_dotenv
            load_dotenv()  # also honor ./.env when used programmatically
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.n_requests = 0

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, messages: list[dict]) -> str:
        """POST a chat completion; returns the assistant message text."""
        if not self.available:
            raise VlmError("No OpenRouter API key (set OPENROUTER_API_KEY)")
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }).encode()
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                API_URL, data=body, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode())
                self.n_requests += 1
                if "error" in data:
                    raise VlmError(f"OpenRouter error: {data['error']}")
                return data["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                    KeyError, json.JSONDecodeError) as e:
                last_err = e
                # 4xx (except 429) won't heal on retry
                if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503):
                    break
                time.sleep(2.0 * (attempt + 1))
        raise VlmError(f"OpenRouter request failed after {self.max_retries} attempts: {last_err}")

    def chat_json(self, messages: list[dict]) -> dict:
        return parse_json_reply(self.chat(messages))
