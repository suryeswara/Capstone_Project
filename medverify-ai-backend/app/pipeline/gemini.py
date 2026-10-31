"""Minimal Gemini REST client (no extra SDK dependency). Optional everywhere."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, Sequence

import requests

logger = logging.getLogger(__name__)


class GeminiClient:
    URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str, models: Sequence[str], timeout: float = 20.0):
        self.api_key = api_key
        self.models = list(models)
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key and self.models)

    def generate_json(self, prompt: str, temperature: float = 0.2, system: Optional[str] = None) -> Optional[dict]:
        """Try each configured model in order; return parsed JSON or None."""
        if not self.available:
            return None
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        for model in self.models:
            try:
                r = requests.post(
                    self.URL.format(model=model), json=body, timeout=self.timeout,
                    headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                )
                if r.status_code >= 400:
                    logger.warning("Gemini %s -> HTTP %s", model, r.status_code)
                    continue
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
                return json.loads(text)
            except Exception as exc:  # network, quota, malformed JSON ... -> try next / fall back
                logger.warning("Gemini %s failed: %s", model, exc)
        return None
