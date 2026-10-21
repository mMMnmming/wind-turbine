"""DeepSeek OpenAI-compatible client and strict response parsing."""
from __future__ import annotations
import json, re
import time
from typing import Callable
from dataclasses import dataclass
from .candidate import ModelCandidate

@dataclass(frozen=True)
class DeepSeekSettings:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.2
    timeout_seconds: int = 180
    retries: int = 3
    max_tokens: int = 10000

    def __post_init__(self):
        if not self.base_url.startswith(("https://", "http://")) or not self.model.strip():
            raise ValueError("DeepSeek base_url and model must be configured.")
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string in the private JSON config.")
        if not 0 <= self.temperature <= 2 or self.timeout_seconds < 1 or self.retries < 0 or self.max_tokens < 1:
            raise ValueError("Invalid DeepSeek temperature, timeout, retries, or max_tokens.")

REQUIRED_RESPONSE_FIELDS = {"algorithm_code", "algorithm_idea", "change_hypothesis", "expected_improvement", "change_summary", "direction_id"}


def parse_model_response(content: str, expected_direction: str | None = None) -> ModelCandidate:
    content = content.strip()
    if content.startswith("```"): content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError("DeepSeek response must be one JSON object.") from error
    if not isinstance(data, dict) or set(data) != REQUIRED_RESPONSE_FIELDS:
        raise ValueError("DeepSeek response must contain exactly the required provenance fields.")
    if not all(isinstance(data[key], str) and data[key].strip() for key in REQUIRED_RESPONSE_FIELDS):
        raise ValueError("DeepSeek response fields must be non-empty strings.")
    if expected_direction and data["direction_id"] != expected_direction:
        raise ValueError(f"DeepSeek response direction_id must be {expected_direction!r}.")
    return ModelCandidate(**data)

class DeepSeekClient:
    def __init__(self, settings: DeepSeekSettings):
        if not settings.api_key: raise ValueError("Missing API key in config/deepseek.local.json.")
        try: from openai import OpenAI
        except ImportError as error: raise RuntimeError("Install stage1+llm/requirements.txt before optimize/resume.") from error
        import httpx
        self.observer = None
        self.actual_requests = 0
        self.last_metadata: dict = {}
        http_client = httpx.Client(event_hooks={'request':[self._on_request]})
        self.client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=settings.timeout_seconds,
                             max_retries=settings.retries,http_client=http_client)
        self.settings = settings

    def set_observer(self, observer: Callable) -> None:
        self.observer = observer

    def _on_request(self, _request) -> None:
        self.actual_requests += 1
        if self.observer:
            self.observer(f'Provider HTTP request attempt={self.actual_requests}; configured retries={self.settings.retries}',
                          actual_attempt=self.actual_requests)

    def close(self) -> None:
        self.client.close()

    def generate(self, system_prompt: str, user_prompt: str, expected_direction: str | None = None) -> ModelCandidate:
        self.actual_requests = 0
        start = time.perf_counter()
        self.last_metadata = {}
        try:
            response = self.client.chat.completions.create(model=self.settings.model, temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens, response_format={"type":"json_object"},
                messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}])
            usage = getattr(response,'usage',None)
            self.last_metadata['usage'] = usage.model_dump() if usage is not None else None
            return parse_model_response(response.choices[0].message.content or "", expected_direction)
        finally:
            self.last_metadata.update(actual_requests=self.actual_requests,
                sdk_retries=max(0,self.actual_requests-1),elapsed_seconds=time.perf_counter()-start,
                usage_note='Usage, when present, is from final response only; not an invoice or total retry usage.')
