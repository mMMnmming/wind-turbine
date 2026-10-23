"""DeepSeek OpenAI-compatible client and strict response parsing."""

from __future__ import annotations
import json, re
import time
from typing import Callable
from dataclasses import dataclass
from .candidate import ModelCandidate
from .artifacts import redact_secrets


@dataclass(frozen=True)
class DeepSeekSettings:
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.2
    timeout_seconds: int = 180
    retries: int = 1
    max_tokens: int = 8192
    thinking: str = "disabled"
    reasoning_effort: str | None = None
    stream: bool = True

    def __post_init__(self):
        if (
            not self.base_url.startswith(("https://", "http://"))
            or not self.model.strip()
        ):
            raise ValueError("DeepSeek base_url and model must be configured.")
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string in the private JSON config.")
        if (
            not 0 <= self.temperature <= 2
            or self.timeout_seconds < 1
            or self.retries < 0
            or self.max_tokens < 1
        ):
            raise ValueError(
                "Invalid DeepSeek temperature, timeout, retries, or max_tokens."
            )
        if type(self.max_tokens) is not int or self.max_tokens > 393216:
            raise ValueError("max_tokens must be an integer between 1 and 393216.")
        if type(self.retries) is not int or type(self.stream) is not bool:
            raise ValueError("retries must be an integer and stream must be a boolean.")
        if self.thinking not in ("enabled", "disabled"):
            raise ValueError("thinking must be enabled or disabled.")
        if self.reasoning_effort not in (None, "low", "high", "max"):
            raise ValueError("reasoning_effort must be null, low, high or max.")
        if self.thinking == "disabled" and self.reasoning_effort is not None:
            raise ValueError(
                "reasoning_effort requires thinking=enabled; do not send conflicting controls."
            )


class _PublicTextBuffer:
    """Coalesce public deltas and redact secrets even when split across chunks."""

    def __init__(self, emit: Callable, secret: str):
        self.emit = emit
        self.secret = secret
        self.pending = ""
        self.last_emit = time.perf_counter()

    def feed(self, text: str = "", *, final: bool = False) -> None:
        self.pending += text
        # Hold enough suffix to recognize a key whose next chunk has not arrived.
        cut = (
            len(self.pending)
            if final
            else max(0, len(self.pending) - max(2, len(self.secret) - 1))
        )
        if not final:
            spans = [
                (m.start(), m.end())
                for m in re.finditer(r"sk-[A-Za-z0-9_-]*", self.pending)
            ]
            if self.secret:
                spans.extend(
                    (m.start(), m.end())
                    for m in re.finditer(re.escape(self.secret), self.pending)
                )
            for start, end in sorted(spans, reverse=True):
                if start < cut < end or (end == len(self.pending) and start < cut):
                    cut = start
        if cut and (final or cut >= 256 or time.perf_counter() - self.last_emit >= 1.0):
            visible = redact_secrets(self.pending[:cut], (self.secret,))
            self.pending = self.pending[cut:]
            self.emit(visible)
            self.last_emit = time.perf_counter()


REQUIRED_RESPONSE_FIELDS = {
    "algorithm_code",
    "algorithm_idea",
    "change_hypothesis",
    "expected_improvement",
    "change_summary",
    "direction_id",
}


def parse_model_response(
    content: str, expected_direction: str | None = None
) -> ModelCandidate:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"DeepSeek response must be one JSON object "
            f"(line {error.lineno}, column {error.colno}, chars={len(content)})."
        ) from error
    if (
        isinstance(data, dict)
        and set(data) == REQUIRED_RESPONSE_FIELDS | {"type"}
        and data["type"] == "json_object"
    ):
        data = {k: v for k, v in data.items() if k != "type"}
    if not isinstance(data, dict) or set(data) != REQUIRED_RESPONSE_FIELDS:
        raise ValueError(
            "DeepSeek response must contain exactly the required provenance fields."
        )
    if not all(
        isinstance(data[key], str) and data[key].strip()
        for key in REQUIRED_RESPONSE_FIELDS
    ):
        raise ValueError("DeepSeek response fields must be non-empty strings.")
    if expected_direction and data["direction_id"] != expected_direction:
        raise ValueError(
            f"DeepSeek response direction_id must be {expected_direction!r}."
        )
    return ModelCandidate(**data)


class DeepSeekClient:
    def __init__(self, settings: DeepSeekSettings):
        if not settings.api_key:
            raise ValueError("Missing API key in config/deepseek.local.json.")
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "Install stage1+llm_v2/requirements.txt before optimize/resume."
            ) from error
        import httpx

        self.observer = None
        self.actual_requests = 0
        self.last_metadata: dict = {}
        self.last_response: dict = {}
        http_client = httpx.Client(event_hooks={"request": [self._on_request]})
        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.retries,
            http_client=http_client,
        )
        self.settings = settings

    def set_observer(self, observer: Callable) -> None:
        self.observer = observer

    def _on_request(self, _request) -> None:
        self.actual_requests += 1
        if self.observer:
            self.observer(
                f"Provider HTTP request attempt={self.actual_requests}; configured retries={self.settings.retries}",
                actual_attempt=self.actual_requests,
            )

    def close(self) -> None:
        self.client.close()

    def _record_response(
        self,
        content: str,
        reason: str | None,
        response_id: str | None,
        model: str,
        usage: dict | None,
        choice_count: int,
        **fields,
    ) -> None:
        self.last_metadata.update(
            usage=usage,
            finish_reason=reason,
            response_id=response_id,
            response_chars=len(content),
            choice_count=choice_count,
            **fields,
        )
        # Whitelist public text. Never serialize a whole SDK message/delta object.
        self.last_response = {**self.last_metadata, "model": model, "content": content}
        if self.observer:
            self.observer(
                "Public LLM response received",
                kind="llm_response",
                **self.last_response,
            )

    def _read_stream(self, stream) -> tuple[str, str | None, int]:
        pieces = []
        reason, response_id, usage = None, None, None
        model, choice_count, complete = self.settings.model, 0, False
        stream_error = None

        def emit(text: str) -> None:
            if self.observer:
                self.observer("Public response text", kind="llm_text", text=text)

        display = _PublicTextBuffer(emit, self.settings.api_key)
        try:
            for index, chunk in enumerate(stream):
                if index == 0 and self.observer:
                    self.observer(
                        "Provider stream connected; receiving response (not yet validated)",
                        kind="llm_stream_start",
                    )
                response_id = getattr(chunk, "id", None) or response_id
                model = getattr(chunk, "model", None) or model
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage.model_dump()
                choices = getattr(chunk, "choices", None) or []
                choice_count = max(choice_count, len(choices))
                for choice in choices:
                    if getattr(choice, "index", 0) != 0:
                        raise ValueError(
                            "Unexpected multiple stream choices; refusing ambiguous candidate."
                        )
                    text = (
                        getattr(getattr(choice, "delta", None), "content", None) or ""
                    )
                    if text:
                        if reason is not None:
                            raise ValueError(
                                "Public text arrived after stream finish_reason."
                            )
                        pieces.append(text)
                        display.feed(text)
                    reason = getattr(choice, "finish_reason", None) or reason
            complete = reason is not None
        except BaseException as error:
            # A lost/interrupted stream is never silently replayed or accepted.
            stream_error = type(error).__name__
            raise
        finally:
            try:
                display.feed(final=True)
                self._record_response(
                    "".join(pieces),
                    reason,
                    response_id,
                    model,
                    usage,
                    choice_count,
                    stream_complete=complete,
                    stream_error=stream_error,
                )
            finally:
                stream.close()
        return "".join(pieces), reason, choice_count

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        expected_direction: str | None = None,
    ) -> ModelCandidate:
        self.actual_requests = 0
        start = time.perf_counter()
        self.last_metadata = {}
        self.last_response = {}
        try:
            request = dict(
                model=self.settings.model,
                max_tokens=self.settings.max_tokens,
                response_format={"type": "json_object"},
                stream=self.settings.stream,
                extra_body={"thinking": {"type": self.settings.thinking}},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            if self.settings.thinking == "disabled":
                request["temperature"] = self.settings.temperature
            elif self.settings.reasoning_effort is not None:
                request["reasoning_effort"] = self.settings.reasoning_effort
            if self.settings.stream:
                request["stream_options"] = {"include_usage": True}
            if self.observer:
                self.observer(
                    f"thinking={self.settings.thinking} effort={self.settings.reasoning_effort} "
                    f"max_tokens={self.settings.max_tokens} stream={self.settings.stream} retries={self.settings.retries} "
                    f"prompt_chars={len(system_prompt)+len(user_prompt)}",
                    kind="llm_request_config",
                )
            response = self.client.chat.completions.create(**request)
            if self.settings.stream:
                content, reason, choice_count = self._read_stream(response)
            else:
                usage = getattr(response, "usage", None)
                choices = getattr(response, "choices", None) or []
                choice = choices[0] if choices else None
                content = (
                    getattr(getattr(choice, "message", None), "content", None) or ""
                )
                reason = getattr(choice, "finish_reason", None)
                choice_count = len(choices)
                self._record_response(
                    content,
                    reason,
                    getattr(response, "id", None),
                    getattr(response, "model", self.settings.model),
                    usage.model_dump() if usage else None,
                    choice_count,
                )
            try:
                if not choice_count:
                    raise ValueError(
                        "DeepSeek response has no choices; inspect response diagnostics."
                    )
                if self.settings.stream and reason is None:
                    raise ValueError(
                        "DeepSeek stream ended without finish_reason; refusing partial content."
                    )
                if reason == "length":
                    raise ValueError(
                        "DeepSeek response truncated (finish_reason=length); inspect max_tokens/context and usage."
                    )
                if reason not in (None, "stop"):
                    raise ValueError(
                        f"DeepSeek response incomplete (finish_reason={reason}); refusing candidate."
                    )
                if not content.strip():
                    raise ValueError(
                        "DeepSeek returned empty final content; inspect finish_reason and usage."
                    )
                parsed = parse_model_response(content, expected_direction)
                raw = content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
                body = json.loads(raw)
                if "type" in body:
                    self.last_metadata["normalization"] = (
                        "removed_known_type_json_object"
                    )
                    if self.observer:
                        self.observer(
                            "Removed only known extra type=json_object; original public response retained.",
                            kind="llm_response_normalized",
                        )
                return parsed
            except ValueError as error:
                if self.observer:
                    self.observer(str(error), kind="llm_response_invalid")
                raise
        finally:
            self.last_metadata.update(
                actual_requests=self.actual_requests,
                sdk_retries=max(0, self.actual_requests - 1),
                elapsed_seconds=time.perf_counter() - start,
                usage_note="Usage, when present, is from final response only; not an invoice or total retry usage.",
            )
