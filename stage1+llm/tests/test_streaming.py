"""Offline API contract and streaming-console regression tests; never call a provider."""

import contextlib
import dataclasses
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage1+llm"))

from test_observability import public_json, response_for, stub_client
from test_v3 import settings_at
from stage1_llm.client import DeepSeekSettings
from stage1_llm.cli import _parser
from stage1_llm.artifacts import RunArtifacts
from stage1_llm.search import SearchController
from stage1_llm.settings import load_deepseek_settings


def chunk(content=None, reason=None, usage=None, choices=True):
    return types.SimpleNamespace(
        id="stream-fixture",
        model="mock",
        usage=usage,
        choices=(
            [
                types.SimpleNamespace(
                    index=0,
                    finish_reason=reason,
                    delta=types.SimpleNamespace(
                        content=content,
                        reasoning_content="PRIVATE_REASONING_DO_NOT_LOG",
                    ),
                )
            ]
            if choices
            else []
        ),
    )


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for item in self.chunks:
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self):
        self.closed = True


def streaming_client(stream):
    client = stub_client(stream)
    client.settings = dataclasses.replace(client.settings, stream=True)
    return client


class StreamingTests(unittest.TestCase):
    def test_economical_defaults_and_config_validation(self):
        defaults = DeepSeekSettings("https://example.invalid", "mock", "unused")
        self.assertEqual(defaults.thinking, "disabled")
        self.assertTrue(defaults.stream)
        self.assertEqual(defaults.max_tokens, 8192)
        self.assertEqual(defaults.retries, 1)
        for overrides in (
            {"thinking": "invalid"},
            {"stream": "true"},
            {"max_tokens": False},
            {"reasoning_effort": "max"},
            {"max_tokens": 393217},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                dataclasses.replace(defaults, **overrides)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                json.dumps(
                    dict(
                        base_url=defaults.base_url,
                        model="mock",
                        api_key="unused",
                        thinking="enabled",
                        reasoning_effort="low",
                        stream=True,
                    )
                ),
                encoding="utf-8",
            )
            loaded = load_deepseek_settings(path)
            self.assertEqual(loaded.reasoning_effort, "low")
            self.assertEqual(loaded.thinking, "enabled")

    def test_cli_can_override_existing_private_budget_without_exposing_key(self):
        args = _parser().parse_args(
            [
                "optimize",
                "--thinking",
                "disabled",
                "--max-tokens",
                "8192",
                "--api-retries",
                "1",
                "--stream",
            ]
        )
        self.assertEqual(args.max_tokens, 8192)
        self.assertEqual(args.api_retries, 1)
        self.assertTrue(args.stream)

    def test_cli_effective_settings_override_old_explicit_budget(self):
        from stage1_llm import cli

        with tempfile.TemporaryDirectory() as directory:
            settings = settings_at(Path(directory))
            old = DeepSeekSettings(
                "https://example.invalid", "mock", "unused", retries=3, max_tokens=10000
            )
            controller = types.SimpleNamespace(
                run=lambda *a, **kw: None,
                client=types.SimpleNamespace(close=lambda: None),
            )
            with patch.object(
                cli, "load_search_settings", return_value=settings
            ), patch.object(
                cli, "load_deepseek_settings", return_value=old
            ), patch.object(
                cli, "SearchController", return_value=controller
            ) as factory:
                self.assertEqual(
                    cli.main(
                        [
                            "optimize",
                            "--thinking",
                            "disabled",
                            "--max-tokens",
                            "8192",
                            "--api-retries",
                            "0",
                            "--stream",
                        ]
                    ),
                    0,
                )
            effective = factory.call_args.args[1]
            self.assertEqual(effective.api_key, "unused")
            self.assertEqual(effective.max_tokens, 8192)
            self.assertEqual(effective.retries, 0)
            self.assertTrue(effective.stream)

    def test_stream_request_public_deltas_usage_and_final_result(self):
        body = public_json()
        stream = FakeStream(
            [
                chunk(body[:120]),
                chunk(body[120:]),
                chunk(reason="stop", usage=response_for("").usage),
            ]
        )
        client = streaming_client(stream)
        events, requests = [], []
        client.client.chat.completions.create = lambda **kw: (
            requests.append(kw) or stream
        )
        client.set_observer(lambda message, **fields: events.append((message, fields)))
        result = client.generate("system", "prompt", "structural_exploration")
        self.assertEqual(result.algorithm_idea, "公开算法思路")
        self.assertEqual(requests[0]["extra_body"], {"thinking": {"type": "disabled"}})
        self.assertNotIn("reasoning_effort", requests[0])
        self.assertTrue(requests[0]["stream"])
        self.assertTrue(requests[0]["stream_options"]["include_usage"])
        public = "".join(
            fields["text"] for _, fields in events if fields.get("kind") == "llm_text"
        )
        self.assertEqual(public, body)
        self.assertNotIn("PRIVATE_REASONING", str(events) + str(client.last_response))
        self.assertEqual(client.last_response["usage"]["total_tokens"], 130)
        self.assertEqual(client.last_response["content"], body)
        self.assertTrue(client.last_response["stream_complete"])
        self.assertTrue(stream.closed)

    def test_usage_only_last_chunk_and_thinking_override(self):
        stream = FakeStream(
            [
                chunk(public_json()),
                chunk(reason="stop"),
                chunk(usage=response_for("").usage, choices=False),
            ]
        )
        client = streaming_client(stream)
        client.settings = dataclasses.replace(
            client.settings, thinking="enabled", reasoning_effort="low"
        )
        requests = []
        client.client.chat.completions.create = lambda **kw: (
            requests.append(kw) or stream
        )
        client.generate("system", "prompt", "structural_exploration")
        self.assertEqual(requests[0]["reasoning_effort"], "low")
        self.assertNotIn("temperature", requests[0])
        self.assertEqual(client.last_metadata["usage"]["total_tokens"], 130)

    def test_truncation_disconnect_and_missing_stop_never_accept_partial(self):
        for ending, expected in [
            (chunk(reason="length"), "truncated"),
            (RuntimeError("connection lost"), "connection lost"),
            (None, "finish_reason"),
            (KeyboardInterrupt(), ""),
        ]:
            with self.subTest(expected=expected):
                stream = FakeStream(
                    [chunk(public_json())] + ([] if ending is None else [ending])
                )
                client = streaming_client(stream)
                with self.assertRaisesRegex(BaseException, expected):
                    client.generate("system", "prompt", "structural_exploration")
                self.assertEqual(client.last_response["content"], public_json())
                self.assertTrue(stream.closed)
                self.assertEqual(client.last_metadata["actual_requests"], 0)

    def test_partial_secret_redaction_across_every_chunk_boundary(self):
        # The known key has punctuation; sk-style keys may be of unknown length.
        secret = "private-key-value"
        body = "公开正文 " + secret + " sk-abcdefghijklmno tail\x1b[2J"
        for boundary in range(1, len(body)):
            with self.subTest(boundary=boundary):
                client = streaming_client(
                    FakeStream(
                        [
                            chunk(body[:boundary]),
                            chunk(body[boundary:]),
                            chunk(reason="stop"),
                        ]
                    )
                )
                events = []
                client.set_observer(
                    lambda message, **fields: events.append((message, fields))
                )
                with self.assertRaisesRegex(ValueError, "JSON"):
                    client.generate("system", "prompt")
                text = "".join(
                    fields["text"]
                    for _, fields in events
                    if fields.get("kind") == "llm_text"
                )
                self.assertEqual(text, "公开正文 [REDACTED] [REDACTED] tail\x1b[2J")

    def test_search_prints_before_stream_finishes_and_saves_interrupted_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            body = public_json()
            client = streaming_client(FakeStream([]))
            observed = []

            def chunks():
                yield chunk(body + " " * 256)
                observed.append("llm_text" in output.getvalue())
                yield chunk(reason="stop", usage=response_for("").usage)

            class GeneratorStream(FakeStream):
                def __iter__(self):
                    return chunks()

            stream = GeneratorStream([])
            client.client.chat.completions.create = lambda **kw: stream

            class Adapter:
                set_observer = client.set_observer
                generate = client.generate

                @property
                def last_metadata(self):
                    return client.last_metadata

            with contextlib.redirect_stdout(output):
                path = SearchController(
                    settings_at(root, evaluation_budget=1),
                    client.settings,
                    root / "outputs",
                    client=Adapter(),
                ).run("streaming")
            self.assertEqual(observed, [True])
            self.assertIn("公开算法思路", output.getvalue())
            saved = json.loads(
                (path / "responses/C0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["content"], body + " " * 256)
            client.client.chat.completions.create = lambda **kw: FakeStream(
                [chunk("partial public output"), RuntimeError("stream lost")]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                failed = SearchController(
                    settings_at(root, evaluation_budget=1),
                    client.settings,
                    root / "outputs",
                    client=Adapter(),
                ).run("disconnected")
            saved = json.loads(
                (failed / "responses/C0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["content"], "partial public output")
            self.assertFalse(saved["stream_complete"])
            self.assertFalse((failed / "candidates/0001.py").exists())

    def test_live_output_redacts_one_character_chunks_without_waiting_for_end(self):
        body = (
            "prefix " * 50 + "private-key-value sk-abcdefghijklmnop " + "suffix " * 50
        )
        client = streaming_client(
            FakeStream([chunk(ch) for ch in body] + [chunk(reason="stop")])
        )
        events = []
        client.set_observer(lambda message, **fields: events.append((message, fields)))
        with self.assertRaisesRegex(ValueError, "JSON"):
            client.generate("system", "prompt")
        deltas = [
            fields["text"] for _, fields in events if fields.get("kind") == "llm_text"
        ]
        self.assertGreater(len(deltas), 1)
        self.assertEqual(
            "".join(deltas), "prefix " * 50 + "[REDACTED] [REDACTED] " + "suffix " * 50
        )

    def test_compact_feedback_keeps_failure_numbers_and_full_parent_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = stub_client(response_for(public_json()))
            controller = SearchController(
                settings_at(root), client.settings, root / "outputs", client=client
            )
            controller.bundle = {"datasets": [{}, {}]}
            artifacts = RunArtifacts(root / "outputs", "prompt")
            source = "def solve(turbines, config, seed):\n return {}\n"
            artifacts.write_text("candidates/parent.py", source)
            violation = "Nest a endurance violation: 2500.000000 > 2460.000000 s."
            row = {
                "name": "validation-kde-1",
                "feasible": False,
                "violations": [violation, violation],
                "metrics": {
                    "nest_count": 1,
                    "minimum_endurance_margin_seconds": -40,
                    "oversized_detail": "DO_NOT_SEND_FULL_DETAILS",
                },
                "proposal": {"assignments": {"x": "y"}},
            }
            parent = {
                "source_path": "candidates/parent.py",
                "evaluation": {
                    "status": "infeasible",
                    "datasets": [row],
                    "violations": [violation],
                },
                "objectives": None,
            }
            prompt = controller._prompt(parent, "constraint_repair", artifacts)
            self.assertIn(source, prompt)
            self.assertEqual(prompt.count(violation), 1)
            self.assertNotIn("DO_NOT_SEND_FULL_DETAILS", prompt)
            self.assertNotIn("assignments", prompt)
            self.assertIn("-40", prompt)

    def test_hidden_code_also_suppresses_invalid_stream_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = "def solve(turbines, config, seed): NOT_JSON"
            client = streaming_client(FakeStream([chunk(body), chunk(reason="stop")]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                path = SearchController(
                    settings_at(root, evaluation_budget=1, show_code=False),
                    client.settings,
                    root / "outputs",
                    client=client,
                ).run("hidden-invalid")
            self.assertNotIn(body, output.getvalue())
            self.assertIn("diagnostic=responses/C0001.json", output.getvalue())
            saved = json.loads(
                (path / "responses/C0001.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved["content"], body)


if __name__ == "__main__":
    unittest.main()
