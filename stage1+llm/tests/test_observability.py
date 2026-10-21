"""Public LLM output diagnostics, without paid requests or private reasoning."""
import contextlib
import dataclasses
import io
import json
import tempfile
import types
import unittest
from pathlib import Path

from test_v3 import SOLO, settings_at
from stage1_llm.artifacts import RunArtifacts
from stage1_llm.client import DeepSeekClient, DeepSeekSettings
from stage1_llm.search import SearchController


def response_for(content, finish_reason='stop'):
    return types.SimpleNamespace(id='response-fixture',model='mock-model',choices=[
        types.SimpleNamespace(finish_reason=finish_reason,message=types.SimpleNamespace(
            content=content,reasoning_content='PRIVATE_REASONING_MUST_NOT_APPEAR'))],
        usage=types.SimpleNamespace(model_dump=lambda:{'prompt_tokens':100,'completion_tokens':30,
            'total_tokens':130,'completion_tokens_details':{'reasoning_tokens':10}}))


def public_json(direction='structural_exploration'):
    return json.dumps(dict(algorithm_code=SOLO,algorithm_idea='公开算法思路',change_hypothesis='公开修改假设',
        expected_improvement='公开预期',change_summary='公开摘要',direction_id=direction),ensure_ascii=False)


def stub_client(response):
    client = DeepSeekClient.__new__(DeepSeekClient)
    client.settings = DeepSeekSettings('https://example.invalid','mock','private-key-value')
    client.observer = None
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw:response)))
    return client


class ResponseDiagnosticsTests(unittest.TestCase):
    def test_public_response_metadata_before_parse(self):
        client = stub_client(response_for(public_json()))
        events = []
        client.set_observer(lambda message,**fields:events.append((message,fields)))
        client.generate('system','prompt','structural_exploration')
        self.assertEqual(client.last_response['finish_reason'],'stop')
        self.assertEqual(client.last_response['usage']['total_tokens'],130)
        self.assertIn('公开算法思路',client.last_response['content'])
        self.assertNotIn('PRIVATE_REASONING',json.dumps(client.last_response))
        self.assertEqual(events[0][1]['kind'],'llm_response')

    def test_empty_length_and_invalid_json_are_distinct(self):
        for content,reason,expected in [('', 'stop','empty'),('{"algorithm_code":', 'length','truncated'),
                                        ('not json','stop','JSON'),(public_json(),'content_filter','content_filter')]:
            with self.subTest(reason=reason,content=content):
                client = stub_client(response_for(content,reason))
                with self.assertRaisesRegex(ValueError,expected):
                    client.generate('system','prompt','structural_exploration')
                self.assertEqual(client.last_response['content'],content)
                self.assertEqual(client.last_metadata['finish_reason'],reason)

    def test_no_choices_is_diagnostic_failure(self):
        response = response_for('')
        response.choices = []
        client = stub_client(response)
        with self.assertRaisesRegex(ValueError,'choices'):
            client.generate('system','prompt','structural_exploration')
        self.assertEqual(client.last_response['choice_count'],0)

    def test_console_public_summary_code_and_failed_body_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root,evaluation_budget=1)
            # Wrap a transport stub, but mark the search as a mock run.
            inner = stub_client(response_for(public_json()))
            class MockAdapter:
                set_observer = inner.set_observer
                generate = inner.generate
                @property
                def last_metadata(self):
                    return inner.last_metadata
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                first = SearchController(settings,inner.settings,root/'outputs',client=MockAdapter())
                path = first.run('visible')
            printed = stream.getvalue()
            self.assertIn('公开算法思路',printed)
            self.assertIn('def solve(turbines, config, seed):',printed)
            self.assertIn('finish_reason=stop',printed)
            self.assertIn('total_tokens',printed)
            self.assertNotIn('PRIVATE_REASONING',printed)
            saved = json.loads((path/'responses/C0001.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['content'],public_json())
            self.assertNotIn('reasoning_content',saved)
            inner.client.chat.completions.create = lambda **kwargs:response_for('bad private-key-value body')
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                failed = SearchController(settings,inner.settings,root/'outputs',client=MockAdapter())
                failed_path = failed.run('failed')
            self.assertIn('bad [REDACTED] body',stream.getvalue())
            self.assertNotIn('private-key-value',(failed_path/'responses/C0001.json').read_text(encoding='utf-8'))
            self.assertFalse((failed_path/'candidates/0001.py').exists())

    def test_hide_code_keeps_source_file_and_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = settings_at(root,evaluation_budget=1,show_code=False)
            # Reuse the existing fixture; it need not support response callbacks.
            from test_v3 import controller
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                path = controller(settings,root,[SOLO]).run('hidden')
            self.assertNotIn('def solve(turbines, config, seed):',stream.getvalue())
            self.assertTrue((path/'candidates/0001.py').exists())
            self.assertIn('algorithm_idea',stream.getvalue())

    def test_terminal_control_sequences_are_visible_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                RunArtifacts(Path(directory),'control').log('before\x1b[2J\rafter\nsecond line')
            self.assertNotIn('\x1b',stream.getvalue())
            self.assertNotIn('\r',stream.getvalue())
            self.assertIn('\\x1b',stream.getvalue())


if __name__ == '__main__':
    unittest.main()
