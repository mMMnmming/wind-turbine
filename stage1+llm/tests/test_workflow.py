"""CLI, interruption, independent-reader and provider-adapter regressions."""
import contextlib
import dataclasses
import importlib.util
import io
import json
import itertools
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_v3 import ROOT, SOLO, GROUP, BROKEN, controller, settings_at
from stage1_llm import cli
from stage1_llm.artifacts import RunArtifacts, run_path
from stage1_llm.client import DeepSeekClient, DeepSeekSettings
from stage1_llm.data import prepare_data
from stage1_llm.domain import DeploymentConfig, load_turbines
from stage1_llm.evaluator import evaluate_proposal
from stage1_llm.exports import export_evaluation, export_candidate
from stage1_llm.sandbox import run_candidate
from stage1_llm.search import SearchController
from stage1_llm.routing import directed_times, proxy_route
from stage1_llm.domain import Turbine
from stage1_llm.exports import candidate_origin


class WorkflowTests(unittest.TestCase):
    def test_sensitive_source_rejected_without_unrecoverable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sensitive = SOLO+'\n# sk-abcdef123456'
            c = controller(settings_at(root,evaluation_budget=2),root,[sensitive,SOLO])
            path = c.run('sensitive-source')
            self.assertEqual(c.history[0]['evaluation']['status'],'generation_error')
            self.assertTrue(c.archive)
            self.assertNotIn('sk-abcdef123456',(path/'search_state.json').read_text(encoding='utf-8'))
            controller(settings_at(root,evaluation_budget=2),root,[]).run('sensitive-source',resume=True)

    def test_mock_provenance_survives_selection_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(settings_at(root,evaluation_budget=1),root,[SOLO])
            path = c.run('origin')
            selected = export_candidate(path,'C0001',allow_mock=True)
            origin = candidate_origin(selected/'algorithm.py',SOLO)
            self.assertEqual(origin['origin'],'offline_mock')
            self.assertTrue(origin['selected'])
            self.assertFalse(candidate_origin(path/'candidates/0001.py',SOLO)['selected'])
            with self.assertRaisesRegex(ValueError,'hash'):
                candidate_origin(selected/'algorithm.py',SOLO+'\n# changed')

    def test_directed_proxy_cost_matches_route_and_exhaustive_lower_bound(self):
        import numpy as np
        config = DeploymentConfig()
        for seed in range(5):
            rng = np.random.default_rng(seed)
            points = [Turbine(str(i),float(x),float(y)) for i,(x,y) in enumerate(rng.random((7,2)))]
            route,flight = proxy_route(points,config,[p.identifier for p in points])
            times = directed_times(points,config)
            indices = [int(key) for key in route]
            self.assertAlmostEqual(flight,sum(times[a,b] for a,b in zip(indices,indices[1:])))
            optimum = min(sum(times[a,b] for a,b in zip((0,*order,0),(0,*order,0)[1:]))
                          for order in itertools.permutations(range(1,len(points))))
            self.assertGreaterEqual(flight+1e-8,optimum)
        # Larger instances exercise relocation; still no optimality claim.
        for seed in range(3):
            rng = np.random.default_rng(seed)
            points = [Turbine(str(i),float(x),float(y)) for i,(x,y) in enumerate(rng.random((15,2)))]
            route,flight = proxy_route(points,config,[p.identifier for p in points])
            self.assertEqual(set(route),{p.identifier for p in points})
            self.assertTrue(flight >= 0)

    def test_source_write_interruption_recovers_saved_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root,evaluation_budget=1)
            first = controller(s,root,[SOLO])
            with patch.object(RunArtifacts,'write_text',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    first.run('write-interrupted')
            state = json.loads((root/'outputs/write-interrupted/search_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['pending']['stage'],'source_ready')
            resumed = controller(s,root,[])
            resumed.run('write-interrupted',resume=True)
            self.assertTrue(resumed.archive)
            self.assertEqual(resumed.client.prompts,[])

    def test_run_id_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_path(Path(directory),'../outside')
            with self.assertRaises(SystemExit):
                cli._parser().parse_args(['export-best','--run-id','../outside','--candidate-id','C0001'])

    def test_mid_dataset_resume_skips_completed_dataset_and_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root/'valid.tsv'
            validation.write_text('e\t1\t1\nf\t1.1\t1\n',encoding='utf-8')
            s = settings_at(root,validation_inputs=(validation,),evaluation_budget=1)
            first = controller(s,root,[SOLO])
            original = run_candidate
            calls = []
            def interrupt(path,points,*args):
                calls.append(points[0].identifier)
                if points[0].identifier == 'e':
                    raise KeyboardInterrupt
                return original(path,points,*args)
            with patch('stage1_llm.search.run_candidate',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    first.run('interrupted')
            saved = json.loads((root/'outputs/interrupted/search_state.json').read_text(encoding='utf-8'))
            self.assertEqual(len(saved['pending']['completed_datasets']),1)
            resumed = controller(s,root,[])
            with patch('stage1_llm.search.run_candidate',wraps=original) as checked:
                resumed.run('interrupted',resume=True)
                self.assertEqual(checked.call_count,1)
            self.assertEqual(len(resumed.history),1)
            self.assertEqual(resumed.client.prompts,[])

    def test_interrupted_generation_not_replayed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root,evaluation_budget=1)
            first = controller(s,root,[])
            with patch.object(first.client,'generate',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    first.run('api-interrupt')
            resumed = controller(s,root,[])
            resumed.run('api-interrupt',resume=True)
            self.assertEqual(resumed.history[0]['evaluation']['status'],'generation_error')
            self.assertEqual(resumed.client.prompts,[])

    def test_budget_source_version_and_identity_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root,evaluation_budget=2)
            c = controller(s,root,[SOLO,GROUP])
            path = c.run('guards')
            with self.assertRaisesRegex(ValueError,'budget'):
                controller(dataclasses.replace(s,evaluation_budget=1),root,[]).run('guards',resume=True)
            source = path/c.archive[0]['source_path']
            source.write_text(SOLO+'\n# tampered',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'hash'):
                controller(s,root,[]).run('guards',resume=True)
            with self.assertRaisesRegex(ValueError,'hash'):
                export_candidate(path,c.archive[0]['candidate_id'],allow_mock=True)
            with self.assertRaisesRegex(ValueError,'Candidate'):
                export_candidate(path,'C9999',allow_mock=True)
            state = json.loads((path/'search_state.json').read_text(encoding='utf-8'))
            state['search_version'] = 2
            (path/'search_state.json').write_text(json.dumps(state),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'Legacy'):
                controller(s,root,[]).run('guards',resume=True)

    def test_candidate_error_and_duplicate_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(settings_at(root,evaluation_budget=3),root,[SOLO,SOLO,'def solve(turbines, config, seed):\n raise ValueError("fixture error")'])
            c.run('errors')
            self.assertEqual(c.history[1]['evaluation']['status'],'duplicate_source')
            self.assertEqual(c.history[1]['reward'],0)
            self.assertEqual(c.history[2]['evaluation']['status'],'candidate_error')

    def test_independent_stage2_and_bnb_readers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root)
            candidate = root/'source.py'
            candidate.write_text(GROUP,encoding='utf-8')
            points = load_turbines(s.target_input)
            evaluated = evaluate_proposal(points,s.deployment,run_candidate(candidate,points,s.deployment,1,5))
            export_evaluation(root/'export',points,evaluated,{'kind':'offline_mock'})
            sys.path.insert(0,str(ROOT))
            from stage2.data import load_region_instances,instance_fingerprint
            regions = load_region_instances(root/'export/regions.json',ROOT/'stage1/config.json')
            payload = json.loads((root/'export/regions.json').read_text(encoding='utf-8'))
            self.assertEqual(payload['instance_fingerprints'],[instance_fingerprint(r) for r in regions])
            spec = importlib.util.spec_from_file_location('bnb_inputs_readonly',ROOT/'stage2/Branch and Bound/inputs.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            audit,_ = module.load_stage1_input(root/'export/summary.json',s.target_input)
            self.assertEqual([instance_fingerprint(r) for r in audit],payload['instance_fingerprints'])

    def test_cli_prepare_evaluate_errors_and_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root)
            config = root/'config.json'
            config.write_text(json.dumps(dataclasses.asdict(s),default=str),encoding='utf-8')
            source = root/'algorithm.py'
            source.write_text(GROUP,encoding='utf-8')
            common = ['--search-config',str(config),'--outputs',str(root/'outputs')]
            self.assertEqual(cli.main(['prepare-data',*common]),0)
            self.assertEqual(cli.main(['evaluate',*common,'--candidate',str(source),'--input',str(s.target_input),'--run-id','evaluated']),0)
            self.assertEqual(cli.main(['evaluate',*common,'--candidate',str(source),'--input',str(s.target_input),'--run-id','evaluated']),1)
            source.write_text(BROKEN,encoding='utf-8')
            self.assertEqual(cli.main(['evaluate',*common,'--candidate',str(source),'--input',str(s.target_input),'--run-id','invalid']),2)
            self.assertEqual(cli.main(['smoke-test','--outputs',str(root/'outputs'),'--run-id','smoke']),0)
            self.assertEqual(cli.main(['list-candidates','--outputs',str(root/'outputs'),'--run-id','smoke']),0)
            self.assertEqual(cli.main(['export-best','--outputs',str(root/'outputs'),'--run-id','smoke','--candidate-id','C0002']),1)

    def test_cli_optimize_resume_mock_adapter_no_paid_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root,evaluation_budget=1)
            c = controller(s,root,[SOLO])
            c.client.close = lambda: None
            config = root/'config.json'
            config.write_text(json.dumps(dataclasses.asdict(s),default=str),encoding='utf-8')
            local = root/'fake-private.json'
            local.write_text(json.dumps(dataclasses.asdict(c.llm)),encoding='utf-8')
            common = ['--search-config',str(config),'--deepseek-config',str(local),'--outputs',str(root/'outputs')]
            with patch('stage1_llm.cli.SearchController',return_value=c):
                self.assertEqual(cli.main(['optimize',*common,'--run-id','cli']),0)
                # No new round; check existing outputs and argument plumbing.
                self.assertEqual(cli.main(['resume',*common,'--resume-run-id','cli','--heartbeat-seconds','.03']),0)
            with patch('stage1_llm.cli.load_search_settings',side_effect=KeyboardInterrupt):
                self.assertEqual(cli.main(['prepare-data',*common[:2]]),130)


class ClientTests(unittest.TestCase):
    def test_adapter_counts_actual_transport_attempts_and_failure_usage(self):
        # SDK and transport are in-memory stubs; no socket is opened.
        hooks = {}
        created = []
        class HttpClient:
            def __init__(self, event_hooks):
                hooks.update(event_hooks)
        class SDK:
            def __init__(self, **kwargs):
                created.append(self)
                self.chat = types.SimpleNamespace(completions=self)
            def close(self):
                pass
            def create(self, **kwargs):
                hooks['request'][0](None)
                hooks['request'][0](None)
                body = json.dumps(dict(algorithm_code=SOLO,algorithm_idea='i',change_hypothesis='h',
                                       expected_improvement='e',change_summary='s',direction_id='reduce_nests'))
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=body))],
                                             usage=types.SimpleNamespace(model_dump=lambda:{'total_tokens':123}))
        modules = {'openai':types.SimpleNamespace(OpenAI=SDK),'httpx':types.SimpleNamespace(Client=HttpClient)}
        with patch.dict(sys.modules,modules):
            client = DeepSeekClient(DeepSeekSettings('https://example.invalid','mock','private-key',stream=False))
            seen = []
            client.set_observer(lambda message,**kwargs:seen.append(kwargs))
            client.generate('system','prompt','reduce_nests')
            self.assertEqual(client.last_metadata['actual_requests'],2)
            self.assertEqual(client.last_metadata['sdk_retries'],1)
            self.assertEqual(client.last_metadata['usage']['total_tokens'],123)
            self.assertEqual(len([row for row in seen if 'actual_attempt' in row]),2)
            with patch.object(created[0],'create',side_effect=ValueError('provider failure')):
                with self.assertRaises(ValueError):
                    client.generate('system','prompt','reduce_nests')
            self.assertEqual(client.last_metadata['actual_requests'],0)
            client.close()


if __name__ == '__main__':
    unittest.main()
