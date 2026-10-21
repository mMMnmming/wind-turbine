"""Regression contracts for the independent v3 deployment search."""
import contextlib
import dataclasses
import io
import json
import math
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'stage1+llm'))

from stage1_llm.artifacts import RunArtifacts
from stage1_llm.candidate import CandidateProposal
from stage1_llm.client import DeepSeekSettings, ModelCandidate
from stage1_llm.data import prepare_data, read_bundle
from stage1_llm.domain import DeploymentConfig, Turbine
from stage1_llm.evaluator import evaluate_proposal
from stage1_llm.exports import export_candidate, export_evaluation, list_candidates
from stage1_llm.reward import dominates, pareto_front, parent_pool, search_reward
from stage1_llm.routing import directed_times, proxy_route
from stage1_llm.search import SearchController
from stage1_llm.settings import SearchSettings


SOLO = "def solve(turbines, config, seed):\n return {'nest_ids':[t.identifier for t in turbines], 'assignments':{t.identifier:t.identifier for t in turbines}}\n"
GROUP = "def solve(turbines, config, seed):\n n=turbines[0].identifier\n return {'nest_ids':[n], 'assignments':{t.identifier:n for t in turbines}}\n"
BROKEN = "def solve(turbines, config, seed):\n return {'nest_ids':[turbines[0].identifier], 'assignments':{}}\n"


def settings_at(root, **overrides):
    source = root / 'tiny.tsv'
    if not source.exists():
        source.write_text('a\t0\t0\nb\t0.1\t0\nc\t0\t0.1\nd\t0.1\t0.1\n', encoding='utf-8')
    defaults = dict(target_input=source, validation_inputs=(), deployment=DeploymentConfig(),
                    data_dir=root / 'data', protected_input=ROOT / 'turbine87.tsv',
                    training_virtual_fields=0, validation_virtual_fields=0,
                    initial_candidates=1, evaluation_budget=3, heartbeat_seconds=0.02)
    defaults.update(overrides)
    return SearchSettings(**defaults)


class FakeClient:
    def __init__(self, sources):
        self.sources = iter(sources)
        self.prompts = []

    def generate(self, system, prompt, direction):
        self.prompts.append(prompt)
        source = next(self.sources)
        if isinstance(source, Exception):
            raise source
        return ModelCandidate(source, '模拟算法', '模拟修改', '模拟预期', '模拟摘要', direction)


def controller(settings, root, sources):
    return SearchController(settings, DeepSeekSettings('https://example.invalid', 'mock', 'private-key-value'),
                            root / 'outputs', client=FakeClient(sources))


class DataTests(unittest.TestCase):
    def test_reproducible_disjoint_and_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root, training_virtual_fields=2, validation_virtual_fields=2)
            bundle = prepare_data(s)
            self.assertEqual(len(bundle['datasets']), 5)
            self.assertEqual(bundle, prepare_data(s))
            records = [row['turbines'] for row in bundle['datasets']]
            self.assertEqual(len({json.dumps(row, sort_keys=True) for row in records}), 5)
            other = dataclasses.replace(s, data_dir=root / 'second')
            self.assertEqual(bundle, prepare_data(other))
            with self.assertRaisesRegex(ValueError, 'identity|different|changed'):
                prepare_data(dataclasses.replace(s, data_seed=999))
            payload = json.loads((s.data_dir / 'manifest.json').read_text(encoding='utf-8'))
            payload['datasets'][1]['turbines'][0]['x_km'] += 1
            (s.data_dir / 'manifest.json').write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash|fingerprint|corrupt'):
                read_bundle(s)

    def test_protected_coordinates_not_just_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root)
            protected = root / 'protected.tsv'
            protected.write_text('different-id\t0\t0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '87|protected'):
                prepare_data(dataclasses.replace(s, protected_input=protected))

    def test_degenerate_kde_and_overlapping_seed_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root, training_virtual_fields=2, validation_virtual_fields=2,
                            data_seed=4, validation_seed=5)
            with self.assertRaisesRegex(ValueError, 'seed'):
                prepare_data(s)
            s.target_input.write_text('a\t0\t0\nb\t1\t0\nc\t2\t0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'KDE|covariance|singular'):
                prepare_data(dataclasses.replace(s, validation_seed=100))


class PhysicsTests(unittest.TestCase):
    def test_self_assignment_and_finite_input(self):
        turbines = [Turbine('a', 0, 0), Turbine('b', .1, 0)]
        result = evaluate_proposal(turbines, DeploymentConfig(), CandidateProposal(['a', 'b'], {'a':'b', 'b':'a'}))
        self.assertFalse(result.feasible)
        self.assertIn('self', ' '.join(result.violations).lower())
        self.assertFalse(evaluate_proposal([Turbine('a', math.nan, 0)], DeploymentConfig(),
                         CandidateProposal(['a'], {'a':'a'})).feasible)

    def test_direction_return_and_zero_tasks(self):
        points = [Turbine('a', 0, 0), Turbine('b', 1, 0)]
        times = directed_times(points, DeploymentConfig())
        self.assertAlmostEqual(times[0, 1], 1000 / 29)
        self.assertAlmostEqual(times[1, 0], 1000 / 17)
        route, flight = proxy_route(points, DeploymentConfig(), ['a', 'b'])
        self.assertEqual(route, ('a', 'b', 'a'))
        self.assertAlmostEqual(flight, 1000 / 29 + 1000 / 17)
        route, flight = proxy_route(points[:1], DeploymentConfig(), ['a'])
        self.assertEqual(route, ('a', 'a'))
        self.assertEqual(flight, 0)

    def test_metrics_and_exact_endurance_boundary(self):
        points = [Turbine('a', 0, 0), Turbine('b', .1, 0)]
        limit = 200 / 17 + 22.8
        s = DeploymentConfig(max_endurance_seconds=limit)
        result = evaluate_proposal(points, s, CandidateProposal(['a'], {'a':'a','b':'a'}))
        self.assertTrue(result.feasible)
        self.assertEqual(result.metrics.maximum_region_tasks, 1)
        self.assertGreater(result.metrics.total_proxy_flight_seconds, 0)
        self.assertAlmostEqual(result.deployment.nests[0].equivalent_energy_seconds, limit)
        self.assertFalse(evaluate_proposal(points, dataclasses.replace(s,max_endurance_seconds=limit-.001),
                        CandidateProposal(['a'], {'a':'a','b':'a'})).feasible)


class ObjectiveTests(unittest.TestCase):
    def test_tradeoffs_and_full_front(self):
        a = {'candidate_id':'C0001','objectives':[.1,10,.5]}
        b = {'candidate_id':'C0002','objectives':[.2,8,.3]}
        c = {'candidate_id':'C0003','objectives':[.3,12,.6]}
        self.assertFalse(dominates(a['objectives'],b['objectives']))
        self.assertEqual(pareto_front([a,b,c]), [a,b])
        self.assertEqual(len(parent_pool([a,b],1)),1)
        self.assertEqual(search_reward(False, a['objectives'], b['objectives'], new_front=True),-1)
        self.assertEqual(search_reward(True, a['objectives'], None, duplicate=True),0)
        self.assertEqual(search_reward(True, a['objectives'], None),1)
        self.assertEqual(search_reward(True,[.1,7,.2],b['objectives']),.75)
        self.assertEqual(search_reward(True,c['objectives'],b['objectives']),-.5)
        self.assertEqual(search_reward(True,a['objectives'],b['objectives'],new_front=True),.25)

    def test_pool_preserves_objective_extremes(self):
        rows = [{'candidate_id':f'C{i:04}', 'objectives':[i, 20-i, (i-10)**2]} for i in range(20)]
        pool = parent_pool(pareto_front(rows), 12)
        self.assertEqual(len(pool), 12)
        self.assertTrue({'C0000','C0010','C0019'} <= {r['candidate_id'] for r in pool})


class LoggingTests(unittest.TestCase):
    def test_console_events_heartbeat_and_atomic_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                artifacts = RunArtifacts(Path(directory),'logs',secrets=('private-key-value',))
                artifacts.log('request private-key-value', event='request')
                with artifacts.heartbeat('waiting', .01):
                    time.sleep(.035)
            self.assertIn('heartbeat', stream.getvalue())
            self.assertNotIn('private-key-value', stream.getvalue())
            rows = [json.loads(line) for line in (artifacts.root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertTrue(all('timestamp' in r for r in rows))
            artifacts.write_json('state.json', {'value':1})
            with patch('stage1_llm.artifacts.os.replace',side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    artifacts.write_json('state.json', {'value':2})
            self.assertEqual(json.loads((artifacts.root/'state.json').read_text())['value'],1)


class SearchTests(unittest.TestCase):
    def test_rounds_failure_front_resume_and_manual_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            s = settings_at(root)
            c = controller(s, root, [ValueError('private-key-value invalid JSON'), SOLO, GROUP])
            path = c.run('rounds')
            self.assertEqual(len(c.history),3)
            self.assertEqual(c.history[0]['evaluation']['status'],'generation_error')
            self.assertEqual(len(c.archive),2)
            self.assertFalse((path/'best_algorithm.py').exists())
            self.assertTrue((path/'pareto.csv').exists())
            self.assertNotIn('private-key-value',(path/'search.log').read_text(encoding='utf-8'))
            self.assertTrue(list_candidates(path))
            with self.assertRaisesRegex(ValueError,'mock'):
                export_candidate(path,c.archive[0]['candidate_id'])
            exported = export_candidate(path,c.archive[0]['candidate_id'],allow_mock=True)
            self.assertTrue((exported/'algorithm.py').exists())
            resumed = controller(dataclasses.replace(s,evaluation_budget=4),root,[GROUP+'\n# duplicate behavior'])
            resumed.run('rounds',resume=True)
            self.assertEqual(len(resumed.history),4)
            self.assertEqual(resumed.history[-1]['reward'],0)
            with self.assertRaisesRegex(ValueError,'identity|configuration'):
                controller(dataclasses.replace(s,deployment=DeploymentConfig(wind_speed_mps=5)),root,[]).run('rounds',resume=True)
            with self.assertRaises(FileExistsError):
                c.run('rounds')

    def test_repairs_actual_failed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c = controller(settings_at(root,evaluation_budget=2),root,[BROKEN, GROUP])
            c.run('repair')
            self.assertIn(BROKEN,c.client.prompts[1])
            self.assertEqual(c.history[1]['direction_id'],'constraint_repair')
            self.assertEqual(c.history[1]['reward'],1)

    def test_validation_failure_never_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root/'validation.tsv'
            validation.write_text('v1\t0\t0\nv2\t20\t0\n',encoding='utf-8')
            s = settings_at(root,validation_inputs=(validation,),evaluation_budget=1)
            c = controller(s,root,[GROUP])
            c.run('validation')
            self.assertFalse(c.archive)
            self.assertLess(c.history[0]['reward'],0)
            self.assertTrue(c.history[0]['evaluation']['datasets'][0]['feasible'])

    def test_export_shape_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            points = [Turbine('a',0,0),Turbine('b',.1,0)]
            evaluated = evaluate_proposal(points,DeploymentConfig(),CandidateProposal(['a'],{'a':'a','b':'a'}))
            export_evaluation(root/'export',points,evaluated,{'kind':'unit-test'})
            regions = json.loads((root/'export'/'regions.json').read_text(encoding='utf-8'))
            self.assertEqual(regions['regions'][0]['identifiers'],['a','b'])
            self.assertEqual(regions['regions'][0]['config']['gust_speed_mps'],12)
            summary = json.loads((root/'export'/'summary.json').read_text(encoding='utf-8'))
            self.assertTrue(summary['all_constraints_satisfied'])
            with self.assertRaises(FileExistsError):
                export_evaluation(root/'export',points,evaluated,{})


if __name__ == '__main__':
    unittest.main()
