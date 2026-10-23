"""Offline checks for progress-based search and recoverable callback interruptions."""
import contextlib
import copy
import io
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from windsearch import runtime
from windsearch.evolution import Evolution
from windsearch.execution import StrategyWorker, StrategyCallbackTimeout, StrategyError
from windsearch.planner import InfrastructureError
from windsearch.plotting import execution_marker
from windsearch.settings import SearchSettings
from test_parent_and_plot import fixture_directory, OfflineEvolution, instance_record
from windsearch.artifacts import Records

SOURCE = '''def initialize(problem, initial_solution, seed):
    return {}
def propose(problem, current_solution, state, feedback):
    return {"assignments": None, "state": state, "phase": "done", "done": True}
'''

class MemoryRecords:
    def __init__(self):
        self.saved = {}
    def json(self, path, value): self.saved[path] = copy.deepcopy(value)
    def progress(self, *args): pass
    def event(self, *args, **kwargs): pass
    def append(self, *args): pass
    def blob(self, value): return "fixture"

class FakeEvaluator:
    def __init__(self, *args): self.seconds = 0
    def check(self, assignments, **kwargs):
        assert 'deadline' not in kwargs
        self.seconds += .01
        k = len(set(assignments.values()))
        return {"feasible": True, "objectives": [k, 10., 10.], "diagnostics": [], "regions": [],
                "metrics": {"nest_count": k, "total_flight_seconds": 10.}}

class RuntimeChecks(unittest.TestCase):
    def run_fixture(self, replies, max_proposals=20000):
        records = MemoryRecords()
        planner = SimpleNamespace(reset_cache=lambda: None, settle=lambda: None,
                                  stats=lambda: {"inferences": 0}, metadata={})
        settings = SearchSettings(search_seconds=.001, callback_timeout_seconds=.1, max_proposals=max_proposals)
        class Worker:
            def __init__(self, *args): self.calls = iter(replies)
            def exchange(self, *args, **kwargs):
                result = next(self.calls)
                if isinstance(result, Exception): raise result
                return result
            def close(self): pass
        clock = iter(range(0, 100000, 61))
        dataset = {"name": "fixture", "split": "training", "turbines": [
            {"identifier": x, "x_km": i, "y_km": 0} for i, x in enumerate(('a', 'b'))]}
        with patch.object(runtime, 'Evaluator', FakeEvaluator), patch.object(runtime, 'StrategyWorker', Worker), \
             patch.object(runtime, 'problem_input', return_value={}), \
             patch.object(runtime, 'time', SimpleNamespace(perf_counter=lambda: next(clock))):
            return runtime.run_instance(SOURCE, 'candidate.py', dataset, settings, planner, records, 'A0001', 1, 1)

    def test_no_total_deadline_and_proposal_budget_safe_log(self):
        # Simulated elapsed time already exceeds 120s before the first proposal.
        row = self.run_fixture([{"strategy_seconds": .01}, {"strategy_seconds": .01, "answer": {
            "assignments": {'a':'a', 'b':'a'}, "phase": 'merge', "done": True}}])
        self.assertGreater(row['timing']['search_seconds'], 120)
        self.assertEqual(row['objectives'][0], 1)
        self.assertEqual(row['execution_status'], 'completed')
        self.assertTrue(row['solution_usable'])

    def test_submission_budget_stop_has_safe_log_label(self):
        row = self.run_fixture([{ "strategy_seconds": .01}, {"strategy_seconds": .01, "answer": {
            "assignments": None, "phase": "search", "done": False}}], max_proposals=1)
        self.assertEqual(row["stop_reason"], "proposal_budget_exhausted")
        self.assertTrue(row["code_success"])

    def test_propose_timeout_keeps_accepted_best_and_audits(self):
        row = self.run_fixture([{"strategy_seconds": .01}, {"strategy_seconds": .01, "answer": {
            "assignments": {'a':'a', 'b':'a'}, "phase": 'merge', "done": False}},
            StrategyCallbackTimeout('propose', .1, .1)])
        self.assertEqual(row['objectives'][0], 1)
        self.assertEqual(row['execution_status'], 'interrupted_timeout')
        self.assertFalse(row['code_success'])
        self.assertTrue(row['solution_usable'])
        self.assertFalse(row['only_initial_solution'])
        self.assertIsNone(row['fatal_error'])
        self.assertEqual(execution_marker(row, {'eligible':True}), 'INTERRUPTED')

    def test_initialize_timeout_and_real_code_error_are_distinct(self):
        row = self.run_fixture([StrategyCallbackTimeout('initialize', .1, .1)])
        self.assertTrue(row['only_initial_solution'])
        self.assertTrue(row['solution_usable'])
        self.assertIsNone(row['fatal_error'])
        row = self.run_fixture([StrategyError({'error':'bad state','error_type':'ValueError'})])
        self.assertEqual(row['execution_status'], 'failed')
        self.assertIsNotNone(row['fatal_error'])
        with self.assertRaises(InfrastructureError):
            self.run_fixture([OSError('disk failure')])

    def test_all_interrupted_instances_qualify_as_parent(self):
        with fixture_directory() as path, contextlib.redirect_stdout(io.StringIO()):
            engine = OfflineEvolution(Path(path), {}, target=1)
            row = {'source_path':'candidates/attempts/A0001.py', 'attempt_id':'A0001',
                   'round_index':1, 'parent_ids':[], 'attempt_number':1, 'operator':'initialize'}
            engine.records.text(row['source_path'], SOURCE)
            def interrupted(*args):
                r = instance_record(args[2]['name'])
                r.update(code_success=False, execution_status='interrupted_timeout', solution_usable=True,
                         fatal_error=None, interruptions=[{'operation':'propose'}])
                return r
            with patch('windsearch.evolution.run_instance', side_effect=interrupted):
                Evolution.evaluate(engine, row)
            self.assertTrue(row['eligible'])
            self.assertEqual(row['interrupted_instances'], 17)
            engine.finish(row)
            self.assertEqual(engine.archive()[0]['candidate_id'], 'C0001')

class WorkerChecks(unittest.TestCase):
    def test_real_worker_interrupt_and_real_error(self):
        with fixture_directory() as path:
            records = Records(Path(path))
            for body, error in [('while True:\n        pass', StrategyCallbackTimeout),
                                ('raise ValueError("actual error")', StrategyError)]:
                source = SOURCE.replace('return {}', body)
                worker = StrategyWorker(records, 'watchdog')
                try:
                    with self.assertRaises(error):
                        worker.exchange({'source':source, 'filename':'candidate.py', 'problem':{},
                                         'initial':{}, 'seed':1}, .2, 1, lambda:None, operation='initialize')
                finally: worker.close()

    def test_queued_on_time_reply_survives_slow_diagnostic_write(self):
        class SlowRecords(MemoryRecords):
            def event(self, *args, **kwargs): time.sleep(.3)
        worker = StrategyWorker(SlowRecords(), 'slow-log')
        try:
            reply = worker.exchange({'source':SOURCE, 'filename':'candidate.py', 'problem':{},
                                     'initial':{}, 'seed':1}, .2, 1, lambda:None, operation='initialize')
            self.assertEqual(reply['kind'], 'initialized')
        finally: worker.close()

if __name__ == '__main__': unittest.main()
