"""Auditable multi-objective program evolution; never consults test scores."""
from __future__ import annotations

import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

from .artifacts import RunArtifacts, run_path, safe_child, redact_secrets
from .candidate import validate_candidate_source
from .client import DeepSeekClient, DeepSeekSettings
from .data import fingerprint, prepare_data, turbines_from_record
from .domain import result_as_dict
from .evaluator import evaluate_proposal
from .exports import write_front
from .reward import OBJECTIVE_VERSION, pareto_front, parent_pool, search_reward
from .sandbox import run_candidate
from .selection import DiscountedUCB
from .settings import SearchSettings

SEARCH_VERSION = 3
PROMPTS = Path(__file__).resolve().parents[1]/'prompts'
INITIAL_DIRECTIONS = ('structural_exploration','rebalance_assignments','reduce_nests')


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value,default=str))


class SearchController:
    def __init__(self, settings: SearchSettings, llm: DeepSeekSettings, outputs: Path,
                 client: DeepSeekClient | None = None):
        self.settings, self.llm, self.outputs = settings,llm,outputs
        self.client = client if client is not None else DeepSeekClient(llm)
        self.mock = not isinstance(self.client,DeepSeekClient)
        self.archive: list[dict] = []
        self.repair_pool: list[dict] = []
        self.history: list[dict] = []
        self.source_hashes: set[str] = set()
        self.behavior_hashes: set[str] = set()
        self.bandit = DiscountedUCB(list(settings.direction_actions),settings.ucb_decay,
                                   settings.ucb_exploration,settings.seed)
        self.rng = random.Random(settings.seed)
        self.stalled = 0
        self.pending: dict | None = None
        self.elapsed_seconds = 0.0

    def _identity(self, bundle: dict, system: str) -> dict:
        config = asdict(self.settings)
        for key in ('evaluation_budget','heartbeat_seconds','data_dir','target_input','validation_inputs','protected_input'):
            config.pop(key)
        implementation = {p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))}
        return _jsonable({
            'search_version':SEARCH_VERSION,'settings':config,
            'data_fingerprint':bundle['fingerprint'],'objective_version':OBJECTIVE_VERSION,
            'prompt_hash':hashlib.sha256(system.encode('utf-8')).hexdigest(),
            'implementation':implementation,'origin':'offline_mock' if self.mock else 'llm',
            'llm':{k:v for k,v in asdict(self.llm).items() if k != 'api_key'},
        })

    def _persist(self, artifacts: RunArtifacts) -> None:
        artifacts.write_json('search_state.json',{
            'search_version':SEARCH_VERSION,'identity':self.identity,
            'evaluation_budget':self.settings.evaluation_budget,'archive':self.archive,
            'repair_pool':self.repair_pool,'history':self.history,
            'source_hashes':sorted(self.source_hashes),'behavior_hashes':sorted(self.behavior_hashes),
            'ucb':self.bandit.state_dict(),'rng_state':self.rng.getstate(),
            'stalled':self.stalled,'pending':self.pending,
            'active_search_seconds':self.elapsed_seconds+time.perf_counter()-self.active_start,
        })

    def _restore(self, artifacts: RunArtifacts) -> None:
        state = json.loads((artifacts.root/'search_state.json').read_text(encoding='utf-8'))
        if state.get('search_version') != SEARCH_VERSION:
            raise ValueError('Legacy search version cannot resume v3; use a new run ID.')
        if state.get('identity') != self.identity:
            raise ValueError('Run identity/configuration changed; do not mix old data, constraints or objectives.')
        if self.settings.evaluation_budget < state['evaluation_budget']:
            raise ValueError('Resume may only increase the evaluation budget.')
        self.archive,self.repair_pool,self.history = state['archive'],state['repair_pool'],state['history']
        self.source_hashes,self.behavior_hashes = set(state['source_hashes']),set(state['behavior_hashes'])
        self.bandit.load_state_dict(state['ucb'])
        def tuples(value):
            return tuple(tuples(v) for v in value) if isinstance(value,list) else value
        self.rng.setstate(tuples(state['rng_state']))
        self.stalled,self.pending = state['stalled'],state['pending']
        self.elapsed_seconds = state['active_search_seconds']
        if self.pending and self.pending.get('stage') == 'source_ready':
            source = self.pending['algorithm_code']
            if hashlib.sha256(source.encode('utf-8')).hexdigest() != self.pending['source_sha256']:
                raise ValueError('Pending source hash changed in checkpoint.')
            artifacts.write_text(self.pending['source_path'],source)
            self.pending.pop('algorithm_code')
            self.pending['stage'] = 'evaluation'
            artifacts.log(f"{self.pending['candidate_id']} recovered saved LLM response without new API request",event='source_recovered')
        # Detect edits to archived or pending source before any new API request.
        for row in [*self.history,*([self.pending] if self.pending else [])]:
            if row.get('source_path') and row.get('source_sha256'):
                text = safe_child(artifacts.root,row['source_path']).read_text(encoding='utf-8')
                if hashlib.sha256(text.encode('utf-8')).hexdigest() != row['source_sha256']:
                    raise ValueError('Saved candidate source hash changed.')

    def _choose(self) -> tuple[dict | None,str,dict,str]:
        if len(self.history) < self.settings.initial_candidates:
            direction = INITIAL_DIRECTIONS[len(self.history)%3]
            return None,direction,{},'independent_initial'
        repair_due = not self.archive or (len(self.history)-self.settings.initial_candidates+1)%4 == 0
        if self.repair_pool and repair_due:
            parent = self.repair_pool[-1]
            text = ' '.join(parent['evaluation'].get('violations',[])).lower()
            diagnostic_patterns = [('coverage','coverage'),('self-assignment','coverage'),
                ('undeclared nest','undeclared_nest'),('refer to an input','invalid_nest'),
                ('endurance','endurance'),('service-radius','service_radius')]
            direction = next((self.settings.diagnostic_direction_map[kind]
                              for needle,kind in diagnostic_patterns
                              if needle in text and kind in self.settings.diagnostic_direction_map),'constraint_repair')
            direction,trace = self.bandit.select_with_trace(direction)
            return parent,direction,trace,'repair_failed_child'
        pool = parent_pool(self.archive,self.settings.archive_size)
        parent = self.rng.choice(pool) if pool else None
        direction,trace = self.bandit.select_with_trace()
        return parent,direction,trace,'diverse_pareto_parent' if parent else 'no_source_restart'

    def _prompt(self, parent: dict | None, direction: str, artifacts: RunArtifacts) -> str:
        feedback = {
            'direction_id':direction,'deployment_config':asdict(self.settings.deployment),
            'objective_definition':OBJECTIVE_VERSION,'dataset_count':len(self.bundle['datasets']),
            'parent_diagnostic':parent['evaluation'] if parent else None,
        }
        # No protected coordinates, test results, API credentials or raw whole data.
        text = json.dumps(feedback,ensure_ascii=False,default=str)
        if parent:
            # Compact diagnostics but retain every failed dataset's numeric messages.
            feedback['parent_diagnostic'] = {
                'objectives':parent.get('objectives'),
                'datasets':[{k:r.get(k) for k in ('name','feasible','metrics','violations')}
                            for r in parent['evaluation'].get('datasets',[])],
                'violations':parent['evaluation'].get('violations',[]),
            }
            text = json.dumps(feedback,ensure_ascii=False)
        source = (artifacts.root/parent['source_path']).read_text(encoding='utf-8') if parent else '无；从零生成。'
        return f'本轮方向必须为 {direction}。\n固定评测诊断：{text}\n父代完整源码：\n{source}'

    def _evaluate(self, artifacts: RunArtifacts, row: dict) -> dict:
        source_path = safe_child(artifacts.root,row['source_path'])
        source = source_path.read_text(encoding='utf-8')
        if hashlib.sha256(source.encode('utf-8')).hexdigest() != row['source_sha256']:
            raise ValueError('Candidate source changed before evaluation.')
        validate_candidate_source(source)
        if row['source_sha256'] in self.source_hashes:
            return {'status':'duplicate_source','feasible':False,'duplicate':True,'violations':[]}
        datasets = row.setdefault('completed_datasets',[])
        for record in self.bundle['datasets'][len(datasets):]:
            turbines = turbines_from_record(record)
            artifacts.log(f"{row['candidate_id']} dataset={record['name']} nodes={len(turbines)} start",event='dataset_start')
            start = time.perf_counter()
            with artifacts.heartbeat(f"{row['candidate_id']} dataset={record['name']} running",self.settings.heartbeat_seconds):
                proposal = run_candidate(source_path,turbines,self.settings.deployment,
                                         self.settings.seed,self.settings.candidate_timeout_seconds)
            algorithm_seconds = time.perf_counter()-start
            check_start = time.perf_counter()
            with artifacts.heartbeat(f"{row['candidate_id']} dataset={record['name']} evaluator",self.settings.heartbeat_seconds):
                checked = evaluate_proposal(turbines,self.settings.deployment,proposal)
            checked_seconds = time.perf_counter()-check_start
            data = {
                'name':record['name'],'split':record['split'],'turbine_count':len(turbines),
                'feasible':checked.feasible,'metrics':asdict(checked.metrics),
                'violations':list(checked.violations),'algorithm_seconds':algorithm_seconds,
                'evaluation_seconds':checked_seconds,
                'proposal':asdict(proposal),
                'recomputed_deployment':result_as_dict(checked.deployment) if checked.deployment else None,
            }
            datasets.append(data)
            self._persist(artifacts)
            artifacts.log(
                f"{row['candidate_id']} dataset={record['name']} feasible={checked.feasible} "
                f"nests={checked.metrics.nest_count if checked.feasible else 'N/A'} "
                f"proxy_flight={checked.metrics.total_proxy_flight_seconds:.3f}s "
                f"max_tasks={checked.metrics.maximum_region_tasks} "
                f"endurance_margin={checked.metrics.minimum_endurance_margin_seconds:.3f}s "
                f"radius_margin={checked.metrics.minimum_service_margin_km:.4f}km "
                f"algorithm={algorithm_seconds:.3f}s evaluator={checked_seconds:.3f}s "
                f"violations={'; '.join(checked.violations) or 'none'}",event='dataset_end')
        behavior = fingerprint([{'nest_ids':sorted(d['proposal']['nest_ids']),
                                 'assignments':d['proposal']['assignments']} for d in datasets])
        duplicate = behavior in self.behavior_hashes
        feasible = all(d['feasible'] for d in datasets)
        result = {'status':'feasible' if feasible else 'infeasible','feasible':feasible,
                  'datasets':datasets,'target':datasets[0],'behavior_sha256':behavior,
                  'duplicate':duplicate,'violations':[f"{d['name']}: {v}" for d in datasets for v in d['violations']]}
        if feasible:
            metrics = [d['metrics'] for d in datasets]
            objectives = [
                sum(d['metrics']['nest_count']/d['turbine_count'] for d in datasets)/len(datasets),
                sum(d['metrics']['total_proxy_flight_seconds']/d['turbine_count'] for d in datasets)/len(datasets),
                max(d['metrics']['maximum_region_tasks']/d['turbine_count'] for d in datasets),
            ]
            result['aggregate'] = {
                'mean_nest_count':sum(m['nest_count'] for m in metrics)/len(metrics),
                'mean_total_proxy_flight_seconds':sum(m['total_proxy_flight_seconds'] for m in metrics)/len(metrics),
                'maximum_region_tasks':max(m['maximum_region_tasks'] for m in metrics),
            }
            result['objectives'] = objectives
        return result

    def _finish(self, artifacts: RunArtifacts, row: dict, evaluation: dict) -> None:
        before = {tuple(r['objectives']) for r in self.archive}
        evaluation['source_sha256'] = row.get('source_sha256')
        row['evaluation'] = evaluation
        row['objectives'] = evaluation.get('objectives')
        parent = next((r for r in self.history if r['candidate_id'] == row['parent_id']),None)
        duplicate = bool(evaluation.get('duplicate'))
        if evaluation['feasible'] and not duplicate:
            self.archive = pareto_front([*self.archive,row])
        after = {tuple(r['objectives']) for r in self.archive}
        new_front = before != after
        row['reward'] = search_reward(evaluation['feasible'],row['objectives'],
            parent.get('objectives') if parent else None,duplicate=duplicate,new_front=new_front,
            violation_count=len(evaluation.get('violations',[])))
        if evaluation['feasible'] and not before and not duplicate:
            row['reward'] = 1.0
        self.bandit.update(row['direction_id'],row['reward'])
        if not evaluation['feasible'] and not duplicate and row.get('source_path'):
            self.repair_pool = [*self.repair_pool,row][-self.settings.archive_size:]
        row['decision'] = ('duplicate' if duplicate else 'front_member' if row in self.archive
                           else 'dominated' if evaluation['feasible'] else 'infeasible_or_error')
        if parent and row['objectives'] and parent.get('objectives'):
            row['objective_delta'] = [a-b for a,b in zip(row['objectives'],parent['objectives'])]
        if row.get('source_sha256'):
            self.source_hashes.add(row['source_sha256'])
        if evaluation.get('behavior_sha256'):
            self.behavior_hashes.add(evaluation['behavior_sha256'])
        row.pop('completed_datasets',None)
        self.history.append(row)
        self.pending = None
        self.stalled = 0 if new_front else self.stalled+1
        self._persist(artifacts)
        self._reports(artifacts)
        artifacts.log(
            f"{row['candidate_id']} reward={row['reward']:+.2f} decision={row['decision']} "
            f"objectives={row['objectives']} delta={row.get('objective_delta')} "
            f"front={len(self.archive)} stalled={self.stalled}/{self.settings.no_improvement_limit}",
            event='iteration_end',candidate_id=row['candidate_id'],reward=row['reward'],decision=row['decision'])

    def _reports(self, artifacts: RunArtifacts) -> None:
        write_front(artifacts.root,self.archive)
        for row in self.history:
            artifacts.write_json(f"evaluations/{row['index']:04d}.json",row)
        with (artifacts.root/'trajectory.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['candidate','parent','direction','feasible','reward','decision','objectives','llm_seconds','algorithm_seconds','evaluation_seconds'])
            for row in self.history:
                data = row['evaluation'].get('datasets',[])
                writer.writerow([row['candidate_id'],row['parent_id'],row['direction_id'],row['evaluation']['feasible'],
                    row['reward'],row['decision'],row['objectives'],row.get('llm_seconds',0),
                    sum(d['algorithm_seconds'] for d in data),sum(d['evaluation_seconds'] for d in data)])

    def run(self, run_id: str | None = None, resume: bool = False, reset_stall: bool = False) -> Path:
        self.bundle = prepare_data(self.settings)
        system = (PROMPTS/'system.md').read_text(encoding='utf-8')
        self.identity = self._identity(self.bundle,system)
        if not resume and run_id and run_path(self.outputs,run_id).exists():
            raise FileExistsError('Run directory exists; choose a new run ID or resume.')
        if resume and (not run_id or not (run_path(self.outputs,run_id)/'search_state.json').exists()):
            raise FileNotFoundError('No resumable search state.')
        artifacts = RunArtifacts(self.outputs,run_id,secrets=(self.llm.api_key,))
        self.active_start = time.perf_counter()
        if resume:
            self._restore(artifacts)
            if reset_stall:
                self.stalled = 0
        artifacts.write_json('config_snapshot.json',self.identity)
        if hasattr(self.client,'set_observer'):
            def observe(message,**fields):
                candidate_id = self.pending['candidate_id'] if self.pending else None
                artifacts.log(f'{candidate_id} {message}',event='api_request',candidate_id=candidate_id,**fields)
            self.client.set_observer(observe)
        artifacts.log(f"Search v3 origin={'offline_mock' if self.mock else 'llm'} datasets={len(self.bundle['datasets'])} "
                      f"completed={len(self.history)} budget={self.settings.evaluation_budget}; objective={OBJECTIVE_VERSION}",event='search_start')
        self._persist(artifacts)
        try:
            while len(self.history) < self.settings.evaluation_budget and self.stalled < self.settings.no_improvement_limit:
                if self.pending is None:
                    parent,direction,trace,reason = self._choose()
                    index = len(self.history)+1
                    self.pending = {
                        'index':index,'candidate_id':f'C{index:04d}','parent_id':parent['candidate_id'] if parent else None,
                        'direction_id':direction,'direction_selection':trace,'selection_reason':reason,
                        'origin':'offline_mock' if self.mock else 'llm','stage':'generation',
                    }
                    self._persist(artifacts)
                    row = self.pending
                    artifacts.log(f"round={index}/{self.settings.evaluation_budget} remaining={self.settings.evaluation_budget-index} "
                                  f"parent={row['parent_id']} direction={direction} reason={reason}",event='iteration_start',direction_selection=trace)
                    start = time.perf_counter()
                    try:
                        artifacts.log(f"{row['candidate_id']} LLM request start",event='llm_start')
                        with artifacts.heartbeat(f"{row['candidate_id']} waiting for LLM",self.settings.heartbeat_seconds):
                            generated = self.client.generate(system,self._prompt(parent,direction,artifacts),direction)
                        if redact_secrets(generated.algorithm_code,(self.llm.api_key,)) != generated.algorithm_code:
                            raise ValueError('Generated source contains sensitive-looking text; refusing unsafe checkpoint.')
                    except Exception as error:
                        row['llm_seconds'] = time.perf_counter()-start
                        row['api_metadata'] = getattr(self.client,'last_metadata',{'actual_requests':0,'note':'offline mock'})
                        artifacts.log(f"{row['candidate_id']} generation failed: {type(error).__name__}: {error}",event='llm_error')
                        self._finish(artifacts,row,{'status':'generation_error','feasible':False,'violations':[f'{type(error).__name__}: {error}']})
                        continue
                    row.update(asdict(generated))
                    row['source_path'] = f'candidates/{index:04d}.py'
                    row['source_sha256'] = hashlib.sha256(generated.algorithm_code.encode('utf-8')).hexdigest()
                    row['llm_seconds'] = time.perf_counter()-start
                    row['api_metadata'] = getattr(self.client,'last_metadata',{'actual_requests':0,'note':'offline mock'})
                    row['stage'] = 'source_ready'
                    # Save returned source before writing its standalone file.
                    # A killed writer can recover without paying for regeneration.
                    self._persist(artifacts)
                    artifacts.write_text(row['source_path'],generated.algorithm_code)
                    row.pop('algorithm_code')
                    row['stage'] = 'evaluation'
                    artifacts.log(f"{row['candidate_id']} LLM finished in {row['llm_seconds']:.3f}s",event='llm_end',api_metadata=row['api_metadata'])
                    self._persist(artifacts)
                row = self.pending
                if row['stage'] == 'generation':
                    # Request may have reached the provider; do not silently repeat it.
                    self._finish(artifacts,row,{'status':'generation_error','feasible':False,
                        'violations':['Interrupted generation: API outcome unknown; round consumed without replay.']})
                    continue
                try:
                    evaluation = self._evaluate(artifacts,row)
                except Exception as error:
                    evaluation = {'status':'candidate_error','feasible':False,'datasets':row.get('completed_datasets',[]),
                                  'violations':[f'{type(error).__name__}: {error}']}
                    artifacts.log(f"{row['candidate_id']} candidate error: {error}",event='candidate_error')
                self._finish(artifacts,row,evaluation)
        except (KeyboardInterrupt,MemoryError,OSError):
            self._persist(artifacts)
            artifacts.log('Interrupted; saved completed work and pending round. Use resume.',event='interrupted')
            raise
        self._reports(artifacts)
        status = 'best_within_budget' if self.archive else 'no_deployable_llm_candidate'
        reason = 'budget_exhausted' if len(self.history) >= self.settings.evaluation_budget else 'front_stalled'
        artifacts.write_json('summary.json',{
            'status':status,'stop_reason':reason,'evaluated_rounds':len(self.history),
            'pareto_candidates':[r['candidate_id'] for r in self.archive],
            'global_optimality_proven':False,'origin':'offline_mock' if self.mock else 'llm',
            'active_search_seconds':self.elapsed_seconds+time.perf_counter()-self.active_start,
            'selection':'User must explicitly select a Pareto candidate; no implicit best algorithm.',
        })
        self._persist(artifacts)
        artifacts.log(f'Search complete status={status} stop={reason} front={len(self.archive)} output={artifacts.root}',event='search_end')
        return artifacts.root
