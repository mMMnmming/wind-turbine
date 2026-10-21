"""CLI for data preparation, program search and explicit deployment selection."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import sys
import time
from pathlib import Path

from .artifacts import RunArtifacts, sha256_file, validate_run_id, run_path
from .data import prepare_data
from .domain import load_turbines
from .evaluator import evaluate_proposal
from .exports import export_candidate, export_evaluation, list_candidates, candidate_origin
from .plotting import write_deployment_figure
from .sandbox import run_candidate
from .search import SearchController
from .settings import load_deepseek_settings, load_search_settings

LOCAL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = LOCAL_ROOT/'outputs'
DEFAULT_SEARCH = LOCAL_ROOT/'config'/'search.json'
DEFAULT_DEEPSEEK = LOCAL_ROOT/'config'/'deepseek.local.json'


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Independent multi-objective UAV deployment program evolution.')
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ('prepare-data','optimize','resume','evaluate'):
        command = sub.add_parser(name)
        command.add_argument('--search-config',type=Path,default=DEFAULT_SEARCH)
        command.add_argument('--outputs',type=Path,default=DEFAULT_OUTPUTS)
        if name != 'prepare-data':
            command.add_argument('--run-id',type=validate_run_id)
    for name in ('optimize','resume'):
        command = sub.choices[name]
        command.add_argument('--deepseek-config',type=Path,default=DEFAULT_DEEPSEEK)
        command.add_argument('--validation-input',action='append',default=[])
        command.add_argument('--evaluation-budget',type=int)
        command.add_argument('--no-improvement-limit',type=int)
        command.add_argument('--heartbeat-seconds',type=float)
        command.add_argument('--show-code',action=argparse.BooleanOptionalAction,default=None,
                             help='Print returned full algorithm source (default enabled); does not stream tokens.')
    sub.choices['resume'].add_argument('--resume-run-id',type=validate_run_id,required=True)
    sub.choices['resume'].add_argument('--reset-stall',action='store_true')
    evaluate = sub.choices['evaluate']
    evaluate.add_argument('--candidate',type=Path,required=True)
    evaluate.add_argument('--input',type=Path,required=True)
    evaluate.add_argument('--timeout',type=int,default=300)
    for name in ('list-candidates','export-best'):
        command = sub.add_parser(name)
        command.add_argument('--run-id',type=validate_run_id,required=True)
        command.add_argument('--outputs',type=Path,default=DEFAULT_OUTPUTS)
    sub.choices['export-best'].add_argument('--candidate-id',required=True)
    sub.choices['export-best'].add_argument('--reason',default='User selected from frozen Pareto table.')
    smoke = sub.add_parser('smoke-test')
    smoke.add_argument('--run-id',type=validate_run_id,required=True)
    smoke.add_argument('--outputs',type=Path,default=DEFAULT_OUTPUTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == 'smoke-test':
            from .smoke import smoke_test
            smoke_test(args.outputs,args.run_id)
            return 0
        if args.command == 'list-candidates':
            list_candidates(run_path(args.outputs,args.run_id))
            return 0
        if args.command == 'export-best':
            path = export_candidate(run_path(args.outputs,args.run_id),args.candidate_id,reason=args.reason)
            print(f'Selected algorithm: {path}',flush=True)
            return 0
        settings = load_search_settings(args.search_config,getattr(args,'validation_input',[]))
        if args.command == 'prepare-data':
            bundle = prepare_data(settings)
            print(f"Frozen datasets={len(bundle['datasets'])} fingerprint={bundle['fingerprint']} path={settings.data_dir}",flush=True)
            return 0
        if args.command in ('optimize','resume'):
            changes = {key:getattr(args,key) for key in ('evaluation_budget','no_improvement_limit','heartbeat_seconds','show_code')
                       if getattr(args,key) is not None}
            settings = replace(settings,**changes)
            llm = load_deepseek_settings(args.deepseek_config)
            controller = SearchController(settings,llm,args.outputs)
            try:
                controller.run(args.resume_run_id if args.command == 'resume' else args.run_id,
                               resume=args.command == 'resume',reset_stall=getattr(args,'reset_stall',False))
            finally:
                controller.client.close()
            return 0
        if args.command == 'evaluate':
            candidate = args.candidate.resolve()
            if args.run_id and run_path(args.outputs,args.run_id).exists():
                raise FileExistsError('Evaluation run exists; choose a new run ID.')
            artifacts = RunArtifacts(args.outputs,args.run_id)
            turbines = load_turbines(args.input.resolve())
            source_text = candidate.read_text(encoding='utf-8')
            provenance = candidate_origin(candidate,source_text)
            artifacts.write_text('candidates/evaluated.py',source_text)
            snapshot = artifacts.root/'candidates/evaluated.py'
            source_hash = sha256_file(snapshot)
            artifacts.log(f'Evaluate fixed algorithm {candidate} input={args.input} nodes={len(turbines)}',event='evaluation_start')
            start = time.perf_counter()
            with artifacts.heartbeat('Deployment algorithm running',settings.heartbeat_seconds):
                proposal = run_candidate(snapshot,turbines,settings.deployment,settings.seed,args.timeout)
            algorithm_seconds = time.perf_counter()-start
            start = time.perf_counter()
            with artifacts.heartbeat('Trusted evaluator running',settings.heartbeat_seconds):
                checked = evaluate_proposal(turbines,settings.deployment,proposal)
            evaluation_seconds = time.perf_counter()-start
            source = {'kind':'stage1_llm_program_evaluation',**provenance,'algorithm_sha256':source_hash,
                      'turbines_sha256':sha256_file(args.input.resolve()),'turbine_count':len(turbines)}
            artifacts.write_json('evaluation.json',{
                'feasible':checked.feasible,'metrics':asdict(checked.metrics),'violations':checked.violations,
                'algorithm_seconds':algorithm_seconds,'evaluation_seconds':evaluation_seconds,'source':source,
            })
            if checked.feasible:
                export_evaluation(artifacts.root,turbines,checked,source)
                write_deployment_figure(artifacts.root/'deployment.png',turbines,checked.deployment)
                artifacts.log(f'PASS nests={checked.metrics.nest_count} proxy_flight={checked.metrics.total_proxy_flight_seconds:.3f}s '
                              f'max_tasks={checked.metrics.maximum_region_tasks} algorithm={algorithm_seconds:.3f}s '
                              f'evaluator={evaluation_seconds:.3f}s output={artifacts.root}',event='evaluation_end')
                return 0
            artifacts.log(f'FAIL {checked.violations}',event='evaluation_end')
            return 2
    except KeyboardInterrupt:
        print('Interrupted; completed search work was checkpointed. Use resume.',file=sys.stderr,flush=True)
        return 130
    except Exception as error:
        # Never display credentials possibly present in provider exception strings.
        from .artifacts import redact_secrets
        secrets = (llm.api_key,) if 'llm' in locals() else ()
        print(redact_secrets(f'error: {type(error).__name__}: {error}',secrets),file=sys.stderr,flush=True)
        return 1
    return 1
