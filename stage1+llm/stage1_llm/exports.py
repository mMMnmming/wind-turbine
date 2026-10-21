"""Manual selection and read-compatible, complete geometry exports."""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path

from .artifacts import atomic_json, safe_child
from .domain import Turbine, result_as_dict
from .evaluator import Evaluation


def load_state(root: Path) -> dict:
    state = json.loads((root/'search_state.json').read_text(encoding='utf-8'))
    if state.get('search_version') != 3:
        raise ValueError('Expected v3 search; legacy results remain readable but cannot be selected as v3.')
    return state


def list_candidates(root: Path) -> list[dict]:
    rows = load_state(root)['archive']
    print('候选 | 平均机巢数 | 平均总飞行时间/s(启发式) | 最大区域任务数 | 三目标',flush=True)
    for row in rows:
        summary = row['evaluation']['aggregate']
        print(f"{row['candidate_id']} | {summary['mean_nest_count']:.3f} | "
              f"{summary['mean_total_proxy_flight_seconds']:.3f} | {summary['maximum_region_tasks']} | "
              f"{row['objectives']}",flush=True)
    return rows


def candidate_origin(candidate: Path, source: str) -> dict:
    """Do not relabel a mock or arbitrary external program as a formal result."""
    actual = hashlib.sha256(source.encode('utf-8')).hexdigest()
    selection = candidate.parent/'selection.json'
    if selection.is_file():
        record = json.loads(selection.read_text(encoding='utf-8'))
        if record['source_sha256'] != actual:
            raise ValueError('Selected algorithm source hash changed.')
        return {'origin':record['origin'],'selected':True,'candidate_id':record['candidate_id']}
    state_path = candidate.parent.parent/'search_state.json'
    if candidate.parent.name == 'candidates' and state_path.is_file():
        state = load_state(candidate.parent.parent)
        for row in state['history']:
            if row.get('source_sha256') == actual:
                return {'origin':row['origin'],'selected':False,'candidate_id':row['candidate_id']}
    return {'origin':'unverified_external','selected':False}


def export_candidate(root: Path, candidate_id: str, *, allow_mock: bool = False,
                     reason: str = 'User selected from frozen Pareto table.') -> Path:
    state = load_state(root)
    selected = next((row for row in state['archive'] if row['candidate_id'] == candidate_id),None)
    if selected is None or not selected['evaluation']['feasible']:
        raise ValueError('Candidate must be a feasible member of the saved Pareto front.')
    if selected['origin'] == 'offline_mock' and not allow_mock:
        raise ValueError('Cannot export offline mock as a formal LLM algorithm.')
    source = safe_child(root,selected['source_path']).read_text(encoding='utf-8')
    if hashlib.sha256(source.encode('utf-8')).hexdigest() != selected['evaluation']['source_sha256']:
        raise ValueError('Candidate source hash changed since evaluation.')
    destination = safe_child(root,f'selected/{candidate_id}')
    destination.mkdir(parents=True,exist_ok=False)
    (destination/'algorithm.py').write_text(source,encoding='utf-8')
    atomic_json(destination/'selection.json',{
        'candidate_id':candidate_id,'reason':reason,'origin':selected['origin'],
        'source_sha256':selected['evaluation']['source_sha256'],
        'objectives':selected['objectives'],'run_identity':state['identity'],
        'reference_note':'No 87 score was read by this selection operation.',
    })
    atomic_json(destination/'evaluation.json',selected['evaluation'])
    return destination


def export_evaluation(output: Path, turbines: list[Turbine], evaluation: Evaluation, source: dict) -> None:
    if not evaluation.feasible or evaluation.deployment is None:
        raise ValueError('Cannot export an infeasible deployment as a completed region list.')
    if (output/'summary.json').exists() or (output/'regions.json').exists():
        raise FileExistsError('Deployment export already exists; use another run ID.')
    output.mkdir(parents=True,exist_ok=True)
    deployment = evaluation.deployment
    by_id = {t.identifier:t for t in turbines}
    environment = {key:value for key,value in asdict(deployment.config).items()
                   if key not in ('communication_radius_km','coordinate_unit')}
    environment['gust_speed_mps'] = 12.0
    regions = []
    signatures = []
    for nest in deployment.nests:
        identifiers = [nest.identifier,*(key for key in nest.turbine_ids if key != nest.identifier)]
        record = {
            'identifiers':identifiers,
            'coordinates_km':[[float(by_id[key].x_km),float(by_id[key].y_km)] for key in identifiers],
            'depot_index':0,'node_count':len(identifiers),'config':environment,
        }
        # Exact encoding used by current Stage2's instance_fingerprint.
        value = {key:v for key,v in record.items() if key != 'identifiers'}
        signatures.append(hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode('utf-8')).hexdigest())
        regions.append(record)
    atomic_json(output/'regions.json',{
        'schema_version':1,'split':'test','frozen_reference':True,'source':source,
        'regions':regions,'instance_fingerprints':signatures,
        'split_fingerprint':hashlib.sha256('\n'.join(signatures).encode()).hexdigest(),
        'note':'Fixed 12 m/s gust feature; proxy routes are heuristic, not proven optimal.',
    })
    summary = result_as_dict(deployment)
    summary['source'] = source
    summary['metric_notes'] = {
        'conservative_energy':'17 m/s audit, hover includes nest turbine.',
        'proxy_energy':'Directed wind times; hover excludes depot. No optimality proof.',
    }
    atomic_json(output/'summary.json',summary)


def write_front(root: Path, rows: list[dict]) -> None:
    atomic_json(root/'pareto.json',{'solutions':rows,'optimality':'Not proven globally optimal.'})
    # Tables are rebuilt from the atomic search state after resume.
    with (root/'pareto.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['候选','平均机巢数','平均总飞行时间/s(启发式)','最大区域任务数','机巢比例','单位风机飞行时间/s','最坏任务比例'])
        for row in rows:
            aggregate = row['evaluation']['aggregate']
            writer.writerow([row['candidate_id'],aggregate['mean_nest_count'],
                             aggregate['mean_total_proxy_flight_seconds'],aggregate['maximum_region_tasks'],
                             *row['objectives']])
