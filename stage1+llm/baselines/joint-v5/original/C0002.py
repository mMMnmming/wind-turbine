import math
import random
from collections import defaultdict
from candidate_tools import (
    BudgetExceeded, ToolContractError, distance, directed_time, region,
    deployment, tick, remaining, phase, remember, best, split_repair,
    repartition_repair, stats, make_proposal, assignments_from_proposal,
    better_metrics
)


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _ids(turbines):
    out = []
    for t in turbines:
        out.append(str(_get(t, 'identifier')))
    return out


def _xy(turbines):
    d = {}
    for t in turbines:
        d[str(_get(t, 'identifier'))] = (float(_get(t, 'x_km')), float(_get(t, 'y_km')))
    return d


def _dist_xy(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _safe_region(nid, members):
    try:
        return region(nid, members)
    except (BudgetExceeded, ToolContractError):
        raise


def _safe_deploy(proposal):
    try:
        return deployment(proposal)
    except (BudgetExceeded, ToolContractError):
        raise


def _feasible_checked(assign):
    p = make_proposal(assign)
    c = _safe_deploy(p)
    return p, c


def _repair(assign):
    p = repartition_repair(assign)
    c = _safe_deploy(p)
    if c.get('feasible'):
        return assignments_from_proposal(p), c
    p2 = split_repair(assign)
    c2 = _safe_deploy(p2)
    if c2.get('feasible'):
        return assignments_from_proposal(p2), c2
    return assign, c2


def _initial(turbines, radius, rng):
    ids = _ids(turbines)
    xy = _xy(turbines)
    assign = {}
    nests = []
    order = list(ids)
    rng.shuffle(order)
    for tid in order:
        chosen = None
        bestd = None
        for nid in nests:
            d = _dist_xy(xy[tid], xy[nid])
            if d <= radius and (bestd is None or d < bestd):
                bestd = d
                chosen = nid
        if chosen is None:
            chosen = tid
            nests.append(tid)
        assign[tid] = chosen
    for nid in nests:
        assign[nid] = nid
    assign, c = _repair(assign)
    return assign, c


def _nests(assign):
    return sorted(set(assign.values()))


def _remove_nest(assign, nid, radius):
    ids = list(assign.keys())
    others = [n for n in _nests(assign) if n != nid]
    if not others:
        return None, None
    cand = dict(assign)
    members = [t for t in ids if assign[t] == nid]
    for t in members:
        if t == nid:
            continue
        bestn = None
        bestscore = None
        for n in others:
            d = distance(t, n)
            if d <= radius:
                score = d
                if bestscore is None or score < bestscore:
                    bestscore = score
                    bestn = n
        if bestn is None:
            return None, None
        cand[t] = bestn
    if nid in cand:
        del cand[nid]
    cand, cc = _repair(cand)
    if cc.get('feasible'):
        return cand, cc
    return None, None


def _move_nest(assign, nid, rng):
    ids = list(assign.keys())
    members = [t for t in ids if assign[t] == nid]
    if not members:
        return None, None
    target = rng.choice(members)
    if target == nid:
        return None, None
    cand = dict(assign)
    cand[target] = target
    for t in members:
        if t != nid and t != target:
            cand[t] = target
    if nid in cand and nid != target:
        del cand[nid]
    cand, cc = _repair(cand)
    if cc.get('feasible'):
        return cand, cc
    return None, None


def _merge_nests(assign, rng):
    nests = _nests(assign)
    if len(nests) < 2:
        return None, None
    a = rng.choice(nests)
    b = rng.choice([n for n in nests if n != a])
    cand = dict(assign)
    for t, n in list(cand.items()):
        if n == b:
            cand[t] = a
    if b in cand:
        del cand[b]
    cand, cc = _repair(cand)
    if cc.get('feasible'):
        return cand, cc
    return None, None


def _swap_tasks(assign, rng):
    ids = list(assign.keys())
    if len(ids) < 2:
        return None, None
    a = rng.choice(ids)
    b = rng.choice(ids)
    if a == b or assign[a] == assign[b]:
        return None, None
    cand = dict(assign)
    cand[a], cand[b] = cand[b], cand[a]
    cand, cc = _repair(cand)
    if cc.get('feasible'):
        return cand, cc
    return None, None


def solve(turbines, config, seed):
    rng = random.Random(seed)
    radius = float(_get(config, 'communication_radius_km', 7.0))
    phase('init')
    assign, checked = _initial(turbines, radius, rng)
    if not checked.get('feasible'):
        return {'nest_ids': [], 'assignments': {}}
    remember(make_proposal(assign))

    current_assign = assign
    current_checked = checked
    best_assign = assign
    best_checked = checked

    phase('improve')
    max_rounds = 400
    for r in range(max_rounds):
        rem = remaining()
        if rem.get('seconds', 0) <= 0.5:
            break
        if rem.get('region_evaluations', 0) <= 0:
            break
        if rem.get('move_attempts', 0) <= 0:
            break
        tick(1)
        op = r % 4
        cand = None
        cc = None
        if op == 0:
            nests = _nests(current_assign)
            if nests:
                nid = rng.choice(nests)
                cand, cc = _remove_nest(current_assign, nid, radius)
        elif op == 1:
            nests = _nests(current_assign)
            if nests:
                nid = rng.choice(nests)
                cand, cc = _move_nest(current_assign, nid, rng)
        elif op == 2:
            cand, cc = _merge_nests(current_assign, rng)
        else:
            cand, cc = _swap_tasks(current_assign, rng)
        if cand is not None and cc is not None and cc.get('feasible'):
            if better_metrics(cc.get('objectives'), current_checked.get('objectives')):
                current_assign = cand
                current_checked = cc
                if better_metrics(cc.get('objectives'), best_checked.get('objectives')):
                    best_assign = cand
                    best_checked = cc
                    remember(make_proposal(best_assign))

    phase('final')
    p = make_proposal(best_assign)
    c = _safe_deploy(p)
    if c.get('feasible'):
        remember(p)
        return {'nest_ids': sorted(set(best_assign.values())), 'assignments': dict(best_assign)}
    b = best()
    if b is not None:
        a = assignments_from_proposal(b)
        return {'nest_ids': sorted(set(a.values())), 'assignments': dict(a)}
    return {'nest_ids': sorted(set(assign.values())), 'assignments': dict(assign)}