import math
import random
from collections import defaultdict
from candidate_tools import (
    BudgetExceeded, ToolContractError, distance, directed_time, region,
    deployment, tick, remaining, phase, remember, best, split_repair,
    repartition_repair, stats, make_proposal, assignments_from_proposal,
    better_metrics,
)


def _get(obj, name):
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


def _ids(turbines):
    out = []
    for t in turbines:
        out.append(_get(t, 'identifier'))
    return out


def _xy(turbines):
    d = {}
    for t in turbines:
        d[_get(t, 'identifier')] = (float(_get(t, 'x_km')), float(_get(t, 'y_km')))
    return d


def _dist_xy(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _radius(config):
    return float(_get(config, 'communication_radius_km'))


def _valid_nest(nid, xy, ids, radius):
    if nid not in xy:
        return False
    p = xy[nid]
    for tid in ids:
        if _dist_xy(p, xy[tid]) > radius + 1e-9:
            return False
    return True


def _candidate_nests(ids, xy, radius):
    out = []
    for nid in ids:
        if _valid_nest(nid, xy, ids, radius):
            out.append(nid)
    return out


def _nearest_nest(tid, nests, xy):
    bestn = None
    bestd = None
    for nid in nests:
        d = _dist_xy(xy[tid], xy[nid])
        if bestd is None or d < bestd or (d == bestd and nid < bestn):
            bestd = d
            bestn = nid
    return bestn


def _assign_all(ids, nests, xy):
    a = {}
    for tid in ids:
        a[tid] = _nearest_nest(tid, nests, xy)
    return a


def _ensure_self(a, nests):
    for nid in nests:
        a[nid] = nid
    return a


def _try_proposal(a):
    try:
        p = make_proposal(a)
        c = deployment(p)
        return p, c
    except (BudgetExceeded, ToolContractError):
        raise


def _repair(a):
    try:
        p = repartition_repair(a)
        c = deployment(p)
        if c['feasible']:
            return p, c
    except (BudgetExceeded, ToolContractError):
        raise
    try:
        p = split_repair(a)
        c = deployment(p)
        if c['feasible']:
            return p, c
    except (BudgetExceeded, ToolContractError):
        raise
    return None, None


def _obj(c):
    if c is None or not c.get('feasible'):
        return None
    return c.get('objectives')


def _better(c, inc):
    o = _obj(c)
    io = _obj(inc)
    if o is None:
        return False
    if io is None:
        return True
    return better_metrics(o, io)


def _nests_of(a):
    s = set()
    for v in a.values():
        s.add(v)
    return s


def _members(a):
    m = defaultdict(list)
    for k, v in a.items():
        m[v].append(k)
    return m


def _budget_ok():
    try:
        r = remaining()
        sec = r.get('seconds', 0.0)
        ev = r.get('region_evaluations', 0)
        mv = r.get('move_attempts', 0)
        if sec is not None and sec <= 0.5:
            return False
        if ev is not None and ev <= 0:
            return False
        if mv is not None and mv <= 0:
            return False
        return True
    except (BudgetExceeded, ToolContractError):
        raise


def _try_assign(a, inc, rng, ids, xy, radius, nests_pool):
    p, c = _try_proposal(a)
    if c is not None and c.get('feasible') and _better(c, inc):
        return p, c
    return None, None


def _remove_nest(a, nid, ids, xy, nests_pool):
    b = dict(a)
    others = [x for x in _nests_of(b) if x != nid]
    if not others:
        return None
    for tid in list(b.keys()):
        if b[tid] == nid:
            b[tid] = _nearest_nest(tid, others, xy)
    b.pop(nid, None)
    return b


def _move_nest(a, old, new, xy):
    b = dict(a)
    for tid in list(b.keys()):
        if b[tid] == old:
            b[tid] = new
    b.pop(old, None)
    b[new] = new
    return b


def _swap_members(a, n1, n2, xy):
    b = dict(a)
    m1 = [k for k, v in b.items() if v == n1 and k != n1]
    m2 = [k for k, v in b.items() if v == n2 and k != n2]
    if not m1 or not m2:
        return None
    changed = False
    for t1 in m1:
        for t2 in m2:
            if _dist_xy(xy[t1], xy[n2]) < _dist_xy(xy[t1], xy[n1]) and _dist_xy(xy[t2], xy[n1]) < _dist_xy(xy[t2], xy[n2]):
                b[t1] = n2
                b[t2] = n1
                changed = True
                break
        if changed:
            break
    if not changed:
        return None
    return b


def solve(turbines, config, seed):
    rng = random.Random(seed)
    ids = _ids(turbines)
    xy = _xy(turbines)
    radius = _radius(config)
    if not ids:
        return {'nest_ids': [], 'assignments': {}}
    pool = _candidate_nests(ids, xy, radius)
    if not pool:
        pool = list(ids)
    phase('init')
    a = _assign_all(ids, pool, xy)
    a = _ensure_self(a, pool)
    p, c = _repair(a)
    if p is None:
        a = _assign_all(ids, ids, xy)
        a = _ensure_self(a, ids)
        p, c = _repair(a)
    if p is None:
        return {'nest_ids': [], 'assignments': {}}
    remember(p)
    inc = c
    cur = assignments_from_proposal(p)
    phase('improve')
    tick(1)
    improved = True
    rounds = 0
    while improved and rounds < 40 and _budget_ok():
        improved = False
        rounds += 1
        nests = sorted(_nests_of(cur))
        for nid in list(nests):
            if not _budget_ok():
                break
            b = _remove_nest(cur, nid, ids, xy, pool)
            if b is None:
                continue
            b = _ensure_self(b, _nests_of(b))
            p2, c2 = _try_proposal(b)
            if c2 is not None and c2.get('feasible') and _better(c2, inc):
                remember(p2)
                inc = c2
                cur = assignments_from_proposal(p2)
                improved = True
                break
        if improved:
            continue
        nests = sorted(_nests_of(cur))
        for i in range(len(nests)):
            if not _budget_ok():
                break
            for j in range(i + 1, len(nests)):
                if not _budget_ok():
                    break
                b = _swap_members(cur, nests[i], nests[j], xy)
                if b is None:
                    continue
                b = _ensure_self(b, _nests_of(b))
                p2, c2 = _try_proposal(b)
                if c2 is not None and c2.get('feasible') and _better(c2, inc):
                    remember(p2)
                    inc = c2
                    cur = assignments_from_proposal(p2)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        nests = sorted(_nests_of(cur))
        for old in nests:
            if not _budget_ok():
                break
            for new in pool:
                if new == old or new in nests:
                    continue
                if not _budget_ok():
                    break
                b = _move_nest(cur, old, new, xy)
                b = _ensure_self(b, _nests_of(b))
                p2, c2 = _try_proposal(b)
                if c2 is not None and c2.get('feasible') and _better(c2, inc):
                    remember(p2)
                    inc = c2
                    cur = assignments_from_proposal(p2)
                    improved = True
                    break
            if improved:
                break
    bp = best()
    if bp is None:
        return {'nest_ids': [], 'assignments': {}}
    ba = assignments_from_proposal(bp)
    bn = sorted(_nests_of(ba))
    return {'nest_ids': bn, 'assignments': ba}
