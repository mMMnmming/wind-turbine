import math
import random
from collections import defaultdict
from candidate_tools import (
    BudgetExceeded, ToolContractError, distance, directed_time, region,
    deployment, tick, phase, remember, best, split_repair,
    repartition_repair, stats, make_proposal, assignments_from_proposal,
    better_metrics, remaining
)


def _get(obj, name):
    if isinstance(obj, dict):
        return obj[name]
    return getattr(obj, name)


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


def _radius(config):
    return float(_get(config, 'communication_radius_km'))


def _safe_remaining():
    try:
        r = remaining()
        if isinstance(r, dict):
            return r
    except (BudgetExceeded, ToolContractError):
        raise
    return {}


def _budget_low():
    r = _safe_remaining()
    sec = r.get('seconds', None)
    ev = r.get('region_evaluations', None)
    mv = r.get('move_attempts', None)
    if sec is not None and float(sec) < 3.0:
        return True
    if ev is not None and int(ev) < 20:
        return True
    if mv is not None and int(mv) < 20:
        return True
    return False


def _eval(assign):
    p = make_proposal(assign)
    c = deployment(p)
    return p, c


def _feasible(assign):
    try:
        p, c = _eval(assign)
    except (BudgetExceeded, ToolContractError):
        raise
    if c.get('feasible'):
        return p, c
    return None, c


def _repair(assign):
    try:
        p = repartition_repair(assign)
        c = deployment(p)
        if c.get('feasible'):
            return p, c
    except (BudgetExceeded, ToolContractError):
        raise
    try:
        p = split_repair(assign)
        c = deployment(p)
        if c.get('feasible'):
            return p, c
    except (BudgetExceeded, ToolContractError):
        raise
    return None, None


def _initial(turbines, config, rng):
    ids = _ids(turbines)
    xy = _xy(turbines)
    rad = _radius(config)
    assign = {}
    nests = []
    order = list(ids)
    rng.shuffle(order)
    for tid in order:
        placed = False
        best_n = None
        best_d = None
        for n in nests:
            d = _dist_xy(xy[tid], xy[n])
            if d <= rad and (best_d is None or d < best_d):
                best_d = d
                best_n = n
        if best_n is not None:
            assign[tid] = best_n
            placed = True
        if not placed:
            assign[tid] = tid
            nests.append(tid)
    p, c = _feasible(assign)
    if p is not None:
        return p, c
    p, c = _repair(assign)
    if p is not None:
        return p, c
    return None, None


def _members(assign):
    m = defaultdict(list)
    for t, n in assign.items():
        m[n].append(t)
    return m


def _nest_ok(n, members, xy, rad):
    if n not in members:
        return False
    for t in members:
        if _dist_xy(xy[n], xy[t]) > rad + 1e-9:
            return False
    return True


def _try_assign(assign, xy, rad, tid, target):
    if target not in xy:
        return None
    if _dist_xy(xy[tid], xy[target]) > rad + 1e-9:
        return None
    a = dict(assign)
    a[tid] = target
    return a


def _remove_nest(assign, nest, xy, rad, rng):
    a = dict(assign)
    members = [t for t, n in a.items() if n == nest]
    if not members:
        return None
    others = sorted(set(n for n in a.values() if n != nest))
    if not others:
        return None
    rng.shuffle(members)
    for t in members:
        cands = []
        for n in others:
            d = _dist_xy(xy[t], xy[n])
            if d <= rad + 1e-9:
                cands.append((d, n))
        if not cands:
            return None
        cands.sort()
        a[t] = cands[0][1]
    for t in list(a.keys()):
        if a[t] == nest:
            del a[t]
    return a


def _merge_two(assign, xy, rad, rng):
    m = _members(assign)
    nests = sorted(m.keys())
    if len(nests) < 2:
        return None
    rng.shuffle(nests)
    for i in range(len(nests)):
        for j in range(i + 1, len(nests)):
            n1 = nests[i]
            n2 = nests[j]
            a = dict(assign)
            ok = True
            for t in m[n1]:
                if _dist_xy(xy[t], xy[n2]) > rad + 1e-9:
                    ok = False
                    break
                a[t] = n2
            if ok:
                for t in list(a.keys()):
                    if a[t] == n1:
                        del a[t]
                return a
    return None


def _move_one(assign, xy, rad, rng):
    m = _members(assign)
    nests = sorted(m.keys())
    if len(nests) < 2:
        return None
    rng.shuffle(nests)
    for src in nests:
        for t in list(m[src]):
            for dst in nests:
                if dst == src:
                    continue
                if _dist_xy(xy[t], xy[dst]) <= rad + 1e-9:
                    a = dict(assign)
                    a[t] = dst
                    if len(m[src]) == 1:
                        del a[src]
                    return a
    return None


def _swap(assign, xy, rad, rng):
    m = _members(assign)
    nests = sorted(m.keys())
    if len(nests) < 2:
        return None
    rng.shuffle(nests)
    for i in range(len(nests)):
        for j in range(i + 1, len(nests)):
            n1 = nests[i]
            n2 = nests[j]
            for t1 in m[n1]:
                for t2 in m[n2]:
                    if _dist_xy(xy[t1], xy[n2]) <= rad + 1e-9 and _dist_xy(xy[t2], xy[n1]) <= rad + 1e-9:
                        a = dict(assign)
                        a[t1] = n2
                        a[t2] = n1
                        return a
    return None


def _accept(cand_c, inc_c):
    if not cand_c.get('feasible'):
        return False
    co = cand_c.get('objectives')
    io = inc_c.get('objectives')
    if co is None or io is None:
        return False
    return better_metrics(co, io)


def solve(turbines, config, seed):
    rng = random.Random(seed)
    ids = _ids(turbines)
    if not ids:
        return {'nest_ids': [], 'assignments': {}}
    xy = _xy(turbines)
    rad = _radius(config)
    p, c = _initial(turbines, config, rng)
    if p is None:
        assign = {t: t for t in ids}
        p, c = _repair(assign)
    if p is None:
        return {'nest_ids': [], 'assignments': {}}
    remember(p)
    inc_p = p
    inc_c = c
    inc_assign = assignments_from_proposal(p)
    tick(1)
    phase('improve')
    max_iter = 400
    for it in range(max_iter):
        if _budget_low():
            break
        tick(1)
        cur_assign = inc_assign
        cand = None
        mode = it % 4
        if mode == 0:
            m = _members(cur_assign)
            nests = sorted(m.keys())
            if nests:
                rng.shuffle(nests)
                for n in nests:
                    a = _remove_nest(cur_assign, n, xy, rad, rng)
                    if a is not None:
                        cand = a
                        break
        elif mode == 1:
            cand = _merge_two(cur_assign, xy, rad, rng)
        elif mode == 2:
            cand = _move_one(cur_assign, xy, rad, rng)
        else:
            cand = _swap(cur_assign, xy, rad, rng)
        if cand is None:
            continue
        cp, cc = _feasible(cand)
        if cp is None:
            cp, cc = _repair(cand)
        if cp is None or cc is None:
            continue
        if _accept(cc, inc_c):
            inc_p = cp
            inc_c = cc
            inc_assign = assignments_from_proposal(cp)
            remember(cp)
    bp = best()
    if bp is not None:
        bc = deployment(bp)
        if bc.get('feasible'):
            ba = assignments_from_proposal(bp)
            return {'nest_ids': sorted(set(ba.values())), 'assignments': ba}
    ba = assignments_from_proposal(inc_p)
    return {'nest_ids': sorted(set(ba.values())), 'assignments': ba}
