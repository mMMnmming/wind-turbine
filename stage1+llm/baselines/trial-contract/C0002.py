import math
import random
from candidate_tools import distance, directed_time, region, deployment, tick, remaining, phase, remember, best, split_repair, make_proposal, assignments_from_proposal, BudgetExceeded, ToolContractError

def solve(turbines, config, seed):
    rng = random.Random(seed)
    ids = [getattr(t, 'identifier') for t in turbines]
    coords = {getattr(t, 'identifier'): (float(getattr(t, 'x_km')), float(getattr(t, 'y_km'))) for t in turbines}
    radius = float(getattr(config, 'communication_radius_km', 7.0))
    n = len(ids)
    if n == 0:
        return {'nest_ids': [], 'assignments': {}}
    def dist(a, b):
        return distance(a, b)
    def build(order, mode):
        nests = []
        assign = {}
        for tid in order:
            if not nests:
                nests.append(tid)
                assign[tid] = tid
                continue
            best_n = None
            best_score = None
            for nid in nests:
                d = dist(tid, nid)
                if d > radius:
                    continue
                members = [x for x in ids if assign.get(x) == nid]
                members.append(tid)
                if mode == 0:
                    score = d
                elif mode == 1:
                    score = d + 0.01 * len(members)
                else:
                    score = d - 0.02 * len(members)
                if best_score is None or score < best_score:
                    best_score = score
                    best_n = nid
            if best_n is None:
                nests.append(tid)
                assign[tid] = tid
            else:
                assign[tid] = best_n
        return assign
    def rough(assign):
        try:
            p = make_proposal(assign)
            chk = deployment(p, proxy=False)
            if chk['feasible']:
                return (len(p['nest_ids']), 0.0, 0.0, p)
        except (BudgetExceeded, ToolContractError):
            raise
        except Exception:
            pass
        return None
    phase('constraint_repair')
    candidates = []
    base_orders = [list(ids), list(reversed(ids))]
    for k in range(6):
        o = list(ids)
        rng.shuffle(o)
        base_orders.append(o)
    for oi, order in enumerate(base_orders):
        for mode in range(3):
            try:
                tick()
            except BudgetExceeded:
                break
            a = build(order, mode)
            r = rough(a)
            if r is not None:
                candidates.append(r)
    if not candidates:
        a = {tid: ids[0] for tid in ids}
        a[ids[0]] = ids[0]
        try:
            p = split_repair(a)
            chk = deployment(p, proxy=True)
            if chk['feasible']:
                remember(p)
                return p
        except (BudgetExceeded, ToolContractError):
            raise
        except Exception:
            pass
        p = make_proposal(a)
        return p
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    chosen = candidates[:3]
    best_prop = None
    best_key = None
    for _, _, _, p in chosen:
        try:
            tick()
            chk = deployment(p, proxy=True)
        except BudgetExceeded:
            break
        if chk['feasible']:
            objs = chk['objectives']
            key = (objs[0], objs[1], objs[2])
            if best_key is None or key < best_key:
                best_key = key
                best_prop = p
                remember(p)
    if best_prop is None:
        best_prop = chosen[0][3]
        try:
            chk = deployment(best_prop, proxy=True)
            if chk['feasible']:
                remember(best_prop)
        except (BudgetExceeded, ToolContractError):
            raise
        except Exception:
            pass
    cur = assignments_from_proposal(best_prop)
    cur_nests = list(best_prop['nest_ids'])
    def eval_assign(a):
        try:
            p = make_proposal(a)
            chk = deployment(p, proxy=True)
            if chk['feasible']:
                o = chk['objectives']
                return (o[0], o[1], o[2]), p
        except (BudgetExceeded, ToolContractError):
            raise
        except Exception:
            pass
        return None, None
    cur_key = best_key
    improved = True
    rounds = 0
    while improved and rounds < 4:
        improved = False
        rounds += 1
        moves = []
        for tid in ids:
            if tid in cur_nests:
                continue
            for nid in cur_nests:
                if nid != cur[tid]:
                    moves.append((tid, nid))
        rng.shuffle(moves)
        for tid, nid in moves[:40]:
            try:
                tick()
            except BudgetExceeded:
                break
            a = dict(cur)
            a[tid] = nid
            k, p = eval_assign(a)
            if k is not None and (cur_key is None or k < cur_key):
                cur = a
                cur_key = k
                cur_nests = list(p['nest_ids'])
                remember(p)
                improved = True
                break
        if improved:
            continue
        for nid in list(cur_nests):
            members = [x for x in ids if cur[x] == nid]
            if len(members) <= 1:
                continue
            for tid in members:
                if tid == nid:
                    continue
                try:
                    tick()
                except BudgetExceeded:
                    break
                a = dict(cur)
                a[tid] = tid
                k, p = eval_assign(a)
                if k is not None and (cur_key is None or k < cur_key):
                    cur = a
                    cur_key = k
                    cur_nests = list(p['nest_ids'])
                    remember(p)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        if len(cur_nests) > 1:
            for nid in list(cur_nests):
                if nid == cur_nests[0]:
                    continue
                a = dict(cur)
                for tid in ids:
                    if a[tid] == nid:
                        a[tid] = cur_nests[0]
                k, p = eval_assign(a)
                if k is not None and (cur_key is None or k < cur_key):
                    cur = a
                    cur_key = k
                    cur_nests = list(p['nest_ids'])
                    remember(p)
                    improved = True
                    break
    b = best()
    if b is not None:
        return b
    return make_proposal(cur)
