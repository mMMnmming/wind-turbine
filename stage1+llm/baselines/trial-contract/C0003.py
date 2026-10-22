import math
import random
from candidate_tools import distance, directed_time, region, deployment, tick, remaining, phase, remember, best, split_repair, make_proposal, assignments_from_proposal, BudgetExceeded, ToolContractError

def solve(turbines, config, seed):
    rng = random.Random(seed)
    radius = float(getattr(config, 'communication_radius_km', 7.0))
    speed = float(getattr(config, 'max_horizontal_speed_mps', 23.0))
    wind = float(getattr(config, 'wind_speed_mps', 6.0))
    endurance = float(getattr(config, 'max_endurance_seconds', 2460.0))
    hover = float(getattr(config, 'hover_seconds_per_turbine', 10.0))
    ratio = float(getattr(config, 'hover_to_flight_energy_ratio', 1.14))

    ids = []
    pos = {}
    for t in turbines:
        tid = getattr(t, 'identifier')
        ids.append(tid)
        pos[tid] = (float(getattr(t, 'x_km')), float(getattr(t, 'y_km')))
    n = len(ids)
    if n == 0:
        return {'nest_ids': [], 'assignments': {}}

    def dist_id(a, b):
        ax, ay = pos[a]
        bx, by = pos[b]
        return math.hypot(ax - bx, ay - by)

    def nearest_nest(tid, nests):
        bestd = None
        bestn = None
        for nid in nests:
            d = dist_id(tid, nid)
            if bestd is None or d < bestd or (d == bestd and nid < bestn):
                bestd = d
                bestn = nid
        return bestn

    def build_from_nests(nests):
        nests = sorted(set(nests))
        asg = {}
        for tid in ids:
            asg[tid] = nearest_nest(tid, nests)
        for nid in nests:
            asg[nid] = nid
        return asg

    def region_members(asg):
        groups = {}
        for tid in ids:
            nid = asg[tid]
            groups.setdefault(nid, []).append(tid)
        return groups

    def coarse_ok(asg):
        groups = region_members(asg)
        for nid, mem in groups.items():
            if nid not in mem:
                return False
            try:
                r = region(nid, list(mem), proxy=False)
            except ToolContractError:
                raise
            if not r['feasible']:
                return False
        return True

    def score_proxy(asg):
        p = make_proposal(asg)
        try:
            d = deployment(p, proxy=True)
        except ToolContractError:
            raise
        if not d['feasible']:
            return None
        o = d['objectives']
        return (float(o[0]), float(o[1]), float(o[2]))

    def better(a, b):
        if b is None:
            return True
        if a is None:
            return False
        eps = 1e-12
        if a[0] < b[0] - eps:
            return True
        if a[0] > b[0] + eps:
            return False
        if a[1] < b[1] - eps:
            return True
        if a[1] > b[1] + eps:
            return False
        return a[2] < b[2] - eps

    def local_metrics(asg):
        groups = region_members(asg)
        info = {}
        for nid, mem in groups.items():
            try:
                r = region(nid, list(mem), proxy=False)
            except ToolContractError:
                raise
            info[nid] = r
        return groups, info

    def try_accept(cur_asg, cur_score, cand_asg):
        if not coarse_ok(cand_asg):
            return cur_asg, cur_score, False
        sc = score_proxy(cand_asg)
        if sc is not None and better(sc, cur_score):
            return cand_asg, sc, True
        return cur_asg, cur_score, False

    phase('initial')
    starts = []
    starts.append(build_from_nests([ids[0]]))
    starts.append(build_from_nests([ids[0], ids[n // 2]]))
    starts.append(build_from_nests([ids[0], ids[n // 3], ids[(2 * n) // 3]]))
    for _ in range(3):
        k = max(1, min(n, 1 + rng.randrange(max(1, min(6, n)))))
        chosen = [ids[rng.randrange(n)] for _ in range(k)]
        starts.append(build_from_nests(chosen))

    best_asg = None
    best_score = None
    for asg in starts:
        tick()
        if not coarse_ok(asg):
            continue
        sc = score_proxy(asg)
        if sc is not None and better(sc, best_score):
            best_asg = dict(asg)
            best_score = sc
            remember(make_proposal(best_asg))

    if best_asg is None:
        asg = {tid: ids[0] for tid in ids}
        asg[ids[0]] = ids[0]
        p = split_repair(asg)
        d = deployment(p, proxy=True)
        if d['feasible']:
            remember(p)
            return p
        return {'nest_ids': [ids[0]], 'assignments': asg}

    cur_asg = dict(best_asg)
    cur_score = best_score

    phase('rebalance')
    max_rounds = 40
    for rnd in range(max_rounds):
        if remaining()['seconds'] <= 0.5:
            break
        groups, info = local_metrics(cur_asg)
        nests = sorted(groups.keys())
        if len(nests) <= 1:
            break
        loads = []
        for nid in nests:
            r = info[nid]
            tc = int(r['task_count'])
            loads.append((tc, nid))
        loads.sort(reverse=True)
        high_nid = loads[0][1]
        low_nid = loads[-1][1]
        high_mem = list(groups[high_nid])
        low_mem = list(groups[low_nid])
        moved = False
        candidates = [x for x in high_mem if x != high_nid]
        candidates.sort(key=lambda x: dist_id(x, low_nid))
        for tid in candidates[:min(6, len(candidates))]:
            tick()
            if remaining()['move_attempts'] <= 0:
                break
            cand = dict(cur_asg)
            cand[tid] = low_nid
            na, ns, ok = try_accept(cur_asg, cur_score, cand)
            if ok:
                cur_asg, cur_score = na, ns
                remember(make_proposal(cur_asg))
                moved = True
                break
        if moved:
            continue
        low_candidates = [x for x in low_mem if x != low_nid]
        low_candidates.sort(key=lambda x: dist_id(x, high_nid))
        for tid in low_candidates[:min(4, len(low_candidates))]:
            tick()
            if remaining()['move_attempts'] <= 0:
                break
            cand = dict(cur_asg)
            cand[tid] = high_nid
            na, ns, ok = try_accept(cur_asg, cur_score, cand)
            if ok:
                cur_asg, cur_score = na, ns
                remember(make_proposal(cur_asg))
                moved = True
                break
        if moved:
            continue
        swapped = False
        for a in candidates[:min(4, len(candidates))]:
            for b in low_candidates[:min(4, len(low_candidates))]:
                tick()
                if remaining()['move_attempts'] <= 0:
                    break
                cand = dict(cur_asg)
                cand[a] = low_nid
                cand[b] = high_nid
                na, ns, ok = try_accept(cur_asg, cur_score, cand)
                if ok:
                    cur_asg, cur_score = na, ns
                    remember(make_proposal(cur_asg))
                    swapped = True
                    break
            if swapped:
                break
        if not swapped:
            break

    phase('polish')
    groups, info = local_metrics(cur_asg)
    nests = sorted(groups.keys())
    for nid in nests:
        if remaining()['seconds'] <= 0.5:
            break
        mem = list(groups[nid])
        for tid in mem:
            if tid == nid:
                continue
            tick()
            if remaining()['move_attempts'] <= 0:
                break
            for other in nests:
                if other == nid:
                    continue
                cand = dict(cur_asg)
                cand[tid] = other
                na, ns, ok = try_accept(cur_asg, cur_score, cand)
                if ok:
                    cur_asg, cur_score = na, ns
                    remember(make_proposal(cur_asg))
                    break

    saved = best()
    if saved is not None:
        return saved
    return make_proposal(cur_asg)
