import math, random

def solve(turbines, config, seed):
    rnd = random.Random(seed)
    ids = [t.identifier for t in turbines]
    pos = {t.identifier: (t.x_km, t.y_km) for t in turbines}
    n = len(ids)
    if n == 0:
        return {'nest_ids': [], 'assignments': {}}
    R = float(config.communication_radius_km)
    vmax = float(config.max_horizontal_speed_mps)
    vw = float(config.wind_speed_mps)
    E = float(config.max_endurance_seconds)
    hov = float(config.hover_seconds_per_turbine)
    hr = float(config.hover_to_flight_energy_ratio)
    wd = float(config.wind_direction_degrees)
    eff = max(1e-9, vmax - vw)
    rad = min(R, eff * E / 2000.0)
    wang = math.radians(90.0 - wd)
    def dist(a, b):
        ax, ay = pos[a]; bx, by = pos[b]
        return math.hypot(ax-bx, ay-by)
    def edge_time(a, b):
        ax, ay = pos[a]; bx, by = pos[b]
        dx = bx-ax; dy = by-ay
        d = math.hypot(dx, dy)
        if d <= 1e-12:
            return 0.0
        ang = math.atan2(dy, dx)
        sp = vmax + vw * math.cos(ang - wang)
        if sp < 1e-6:
            sp = 1e-6
        return d * 1000.0 / sp
    def route(nest, members):
        rem = set(members); rem.discard(nest)
        cur = nest; total = 0.0
        while rem:
            nxt = min(rem, key=lambda j: (dist(cur, j), j))
            total += dist(cur, nxt); rem.remove(nxt); cur = nxt
        total += dist(cur, nest)
        return total
    def proxy(nest, members):
        rem = set(members); rem.discard(nest)
        cur = nest; total = 0.0
        while rem:
            nxt = min(rem, key=lambda j: (edge_time(cur, j), j))
            total += edge_time(cur, nxt); rem.remove(nxt); cur = nxt
        total += edge_time(cur, nest)
        return total
    def valid(nest, members):
        if nest not in members:
            return False
        for m in members:
            if dist(nest, m) > rad + 1e-9:
                return False
        d = route(nest, members)
        ft = d * 1000.0 / eff
        en = ft + len(members) * hov * hr
        return en <= E + 1e-9
    def score(assign):
        nests = sorted(set(assign.values()))
        groups = {k: [] for k in nests}
        for t, k in assign.items():
            groups[k].append(t)
        if any(not valid(k, groups[k]) for k in nests):
            return (1e9, 1e9, 1e9)
        pt = sum(proxy(k, groups[k]) for k in nests)
        mx = max((len(groups[k]) - 1) / n for k in nests) if nests else 0.0
        return (len(nests) / n, pt / n, mx)
    def construct(order):
        assign = {}; nests = []
        for t in order:
            best = None
            for k in nests:
                mem = [x for x, y in assign.items() if y == k] + [t]
                if valid(k, mem):
                    c = proxy(k, mem) - proxy(k, [x for x, y in assign.items() if y == k])
                    if best is None or c < best[0]:
                        best = (c, k)
            if best is not None:
                assign[t] = best[1]
            else:
                assign[t] = t; nests.append(t)
        return assign
    def repair(assign):
        assign = dict(assign)
        for _ in range(3 * n + 6):
            nests = sorted(set(assign.values()))
            groups = {k: [t for t, v in assign.items() if v == k] for k in nests}
            bad = [k for k in nests if not valid(k, groups[k])]
            if not bad:
                return assign
            k = bad[0]
            mem = groups[k]
            if len(mem) <= 1:
                return None
            moved = False
            for t in sorted(mem, key=lambda x: (dist(k, x), x), reverse=True):
                if t == k:
                    continue
                for k2 in nests:
                    if k2 == k:
                        continue
                    m2 = groups[k2] + [t]
                    m1 = [x for x in mem if x != t]
                    if valid(k2, m2) and valid(k, m1):
                        assign[t] = k2; moved = True; break
                if moved:
                    break
            if not moved:
                for t in sorted(mem, key=lambda x: (dist(k, x), x), reverse=True):
                    if t == k:
                        continue
                    m1 = [x for x in mem if x != t]
                    if valid(k, m1):
                        assign[t] = t; moved = True; break
            if not moved:
                return None
        return None
    def local(assign, rounds):
        cur = dict(assign); cs = score(cur)
        for _ in range(rounds):
            nests = sorted(set(cur.values()))
            groups = {k: [t for t, v in cur.items() if v == k] for k in nests}
            cand = None
            for k in nests:
                for t in groups[k]:
                    if t == k:
                        continue
                    for k2 in nests:
                        if k2 == k:
                            continue
                        m1 = [x for x in groups[k] if x != t]
                        m2 = groups[k2] + [t]
                        if valid(k, m1) and valid(k2, m2):
                            a = dict(cur); a[t] = k2
                            s = score(a)
                            if s < cs:
                                cand = (a, s); break
                    if cand: break
                if cand: break
            if not cand:
                break
            cur, cs = cand
        return cur, cs
    def reduce(assign):
        cur = dict(assign); cs = score(cur)
        for _ in range(n + 2):
            nests = sorted(set(cur.values()))
            if len(nests) <= 1:
                break
            best = None
            groups = {x: [t for t, v in cur.items() if v == x] for x in nests}
            for k in nests:
                mem = groups[k]
                if len(mem) <= 1:
                    continue
                for k2 in nests:
                    if k2 == k:
                        continue
                    m2 = groups[k2] + mem
                    if valid(k2, m2):
                        a = {t: (k2 if v == k else v) for t, v in cur.items()}
                        s = score(a)
                        if best is None or s < best[1]:
                            best = (a, s)
            if best is None or not (best[1] < cs):
                break
            cur, cs = best
        return cur, cs
    orders = []
    orders.append(sorted(ids, key=lambda t: (pos[t][0], pos[t][1], t)))
    orders.append(sorted(ids, key=lambda t: (pos[t][1], pos[t][0], t)))
    orders.append(sorted(ids, key=lambda t: (dist(t, ids[0]), t)))
    o = list(ids); rnd.shuffle(o); orders.append(o)
    best = None
    for o in orders:
        a = construct(o)
        a = repair(a)
        if a is None:
            continue
        a, s = local(a, n + 3)
        a, s = reduce(a)
        a, s = local(a, n + 2)
        if best is None or s < best[1]:
            best = (a, s)
    if best is None:
        a = {t: t for t in ids}
        a = repair(a) or a
        best = (a, score(a))
    a = best[0]
    nests = sorted(set(a.values()))
    return {'nest_ids': nests, 'assignments': {t: a[t] for t in ids}}
