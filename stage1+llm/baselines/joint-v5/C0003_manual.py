"""Human revision of C0003, NOT a new LLM discovery. Frozen-network evaluation."""

import random

from candidate_tools import (
    assignments_from_proposal,
    best,
    better_metrics,
    deployment,
    distance,
    make_proposal,
    phase,
    reassign_tasks,
    region,
    relocate_nest,
    remaining,
    remember,
    remove_nest,
    repartition_repair,
    tick,
)


def _signature(assignments):
    return tuple(sorted(assignments.items()))


def _remove(assignments, rng, radius):
    nests = sorted(set(assignments.values()))
    if len(nests) < 2:
        return None
    old = rng.choice(nests)
    targets = [n for n in nests if n != old]
    updates = {}
    for task in sorted(assignments):
        if assignments[task] != old:
            continue
        nearby = [n for n in targets if distance(task, n) <= radius]
        if not nearby:
            return None
        updates[task] = min(nearby, key=lambda n: (distance(task, n), n))
    return remove_nest(assignments, old, updates)


def _relocate(assignments, rng):
    nests = sorted(set(assignments.values()))
    eligible = [
        n
        for n in nests
        if any(t != n and owner == n for t, owner in assignments.items())
    ]
    if not eligible:
        return None
    old = rng.choice(eligible)
    new = rng.choice(
        sorted(t for t, owner in assignments.items() if owner == old and t != old)
    )
    return relocate_nest(assignments, old, new)


def _redistribute(assignments, rng, exchange):
    nests = sorted(set(assignments.values()))
    tasks = sorted(t for t in assignments if t not in nests)
    if len(nests) < 2 or not tasks:
        return None
    first = rng.choice(tasks)
    if exchange:
        choices = [t for t in tasks if assignments[t] != assignments[first]]
        if not choices:
            return None
        second = rng.choice(choices)
        updates = {first: assignments[second], second: assignments[first]}
    else:
        target = rng.choice([n for n in nests if n != assignments[first]])
        updates = {first: target}
    return reassign_tasks(assignments, updates)


def _evaluate_change(change, seen, radius):
    if change is None:
        return None, None
    assignments = change["assignments"]
    signature = _signature(assignments)
    if signature in seen:
        return None, None
    seen.add(signature)
    groups = {
        n: [t for t, owner in assignments.items() if owner == n]
        for n in set(assignments.values())
    }
    # All geometric checks first; then network checks on changed active regions.
    for n in change["affected_nests"]:
        if n in groups and any(distance(n, t) > radius for t in groups[n]):
            return None, None
    for n in change["affected_nests"]:
        if n in groups and not region(n, groups[n])["feasible"]:
            return None, None
    proposal = make_proposal(assignments)
    return proposal, deployment(proposal)


def solve(turbines, config, seed):
    rng = random.Random(seed)
    ids = [t.identifier for t in turbines]
    radius = config.communication_radius_km
    phase("construct")
    order = list(ids)
    rng.shuffle(order)
    nests = []
    assignments = {}
    for task in order:
        nearby = [n for n in nests if distance(task, n) <= radius]
        if nearby:
            assignments[task] = min(nearby, key=lambda n: (distance(task, n), n))
        else:
            nests.append(task)
            assignments[task] = task
    # Explicit construction repair, never a hidden optimizer.
    proposal = repartition_repair(assignments)
    checked = deployment(proposal)
    if not checked["feasible"]:
        return proposal
    remember(proposal)
    current = assignments_from_proposal(proposal)
    seen = {_signature(current)}
    phase("improve_reduce_fixed_fixed")
    for iteration in range(500):
        budget = remaining()
        if (
            budget["seconds"] <= 0.5
            or budget["region_evaluations"] <= 0
            or budget["move_attempts"] <= 1
        ):
            break
        tick()
        operation = iteration % 3
        if operation == 0:
            change = _remove(current, rng, radius)
        elif operation == 1:
            change = _relocate(current, rng)
        else:
            change = _redistribute(current, rng, (iteration // 3) % 2 == 1)
        candidate, result = _evaluate_change(change, seen, radius)
        if (
            candidate is not None
            and result["feasible"]
            and better_metrics(result["objectives"], checked["objectives"])
        ):
            remember(candidate)
            current = assignments_from_proposal(candidate)
            checked = result
    return best()
