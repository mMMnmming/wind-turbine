"""Optional pure transformations; no selection, inference or automatic repair."""


def merge(assignments, source, target):
    if source == target or assignments.get(source) != source or assignments.get(target) != target:
        raise ValueError("merge needs two distinct active nests")
    return {t: target if n == source else n for t, n in assignments.items()}


def relocate(assignments, source, target):
    if assignments.get(source) != source or target not in assignments or assignments[target] == target:
        raise ValueError("relocate needs an active source and inactive target turbine")
    result = {t: target if n == source else n for t, n in assignments.items()}
    result[target] = target
    return result


def remove(assignments, source, replacements):
    members = {t for t, n in assignments.items() if n == source}
    nests = set(assignments.values()) - {source}
    if not members or set(replacements) != members or not set(replacements.values()) <= nests:
        raise ValueError("replacements keys must equal ALL source members, including source itself; targets are remaining nests")
    return dict(assignments, **replacements)


def reassign(assignments, updates):
    nests = set(assignments.values())
    if any(t not in assignments or t in nests or n not in nests for t, n in updates.items()):
        raise ValueError("reassign only moves non-nest tasks to active nests")
    result = dict(assignments)
    result.update(updates)
    return result
