"""Pure assignment transformations. No geometry, inference, repair or selection."""


def validate(assignments, input_ids):
    if not isinstance(assignments, dict):
        raise ValueError("assignments must be a dict[str,str]")
    if any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in assignments.items()
    ):
        raise ValueError("assignment keys and targets must be ID strings")
    ids = set(input_ids)
    missing, extra = ids - set(assignments), set(assignments) - ids
    invalid = set(assignments.values()) - ids
    if missing or extra or invalid:
        raise ValueError(
            f"complete mapping required; missing_ids={sorted(missing)[:5]} "
            f"extra_ids={sorted(extra)[:5]} invalid_targets={sorted(invalid)[:5]}"
        )
    bad = [n for n in set(assignments.values()) if assignments[n] != n]
    if bad:
        raise ValueError(f"nest self-assignment required: {sorted(bad)[:5]}")


def transform(assignments, input_ids, operation, **params):
    validate(assignments, input_ids)
    before = dict(assignments)
    after = dict(before)
    nests = set(before.values())
    if operation == "reassign_tasks":
        updates = params["updates"]
        if not isinstance(updates, dict) or any(
            t not in before or n not in nests for t, n in updates.items()
        ):
            raise ValueError("updates must map input tasks to existing nests")
        if any(t in nests and n != t for t, n in updates.items()):
            raise ValueError("use remove_nest/relocate_nest for depot tasks")
        after.update(updates)
    else:
        source = params["source"]
        if not isinstance(source, str) or source not in nests:
            raise ValueError("source must be an existing nest ID")
        members = {t for t, n in before.items() if n == source}
        if operation == "remove_nest":
            updates = params["updates"]
            if not isinstance(updates, dict) or set(updates) != members:
                raise ValueError(
                    "updates must include every source member, including the old nest turbine"
                )
            if any(n not in nests - {source} for n in updates.values()):
                raise ValueError("removal targets must be remaining nests")
            after.update(updates)
        else:
            target = params["target"]
            if not isinstance(target, str) or target not in before or target == source:
                raise ValueError("target must be a different input turbine ID")
            if operation == "relocate_nest" and target in nests:
                raise ValueError(
                    "relocation target must not already be a nest; use merge_nests"
                )
            if operation == "merge_nests" and target not in nests:
                raise ValueError("merge target must be an existing nest")
            after.update({t: target for t in members})
            after[target] = target
    validate(after, input_ids)
    all_nests = nests | set(after.values())
    affected = sorted(
        n
        for n in all_nests
        if {t for t, owner in before.items() if owner == n}
        != {t for t, owner in after.items() if owner == n}
    )
    return {"assignments": after, "affected_nests": affected}
