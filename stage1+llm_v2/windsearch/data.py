"""Frozen local KDE data; no deployment algorithm or routing package imports."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.stats import gaussian_kde

from .artifacts import atomic_json, sha256_file
from .domain import Turbine, load_turbines
from .settings import SearchSettings

DATA_VERSION = 1


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _identity(settings: SearchSettings) -> dict:
    paths = [settings.target_input, *settings.validation_inputs, settings.protected_input]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Dataset not found: {path}')
    return {
        'version': DATA_VERSION,
        'sources': [{'role': 'target' if index == 0 else 'protected' if index == len(paths)-1 else f'validation-{index}',
                     'sha256': sha256_file(path)} for index,path in enumerate(paths)],
        'training_virtual_fields': settings.training_virtual_fields,
        'validation_virtual_fields': settings.validation_virtual_fields,
        'data_seed': settings.data_seed, 'validation_seed': settings.validation_seed,
    }


def _reject_protected(points: list[Turbine], protected: list[Turbine]) -> None:
    # Compare geometry, not labels. Tolerance is 1e-8 km (0.01 mm).
    coordinates = np.array([(t.x_km, t.y_km) for t in points])
    blocked = np.array([(t.x_km, t.y_km) for t in protected])
    if np.any(np.all(np.isclose(coordinates[:, None, :], blocked[None, :, :],
                               rtol=0, atol=1e-8), axis=-1)):
        raise ValueError('Search data overlaps protected 87 coordinates; final test data cannot enter search feedback.')


def prepare_data(settings: SearchSettings) -> dict:
    """Create once; existing manifests must match both identity and content."""
    path = settings.data_dir / 'manifest.json'
    if path.exists():
        return read_bundle(settings)
    identity = _identity(settings)
    protected = load_turbines(settings.protected_input)
    source = load_turbines(settings.target_input)
    datasets = []
    for index, input_path in enumerate([settings.target_input, *settings.validation_inputs]):
        points = load_turbines(input_path)
        _reject_protected(points, protected)
        datasets.append({'name': 'real102' if index == 0 else f'external-validation-{index}',
                         'split': 'training' if index == 0 else 'validation',
                         'turbines': [asdict(t) for t in points]})
    train_seeds = set(range(settings.data_seed, settings.data_seed + settings.training_virtual_fields))
    validation_seeds = set(range(settings.validation_seed, settings.validation_seed + settings.validation_virtual_fields))
    if train_seeds & validation_seeds:
        raise ValueError('Training and validation seed ranges overlap.')
    kernel = None
    if train_seeds or validation_seeds:
        try:
            kernel = gaussian_kde(np.array([(t.x_km, t.y_km) for t in source]).T)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError('KDE requires non-singular two-dimensional covariance.') from error
    def geometry(rows: list[dict]) -> str:
        return fingerprint(sorted((t['x_km'],t['y_km']) for t in rows))
    seen = {geometry(row['turbines']) for row in datasets}
    for split, seeds in [('training', sorted(train_seeds)), ('validation', sorted(validation_seeds))]:
        for index, seed in enumerate(seeds):
            coordinates = kernel.resample(len(source), seed=np.random.default_rng(seed)).T
            points = [Turbine(f'kde-{i:04d}', float(x), float(y)) for i, (x, y) in enumerate(coordinates)]
            _reject_protected(points, protected)
            record = [asdict(t) for t in points]
            signature = geometry(record)
            if signature in seen:
                raise ValueError('Duplicate generated topology; choose independent seeds.')
            seen.add(signature)
            datasets.append({'name': f'{split}-kde-{index:03d}', 'split': split, 'seed': seed, 'turbines': record})
    payload = {'schema_version': DATA_VERSION, 'identity': identity, 'datasets': datasets}
    payload['fingerprint'] = fingerprint(payload)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: another run cannot replace the frozen dataset.
    temporary = path.with_name('.manifest-'+uuid4().hex+'.json')
    atomic_json(temporary,payload)
    try:
        # Atomic, exclusive publication on NTFS/POSIX filesystems; no stale lock
        # after a killed preparer and no overwrite of a concurrent manifest.
        try:
            os.link(temporary,path)
        except FileExistsError:
            return read_bundle(settings)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def read_bundle(settings: SearchSettings) -> dict:
    path = settings.data_dir / 'manifest.json'
    payload = json.loads(path.read_text(encoding='utf-8'))
    expected = payload.get('fingerprint')
    if expected != fingerprint({k:v for k,v in payload.items() if k != 'fingerprint'}):
        raise ValueError('Frozen data hash/fingerprint corrupt.')
    if payload.get('identity') != _identity(settings):
        raise ValueError('Frozen data identity changed; use a new data directory.')
    return payload


def turbines_from_record(record: dict) -> list[Turbine]:
    return [Turbine(**row) for row in record['turbines']]
