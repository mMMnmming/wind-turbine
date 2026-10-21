"""Deterministic ATSP proxy, deliberately not an optimality oracle."""
from __future__ import annotations

import math
import numpy as np

from .domain import DeploymentConfig, Turbine


def directed_times(points: list[Turbine], config: DeploymentConfig) -> np.ndarray:
    xy = np.array([(t.x_km, t.y_km) for t in points],dtype=float)
    delta = xy[None,:,:] - xy[:,None,:]
    angles = np.arctan2(delta[...,1],delta[...,0])
    wind_angle = math.radians(90-config.wind_direction_degrees)
    speed = config.max_horizontal_speed_mps + config.wind_speed_mps*np.cos(angles-wind_angle)
    if np.any(speed <= 0):
        raise ValueError('Non-flyable directed edge.')
    return np.linalg.norm(delta,axis=-1)*1000/speed


def route_cost(route: list[int], times: np.ndarray) -> float:
    return float(sum(times[a,b] for a,b in zip(route,route[1:])))


def proxy_route(points: list[Turbine], config: DeploymentConfig,
                audit_ids: list[str] | tuple[str,...]) -> tuple[tuple[str,...],float]:
    """Depot is first; include audit tour and up to eight directed NN starts."""
    by_id = {t.identifier:i for i,t in enumerate(points)}
    times = directed_times(points,config)
    candidates = [[0,*(by_id[key] for key in audit_ids if by_id[key] != 0),0]]
    count = len(points)-1
    if count:
        for start in np.linspace(1,count,min(8,count),dtype=int):
            route = [0,int(start)]
            remaining = set(range(1,len(points)))-{int(start)}
            while remaining:
                nxt = min(remaining,key=lambda j:(times[route[-1],j],j))
                route.append(nxt)
                remaining.remove(nxt)
            candidates.append([*route,0])
    best = min(candidates,key=lambda route:(route_cost(route,times),route))
    for _ in range(20):
        improved, value = best, route_cost(best,times)
        best_value = value
        # Directed relocation delta is exact even for adjacent edges.
        for old in range(1,len(best)-1):
            a,node,b = best[old-1:old+2]
            removed = -times[a,node]-times[node,b]+times[a,b]
            shortened = best[:old]+best[old+1:]
            for insert in range(1,len(shortened)):
                left,right = shortened[insert-1:insert+1]
                cost = value+removed-times[left,right]+times[left,node]+times[node,right]
                if cost < best_value-1e-8:
                    improved = shortened[:insert]+[node]+shortened[insert:]
                    best_value = cost
        if improved is best:
            break
        best = improved
    if best[0] != 0 or best[-1] != 0 or sorted(best[1:-1]) != list(range(1,len(points))):
        raise ValueError('Proxy route must visit all tasks and return to depot.')
    flight = route_cost(best,times)
    energy = flight+count*config.hover_seconds_per_turbine*config.hover_to_flight_energy_ratio
    if energy > config.max_endurance_seconds+1e-9:
        raise ValueError('Proxy route violates endurance despite conservative audit.')
    return tuple(points[i].identifier for i in best),flight
