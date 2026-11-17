"""
rca.twin.ctm
============
Digital twin of a road segment: Cell Transmission Model (Daganzo 1994) with
the Greenshields speed-density relation (the model Indo-HCM Annex 5E.4 uses
for speed-flow analysis). Pure Python/NumPy: no MATLAB.

Fundamental diagram (Greenshields):
    v = vf (1 - k/kj),   q = vf k (1 - k/kj),   C = vf kj / 4,   kc = kj / 2
For each road state the jam density follows from the capacity and the
free-flow speed: kj = 4 C / vf.

Obstruction: the cells covering the obstructed length use the defect-adjusted
capacity C_d. With vf unchanged, kj_d = 4 C_d / vf, i.e. jam density falls
in proportion to usable width (CHOICE: same free-flow speed past the
obstruction; drivers slow only through congestion).

CTM update (dt such that vf dt = dx, CFL = 1):
    S_i = min(q(k_i) if k_i < kc_i else C_i, ...)    sending
    R_i = C_i if k_i < kc_i else q(k_i)               receiving
    y_i = min(S_{i-1}, R_i);  k_i += (y_i - y_{i+1}) dt / dx

Outputs: density/speed time-space fields, queue length, throughput, and
travel time from cumulative curves (N_in, N_out), for the clear road and the
obstructed road under the same demand.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def simulate(capacity_pcu_h: float, vf_kmh: float, demand_pcu_h: float, length_m: float = 1000.0,
             obstruction_capacity_pcu_h: Optional[float] = None, obstruction_at: float = 0.6,
             obstruction_length_m: float = 30.0, duration_s: float = 1800.0,
             demand_profile: str = "constant", snapshots: int = 120) -> dict:
    if capacity_pcu_h <= 0 or vf_kmh <= 0:
        raise ValueError("Capacity and free-flow speed must be positive.")
    vf = vf_kmh / 3.6                                  # m/s
    dx = max(5.0, min(25.0, length_m / 60))            # 40-200 cells
    dt = dx / vf
    n = max(10, int(round(length_m / dx)))
    steps = int(duration_s / dt)
    C = np.full(n, capacity_pcu_h / 3600.0)            # PCU/s
    if obstruction_capacity_pcu_h is not None and obstruction_capacity_pcu_h < capacity_pcu_h:
        i0 = int(obstruction_at * n)
        i1 = min(n, i0 + max(1, int(round(obstruction_length_m / dx))))
        C[i0:i1] = max(obstruction_capacity_pcu_h, 1.0) / 3600.0
        obs = (i0 * dx, i1 * dx)
    else:
        obs = None
    kj = 4 * C / vf                                    # PCU/m
    kc = kj / 2
    k = np.zeros(n)

    def q(kk):
        return vf * kk * (1 - kk / kj)

    t_save = np.linspace(0, steps - 1, min(snapshots, steps)).astype(int)
    save = set(t_save.tolist())
    dens, spd, times = [], [], []
    n_in = n_out = 0.0
    cum_in, cum_out, cum_t = [0.0], [0.0], [0.0]
    queue = []
    stored_demand = 0.0                                # vertical queue at entry if road is full
    for s in range(steps):
        t = s * dt
        d = demand_pcu_h / 3600.0 * _profile(demand_profile, t / duration_s)
        send = np.where(k < kc, q(k), C)
        recv = np.where(k < kc, C, q(k))
        stored_demand += d * dt
        inflow = min(stored_demand / dt, recv[0])
        y = np.empty(n + 1)
        y[0] = inflow
        y[1:n] = np.minimum(send[:-1], recv[1:])
        y[n] = send[-1]                                # free exit
        k = np.clip(k + (y[:-1] - y[1:]) * dt / dx, 0, kj)
        stored_demand -= inflow * dt
        n_in += inflow * dt
        n_out += y[n] * dt
        cum_in.append(n_in); cum_out.append(n_out); cum_t.append(t + dt)
        congested = k > kc * 1.001
        queue.append(float(congested.sum() * dx + stored_demand / max(kj[0], 1e-9)))
        if s in save:
            times.append(round(t, 1))
            dens.append(np.round(k * 1000, 2).tolist())          # PCU/km
            v = np.where(k > 1e-9, q(k) / np.maximum(k, 1e-9), vf) * 3.6
            spd.append(np.round(v, 1).tolist())
    tt = _travel_times(np.array(cum_t), np.array(cum_in), np.array(cum_out))
    free_tt = length_m / vf
    last_window = max(1, int(300 / dt))
    thr = (cum_out[-1] - cum_out[-1 - last_window]) / (last_window * dt) * 3600
    return {
        "inputs": {"capacity_pcu_h": capacity_pcu_h, "vf_kmh": vf_kmh, "demand_pcu_h": demand_pcu_h,
                   "length_m": length_m, "obstruction_capacity_pcu_h": obstruction_capacity_pcu_h,
                   "obstruction_m": obs, "duration_s": duration_s, "demand_profile": demand_profile},
        "cells": n, "dx_m": round(dx, 2), "dt_s": round(dt, 3),
        "jam_density_pcu_km": round(float(kj.min()) * 1000, 1),
        "times_s": times, "density_pcu_km": dens, "speed_kmh": spd,
        "x_m": [round((i + 0.5) * dx, 1) for i in range(n)],
        "queue_m": [round(v, 1) for v in queue[:: max(1, len(queue) // snapshots)]],
        "max_queue_m": round(max(queue) if queue else 0.0, 1),
        "throughput_pcu_h_last5min": round(thr, 1),
        "mean_travel_time_s": round(tt, 1) if tt else None,
        "free_flow_travel_time_s": round(free_tt, 1),
        "mean_delay_s_per_pcu": round(max(tt - free_tt, 0.0), 1) if tt else None,
        "entry_queue_pcu": round(stored_demand, 1),          # demand that could not enter yet
        "vehicles_on_segment_pcu": round(float((k * dx).sum()), 1),
    }


def _profile(kind: str, u: float) -> float:
    if kind == "peak":            # rises to 1.2x demand mid-period, symmetric
        return 0.8 + 0.4 * np.sin(np.pi * u)
    return 1.0


def _travel_times(t, n_in, n_out) -> Optional[float]:
    """Mean horizontal distance between cumulative curves (FIFO)."""
    if n_out[-1] < 1:
        return None
    levels = np.linspace(0, n_out[-1], 200)[1:]
    t_in = np.interp(levels, n_in, t)
    t_out = np.interp(levels, n_out, t)
    return float(np.mean(t_out - t_in))


def compare(c_clear: float, c_defect: float, vf_kmh: float, demand_pcu_h: float, **kw) -> dict:
    clear = simulate(c_clear, vf_kmh, demand_pcu_h, **kw)
    obstructed = simulate(c_clear, vf_kmh, demand_pcu_h, obstruction_capacity_pcu_h=c_defect, **kw)
    d0, d1 = clear["mean_delay_s_per_pcu"] or 0, obstructed["mean_delay_s_per_pcu"] or 0
    return {"clear": clear, "obstructed": obstructed,
            "summary": {"extra_delay_s_per_pcu": round(d1 - d0, 1),
                        "max_queue_m": obstructed["max_queue_m"],
                        "throughput_loss_pcu_h": round(float(clear["throughput_pcu_h_last5min"]
                                                       - obstructed["throughput_pcu_h_last5min"]), 1),
                        "breakdown": bool(demand_pcu_h > c_defect),
                        "queue_note": "Queue length includes demand waiting upstream of the modelled segment.",
                        "note": ("Demand exceeds the obstructed capacity: a queue grows for as long as "
                                 "this demand lasts.") if demand_pcu_h > c_defect else
                                ("Demand is below the obstructed capacity: traffic slows locally but no "
                                 "standing queue forms.")}}
