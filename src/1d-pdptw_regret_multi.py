"""
1D-PDPTW -- multi-start regret-2 heuristic + restricted master problem.

Runs `n_heuristic_runs` regret-2 + or-opt constructions (run 1 deterministic,
the rest with insertion noise), turns every resulting route into a column
(one column per distinct request set, cheapest kept), and solves the
set-partitioning MIP over exactly that column pool. No route enumeration.
This is Phase 0 + Phase 2 of the extended solver.

The result is optimal only relative to the heuristic column pool (mip_gap
refers to the restricted master problem, not to a global bound).

Usage:
    python 1d-pdptw_regret_multi.py <instance.json> [mip_time_limit_secs]
           [n_heuristic_runs] [noise_scale] [mip_setting_mode] [--single-tw]
    defaults: no time limit, 15 runs, noise 0.15, mode 0

Output:
    tests/results/out_1d_pdptw_regret_multi/out_1d_pdptw_regret_multi_<date>/
        out_1dpdptw_regret_multi_<date>_<instance>.json
"""
import sys
import os
import json
import signal
import time as _time
from datetime import datetime

import gurobipy as gp
from gurobipy import GRB

from common.instance import parse_instance
from common.paths import results_dir
from common.heuristics import generate_heuristic_routes, run_regret_oropt
from common.export import (build_result_base, group_events_for_export,
                           format_route_string)
from common.mip_settings import EXTENDED_MIP_MODES, apply_mip_mode

_sigterm_received = [False]


def _handle_sigterm(signum, frame):
    _sigterm_received[0] = True
    print("\n[SIGTERM] Will stop at next callback and export current best.", flush=True)


signal.signal(signal.SIGTERM, _handle_sigterm)


def solve(instance_path, mip_time_limit=None, n_heuristic_runs=15,
          noise_scale=0.15, mip_setting_mode=0, single_tw=False):
    start = _time.time()
    print(f"\n{'='*60}\nInstance : {instance_path}\n{'='*60}\n")

    Q, K, depot, time_horizon, requests, t, c, pen = parse_instance(
        instance_path, single_tw=single_tw)
    R = list(requests.keys())
    print(f"Requests : {len(R)}   Vehicles : {len(K)}   Capacity : {Q}   "
          f"Horizon : {time_horizon}   Single TW : {single_tw}")
    print(f"Heuristic runs : {n_heuristic_runs} (noise_scale={noise_scale})   "
          f"MIP time limit : {mip_time_limit or 'unlimited'}   "
          f"MIP mode : {mip_setting_mode}\n")

    # Phase 0: multi-start heuristic -> columns
    gen_start = _time.time()
    routes, _best, oropt_passes, oropt_moves = generate_heuristic_routes(
        Q, K, depot, time_horizon, requests, t, c, R,
        non_loop_penalty_ratio=pen, n_runs=n_heuristic_runs,
        noise_scale=noise_scale)
    gen_time = _time.time() - gen_start

    det = run_regret_oropt(Q, K, depot, time_horizon, requests, t, c, R, pen)

    with open(instance_path) as f:
        raw = json.load(f)
    res = build_result_base(
        raw, instance_path,
        "1dpdptw_regret_multi (heuristic column pool + set partitioning, "
        "no depot cost, non-loop penalty)", pen)
    res.update({
        "heuristic_only": False, "mip_status": None, "mip_time": None,
        "total_time": None, "mip_obj": None, "mip_bound": None, "mip_gap": None,
        "mip_setting_mode": mip_setting_mode, "single_tw": single_tw,
        "fleet_used": None, "valid_arcs": None,
        "ws_cost": det["cost"], "ws_fleet": det["fleet"], "ws_valid": det["valid"],
        "routes_generated": 0, "heuristic_routes": len(routes),
        "oropt_passes": oropt_passes, "oropt_moves": oropt_moves,
        "gen_time": gen_time, "enumeration_complete": None,
        "max_subset_size_configured": None, "max_subset_size_reached": None,
        "n_heuristic_runs": n_heuristic_runs, "noise_scale": noise_scale,
        "routes": [],
    })

    covered = set()
    for r in routes:
        covered |= r["requests"]
    missing = set(R) - covered
    if missing:
        print(f"[WARNING] Requests without any column: {sorted(missing)} -- MIP infeasible.")

    m = None
    mip_time = None
    if routes and not missing:
        print(f"\n{'─'*60}\nSolving restricted master ({len(routes)} columns) ...\n{'─'*60}\n")
        m = gp.Model("1D_PDPTW_regret_multi")
        m.setParam("NodefileStart", 20)
        apply_mip_mode(m, mip_setting_mode, EXTENDED_MIP_MODES, threads=8)
        if mip_time_limit is not None:
            m.setParam("TimeLimit", mip_time_limit)

        lam = m.addVars(len(routes), vtype=GRB.BINARY, name="lambda")
        m.setObjective(gp.quicksum(routes[r]["cost"] * lam[r] for r in range(len(routes))),
                       GRB.MINIMIZE)
        for rid in R:
            m.addConstr(gp.quicksum(lam[r] for r in range(len(routes))
                                    if rid in routes[r]["requests"]) == 1,
                        name=f"cover_{rid}")
        m.addConstr(gp.quicksum(lam.values()) <= len(K), name="fleet_size")

        # MIP start: greedy disjoint cover from the heuristic columns.
        cov, hint = set(), set()
        for i, rt in enumerate(routes):
            if rt["requests"].isdisjoint(cov):
                cov |= rt["requests"]
                hint.add(i)
            if cov >= set(R):
                break
        if cov >= set(R):
            for i in range(len(routes)):
                lam[i].Start = 1 if i in hint else 0

        def cb(model, where):
            if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL) and _sigterm_received[0]:
                model.terminate()
            elif where == GRB.Callback.MIPSOL:
                print(f"[SOLUTION] obj={model.cbGet(GRB.Callback.MIPSOL_OBJ):.4f}  "
                      f"t={model.cbGet(GRB.Callback.RUNTIME):.1f}s")

        mip_start = _time.time()
        m.optimize(cb)
        mip_time = _time.time() - mip_start

    total_time = _time.time() - start
    res["total_time"] = total_time
    res["mip_time"] = mip_time

    if m is not None:
        res["mip_status"] = m.status
        res["mip_bound"] = m.ObjBound if hasattr(m, "ObjBound") else None
        res["mip_gap"] = m.MIPGap if hasattr(m, "MIPGap") else None
        if m.SolCount > 0:
            res["mip_obj"] = m.objVal
            strs = []
            for r in range(len(routes)):
                if lam[r].X > 0.5:
                    raw_ev = [{"type": ty, "request": rid} for ty, rid in routes[r]["events"]]
                    grouped = group_events_for_export(raw_ev, requests, depot, t)
                    strs.append(format_route_string(grouped, routes[r]["cost"]))
            res["fleet_used"] = len(strs)
            res["routes"] = strs
            print(f"\nObjective : {m.objVal:.4f}   gap {m.MIPGap*100:.2f}%   "
                  f"vehicles {len(strs)} / {len(K)}   status {m.status}")
        else:
            print(f"\nNo feasible solution. Gurobi status: {m.status}")

    run_date = datetime.now().strftime("%Y-%m-%d")
    base_name = os.path.basename(instance_path).replace(".json", "")
    out_name = os.path.join(results_dir("out_1d_pdptw_regret_multi"),
                            f"out_1dpdptw_regret_multi_{run_date}_{base_name}.json")
    with open(out_name, "w") as f:
        json.dump(res, f, indent=4)
    print(f"Total time : {total_time:.2f}s\nResults written to: {out_name}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 1:
        print("python 1d-pdptw_regret_multi.py <instance.json> [mip_time_limit_secs] "
              "[n_heuristic_runs] [noise_scale] [mip_setting_mode] [--single-tw]")
        sys.exit(1)
    tl = float(args[1]) if len(args) >= 2 else None
    if tl is not None and tl <= 0:
        tl = None
    solve(args[0], tl,
          n_heuristic_runs=int(args[2]) if len(args) >= 3 else 15,
          noise_scale=float(args[3]) if len(args) >= 4 else 0.15,
          mip_setting_mode=int(args[4]) if len(args) >= 5 else 0,
          single_tw="--single-tw" in sys.argv)
