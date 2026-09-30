"""
1D-PDPTW -- regret-2 construction heuristic + or-opt only (no MIP).

Runs the deterministic regret-2 construction followed by or-opt passes and
exports the resulting solution. The JSON has the same keys as the extended
solver's output: `mip_obj` holds the heuristic objective (a valid upper
bound), MIP-specific fields are None, and `heuristic_only` is True.

Usage:
    python 1d-pdptw_regret.py <instance.json> [--single-tw]

Output:
    tests/results/out_1d_pdptw_regret/out_1d_pdptw_regret_<date>/
        out_1dpdptw_regret_<date>_<instance>.json
"""
import sys
import os
import json
import time as _time
from datetime import datetime

from common.instance import parse_instance
from common.paths import results_dir
from common.feasibility import route_cost
from common.heuristics import run_regret_oropt
from common.export import build_result_base, route_strings


def solve(instance_path, single_tw=False):
    start = _time.time()
    print(f"\n{'='*60}\nInstance : {instance_path}\n{'='*60}\n")

    Q, K, depot, time_horizon, requests, t, c, pen = parse_instance(
        instance_path, single_tw=single_tw)
    R = list(requests.keys())
    print(f"Requests : {len(R)}   Vehicles : {len(K)}   Capacity : {Q}   "
          f"Horizon : {time_horizon}   Single TW : {single_tw}")

    h_start = _time.time()
    out = run_regret_oropt(Q, K, depot, time_horizon, requests, t, c, R, pen)
    h_time = _time.time() - h_start

    if out["valid"]:
        print(f"\nHeuristic objective : {out['cost']:.4f}  "
              f"({out['fleet']} / {len(K)} vehicles, "
              f"{out['oropt_moves']} or-opt moves in {out['oropt_passes']} passes)")
    else:
        print(f"\n[WARNING] Construction invalid: {out['violations']} violations, "
              f"missing requests: {sorted(out['missing'])}")

    with open(instance_path) as f:
        raw = json.load(f)

    res = build_result_base(raw, instance_path,
                            "1dpdptw_regret (heuristic only, no depot cost, non-loop penalty)", pen)
    total_time = _time.time() - start
    res.update({
        "heuristic_only": True,
        "mip_status": None,
        "mip_time": None,
        "total_time": total_time,
        "mip_obj": out["cost"],
        "mip_bound": None,
        "mip_gap": None,
        "mip_setting_mode": None,
        "single_tw": single_tw,
        "fleet_used": out["fleet"],
        "valid_arcs": None,
        "ws_cost": out["cost"],
        "ws_fleet": out["fleet"],
        "ws_valid": out["valid"],
        "routes_generated": 0,
        "heuristic_routes": sum(1 for r in out["events"] if r),
        "oropt_passes": out["oropt_passes"],
        "oropt_moves": out["oropt_moves"],
        "gen_time": h_time,
        "enumeration_complete": None,
        "max_subset_size_configured": None,
        "max_subset_size_reached": None,
        "routes": route_strings(
            out["events"], requests, depot, t, c,
            lambda ev: route_cost(ev, requests, depot, c, pen)) if out["valid"] else [],
    })

    run_date = datetime.now().strftime("%Y-%m-%d")
    base_name = os.path.basename(instance_path).replace(".json", "")
    out_name = os.path.join(results_dir("out_1d_pdptw_regret"),
                            f"out_1dpdptw_regret_{run_date}_{base_name}.json")
    with open(out_name, "w") as f:
        json.dump(res, f, indent=4)
    print(f"Total time : {total_time:.2f}s\nResults written to: {out_name}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 1:
        print("python 1d-pdptw_regret.py <instance.json> [--single-tw]")
        sys.exit(1)
    solve(args[0], single_tw="--single-tw" in sys.argv)
