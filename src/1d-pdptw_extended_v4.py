"""
1D-PDPTW Extended Formulation (no depot leg cost + non-loop route penalty)

Solves the 1D-PDPTW using a set-partitioning formulation, with the same
cost rules as the compact MIP

Designed for Sartori-Buriol PDPTW instances, in particular the node-restricted variants.

Brief overview of solution pipeline (see solve() ):
1. Parse JSON instance.
2. Generate a pool of heuristic routes via the same strong regret-2
   construction + or-opt local search used by the compact model, run
   multiple times with increasing insertion noise to diversify Omega.
3. Enumerate feasible route set using a best-first subset search (same
   no-depot-cost + penalty cost rule as the heuristic and the MIP).
4. Build and solve the final set-partitioning MIP.
5. Export results to a JSON results file (same structure as the compact
   model's output).

MIP setting modes:
    0 - standard (default)
    1 - focus on closing bounds
    2 - focus on finding new incumbents

Usage:
python 1d-pdptw_extended_v4.py <instance> <MIP time limit> <route generation algorithm time limit> <MIP setting mode> <number of heuristic runs> <noise scale> [--single-tw]
"""


import sys
import json
import os
import signal
import time as _time
import heapq
import itertools
import gurobipy as gp

from gurobipy import GRB
from datetime import datetime
from common.instance import parse_instance
from common.paths import results_dir
from common.feasibility import check_heuristic as _check_heuristic
from common.local_search import or_opt_pass as _or_opt_pass
from common.heuristics import (
    construction_heuristic as _construction_heuristic,
    heuristic_cost as _heuristic_cost,
    generate_heuristic_routes,
)
from common.export import (
    group_events_for_export as _group_events_for_export,
    format_route_string as _format_route_string,
)
from common.mip_settings import EXTENDED_MIP_MODES as MIP_SETTING_MODES, apply_mip_mode


# SIGTERM handling: Slurm sends SIGTERM some grace period before the hard kill at the wall-clock limit.
_sigterm_received = [False]

def _handle_sigterm(signum, frame):
    _sigterm_received[0] = True
    print("\n[SIGTERM] Caught termination signal -- will stop at next "
          "callback poll and export current best incumbent.", flush=True)

signal.signal(signal.SIGTERM, _handle_sigterm)


# 1.  PARSE INSTANCE


# 2.  REGRET-2 CONSTRUCTION HEURISTIC (ported from the compact model)


# or-opt pass


# 3.  Convert heuristic route_events -> extended route (column) format.


# 4.  Heuristic route injection


# 5.  State-Space Route Generation (subset enumeration)

def generate_routes(
    Q,
    time_horizon,
    requests,
    t,
    c,
    max_subset_size,
    non_loop_penalty_ratio=0.0,
    gen_time_limit=None,
    existing_best=None,
    heuristic_routes=None,
):
    if existing_best is None:
        existing_best = {}

    R = list(requests.keys())
    routes = []
    gen_start_ts = _time.time()
    if heuristic_routes is None:
        heuristic_routes = []

    # Precompute single-request routes (initialize with heuristic).
    best_cost_for_set = dict(existing_best)

    for rid in R:
        req  = requests[rid]
        p, d = req["p_node"], req["d_node"]

        start_p = req["p_earliest"]
        if start_p > req["p_latest"]:
            continue

        start_d = max(start_p + req["p_duration"] + t(p, d), req["d_earliest"])
        if start_d > req["d_latest"]:
            continue
        if start_d + req["d_duration"] > time_horizon:
            continue

        cost = c(p, d)
        if p != d:
            cost += non_loop_penalty_ratio * c(d, p)
        s = frozenset([rid])

        best_cost_for_set[s] = cost
        routes.append({
            "requests": s,
            "path":     (p, d),
            "cost":     cost,
        })

    max_size_completed = 1

    pruned_total = 0

    def is_dominated(state_key, time, cost):
        if state_key in dominance:
            for (bt, bc) in dominance[state_key]:
                if time >= bt and cost >= bc:
                    return True
        return False

    def update_dominance(state_key, time, cost):
        if state_key not in dominance:
            dominance[state_key] = [(time, cost)]
            return
        frontier = dominance[state_key]
        dominance[state_key] = [(bt, bc) for (bt, bc) in frontier if not (bt >= time and bc >= cost)]
        dominance[state_key].append((time, cost))

    def deliver(node, time, cost, batch, dest):
        """
        Simulate sequential delivery of every request in batch at dest.
        """
        reqs = sorted(
            [requests[r] for r in batch],
            key=lambda r: (r["d_latest"], r["d_earliest"]),
        )
        arrival  = time + t(node, dest)
        new_cost = cost + c(node, dest)
        cur_time = arrival

        for req in reqs:
            start = max(cur_time, req["d_earliest"])
            if start > req["d_latest"]:
                return None
            cur_time = start + req["d_duration"]

        return cur_time, new_cost

    for k in range(2, max_subset_size + 1):

        for subset in itertools.combinations(R, k):

            if gen_time_limit is not None:
                elapsed = _time.time() - gen_start_ts
                if elapsed >= gen_time_limit:
                    print(f"Generation time limit ({gen_time_limit:.0f}s) reached.")
                    print(f"All subsets up to size {k - 1} are completed.")
                    print(f"Routes found so far: {len(routes):,}.")
                    return routes, False, max_size_completed
                print(f"Subset size {k} complete.")
                print(f"Elapsed time: {elapsed:.1f}s / {gen_time_limit:.1f}s")
                print(f"Routes found so far: {len(routes):,}.")

            S = frozenset(subset)
            UB = float("inf")
            UB_sum = sum(best_cost_for_set.get(frozenset([r]), float("inf")) for r in S)

            if S in best_cost_for_set:
                UB = best_cost_for_set[S]

            # INITIAL STATE (virtual, before the first pickup; no depot)
            # (cost, node, load, time, batch, active_dest, served, path,
            #  first_loc)
            # first_loc = route's first physical node (pi_1)
            start = (0.0, None, 0, 0.0, frozenset(), None, frozenset(), (), None)
            queue = [start]
            best_route = None
            dominance = {}

            while queue:

                cost, node, load, time, batch, active_dest, served, path, first_loc = heapq.heappop(queue)
                state_key = (node, load, active_dest, batch, served, first_loc)

                if cost >= UB:
                    break
                if cost > UB_sum:
                    break
                if is_dominated(state_key, time, cost):
                    pruned_total += 1
                    continue

                update_dominance(state_key, time, cost)

                # T4: CLOSE ROUTE. Deliver the last batch; the last delivery
                # must be finished by the time horizon. The non-loop penalty
                # is cost only (no travel time).
                if served == S:
                    result = deliver(node, time, cost, batch, active_dest)
                    if not result:
                        continue
                    d_time, d_cost = result

                    end_cost = d_cost
                    if first_loc != active_dest:
                        end_cost += non_loop_penalty_ratio * c(active_dest, first_loc)

                    if d_time <= time_horizon and end_cost < UB:
                        UB = end_cost
                        best_route = {
                            "requests": S,
                            "path": path + (active_dest,),
                            "cost": end_cost
                        }

                    continue

                # ROUTE EXPANSION
                for rid in S:
                    if rid in served:
                        continue

                    req = requests[rid]
                    p, d = req["p_node"], req["d_node"]
                    demand = req["demand"]

                    # T1: ROUTE START at the first pickup, at its earliest time
                    if load == 0:
                        start_t = req["p_earliest"]

                        if start_t <= req["p_latest"]:
                            heapq.heappush(queue, (
                                cost,
                                p,
                                demand,
                                start_t + req["p_duration"],
                                frozenset([rid]),
                                d,
                                served | {rid},
                                path + (p,),
                                p,
                            ))

                    # T2 & T3: PICKUP -> PICKUP, SAME DEST
                    elif active_dest == d:
                        arrival  = time + t(node, p)
                        start    = max(arrival, req["p_earliest"])
                        p_time   = start + req["p_duration"]
                        new_cost = cost + c(node, p)

                        # T2: extend current batch (no intermediate delivery)
                        if load + demand <= Q and start <= req["p_latest"]:
                            if deliver(p, p_time, new_cost, batch | {rid}, active_dest) is not None:
                                heapq.heappush(queue, (
                                    new_cost, p, load + demand,
                                    start + req["p_duration"],
                                    batch | {rid}, active_dest,
                                    served | {rid}, path + (p,), first_loc
                                ))

                        # T3: close current batch first, then pick up rid as new singleton
                        result = deliver(node, time, cost, batch, active_dest)
                        if result is not None:
                            d_time, d_cost = result
                            arr2   = d_time + t(active_dest, p)
                            start2 = max(arr2, req["p_earliest"])
                            if start2 <= req["p_latest"]:
                                heapq.heappush(queue, (
                                    d_cost + c(active_dest, p), p, demand,
                                    start2 + req["p_duration"],
                                    frozenset([rid]), d,
                                    served | {rid}, path + (active_dest, p), first_loc
                                ))

                    # T3: PICKUP -> PICKUP, DIFFERENT DEST (deliver, then new batch)
                    else:
                        result = deliver(node, time, cost, batch, active_dest)
                        if not result:
                            continue

                        d_time, d_cost = result

                        arrival = d_time + t(active_dest, p)
                        start = max(arrival, req["p_earliest"])

                        if start <= req["p_latest"]:
                            new_cost = d_cost + c(active_dest, p)

                            heapq.heappush(queue, (
                                    new_cost,
                                    p,
                                    demand,
                                    start + req["p_duration"],
                                    frozenset([rid]),
                                    d,
                                    served | {rid},
                                    path + (active_dest, p),
                                    first_loc
                                ))

            if best_route:
                S_key = best_route["requests"]
                if S_key not in best_cost_for_set or best_route["cost"] < best_cost_for_set[S_key]:
                    routes.append(best_route)
                    best_cost_for_set[S_key] = best_route["cost"]

        max_size_completed = k

    print(f"Dominance prunings: {pruned_total:,} routes")
    return routes, True, max_size_completed


# Route-event reconstruction helper
def _reconstruct_events_1d(path, req_ids, requests):
    """
    Route enumeration algorithm only stores the physical path, not which
    request was picked up or delivered at each stop. This function recovers
    this information.

    Reconstruct the ordered event sequence using DFS to handle intra-node
    requests and multiple visits to the same physical node correctly.
    """
    path_nodes = list(path)
    target_reqs = set(req_ids)

    def dfs(idx, curr_load, unserved, batch_dest):
        if idx == len(path_nodes):
            return [] if not curr_load and not unserved else None

        curr_node = path_nodes[idx]

        if batch_dest is not None and curr_node == batch_dest:
            delivery_events = []
            for rid in sorted(curr_load, key=lambda r: (requests[r]["d_latest"], requests[r]["d_earliest"])):
                delivery_events.append({"node": curr_node, "type": "D", "request": rid})

            res = dfs(idx + 1, set(), unserved, None)
            if res is not None:
                return delivery_events + res

        eligible_p = [
            rid for rid in unserved
            if requests[rid]["p_node"] == curr_node
            and (batch_dest is None or requests[rid]["d_node"] == batch_dest)
        ]

        for rid in eligible_p:
            new_batch_dest = requests[rid]["d_node"]
            pickup_event = {"node": curr_node, "type": "P", "request": rid}

            res = dfs(idx + 1, curr_load | {rid}, unserved - {rid}, new_batch_dest)
            if res is not None:
                return [pickup_event] + res

        return None

    events = dfs(0, set(), target_reqs, None)
    return events


# 6.  Set-Partitioning MIP

def solve(
    instance_path:      str,
    mip_time_limit:     float | None = None,
    gen_time_limit:     float | None = None,
    mip_setting_mode:   int = 0,
    max_subset_size:    int = 10,
    n_heuristic_runs:   int = 10,
    noise_scale:        float = 0.15,
    single_tw:          bool = False,
):

    print(f"\n{'='*60}")
    print(f"Instance : {instance_path}")
    print(f"{'='*60}\n")

    overall_start = _time.time()

    Q, K, depot, time_horizon, requests, t, c, non_loop_penalty_ratio = parse_instance(instance_path, single_tw=single_tw, free_depot=True)
    R = list(requests.keys())

    print(f"Requests               : {len(R)}")
    print(f"Vehicles               : {len(K)}")
    print(f"Capacity                : {Q}")
    print(f"Time horizon            : {time_horizon}")
    print(f"Non-loop penalty ratio  : {non_loop_penalty_ratio}")
    print(f"Max subset size         : {max_subset_size}")
    print(f"Heuristic runs          : {n_heuristic_runs}  (noise_scale={noise_scale})")
    print(f"Gen time limit          : {gen_time_limit if gen_time_limit else 'unlimited'}")
    print(f"MIP time limit          : {mip_time_limit if mip_time_limit else 'unlimited'}")
    print(f"MIP setting mode        : {mip_setting_mode}")
    print(f"Single time window      : {single_tw}")
    print()

    heuristic_routes = []
    heuristic_best   = {}
    total_oropt_passes = 0
    total_oropt_moves  = 0

    ws_cost, ws_fleet, ws_valid = None, None, None

    if n_heuristic_runs > 0:
        heuristic_routes, heuristic_best, total_oropt_passes, total_oropt_moves = generate_heuristic_routes(
            Q, K, depot, time_horizon, requests, t, c, R,
            non_loop_penalty_ratio=non_loop_penalty_ratio,
            n_runs=n_heuristic_runs,
            noise_scale=noise_scale,
        )

        det_events = _construction_heuristic(
            Q, K, depot, time_horizon, requests, t, c, R,
            non_loop_penalty_ratio=non_loop_penalty_ratio,
        )
        n_viol, det_missing = _check_heuristic(
            det_events, R, Q, depot, time_horizon, requests, t
        )
        ws_valid = (n_viol == 0 and not det_missing)
        if ws_valid:
            det_events, _c, _m, _p = _or_opt_pass(
                det_events, requests, depot, Q, time_horizon, t, c,
                non_loop_penalty_ratio,
            )
            total_oropt_passes += _p
            total_oropt_moves += _m
            ws_cost = _heuristic_cost(det_events, requests, c, depot, non_loop_penalty_ratio)
            ws_fleet = sum(1 for r in det_events if r)

    print(f"{'─'*60}")
    print(f"Phase 1: Generating feasible 1D-PDPTW routes ...")
    print(f"Start clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}")
    print()

    gen_start = _time.time()
    enum_routes, enumeration_complete, max_subset_size_reached = generate_routes(
        Q, time_horizon, requests, t, c,
        max_subset_size=max_subset_size,
        non_loop_penalty_ratio=non_loop_penalty_ratio,
        gen_time_limit=gen_time_limit,
        existing_best=heuristic_best,
        heuristic_routes=heuristic_routes,
    )
    gen_time = _time.time() - gen_start

    print(f"\nRoutes generated (enumeration) : {len(enum_routes):,}")
    print(f"Generation time                 : {gen_time:.2f}s")
    print(f"Enumeration complete             : {enumeration_complete}"
          + ("" if enumeration_complete else
             f"  (only fully covered subset sizes up to {max_subset_size_reached} "))
    print()

    routes = heuristic_routes + enum_routes

    covered = set()
    for r in routes:
        covered |= r["requests"]
    missing = set(requests.keys()) - covered

    print(f"Total columns in Omega_hat : {len(routes):,}  "
          f"({len(heuristic_routes)} heuristic + {len(enum_routes)} enumerated)")
    print(f"Covered requests           : {len(covered)} / {len(requests)}")
    if missing:
        print(f"[WARNING] Missing requests (no feasible route found): {missing}")

    if not routes:
        print("No feasible routes found. Cannot solve MIP.")
        return

    print(f"\n{'─'*60}")
    print(f"Phase 2: Solving set-partitioning MIP ...")
    print(f"Start clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}\n")

    m = gp.Model("1D_PDPTW")

    m.setParam("NodefileStart", 20)
    applied_params = apply_mip_mode(m, mip_setting_mode, MIP_SETTING_MODES, threads=8)
    print(f"MIP setting mode {mip_setting_mode} params: {applied_params}\n")

    if mip_time_limit is not None:
        print(f"Gurobi time limit : {mip_time_limit:.1f}s\n")
        m.setParam("TimeLimit", mip_time_limit)

    # Binary selection variable for each candidate route r
    lam = m.addVars(len(routes), vtype=GRB.BINARY, name="lambda")

    # Objective: minimize total route cost
    m.setObjective(
        gp.quicksum(routes[r]["cost"] * lam[r] for r in range(len(routes))),
        GRB.MINIMIZE,
    )

    # Constraint: every request covered exactly once
    for rid in R:
        m.addConstr(
            gp.quicksum(
                lam[r]
                for r in range(len(routes))
                if rid in routes[r]["requests"]
            ) == 1,
            name=f"cover_{rid}",
        )

    # Constraint: fleet-size limit
    m.addConstr(
        gp.quicksum(lam[r] for r in range(len(routes))) <= len(K),
        name="fleet_size",
    )

    # Warm-start hint: inject the best heuristic cover as an initial solution.
    _inject_warm_start_hint(lam, routes, heuristic_routes, R)

    # Callback: print every new incumbent solution, and poll for SIGTERM.
    sol_count = [0]
    last_obj  = [None]

    def print_incumbent(model, where):
        if where in (GRB.Callback.MIP, GRB.Callback.MIPSOL) and _sigterm_received[0]:
            print("[SIGTERM] Terminating solve at "
                  f"{model.cbGet(GRB.Callback.RUNTIME):.1f}s -- "
                  "exporting current best incumbent.", flush=True)
            model.terminate()
            return

        if where != GRB.Callback.MIPSOL:
            return

        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        if last_obj[0] is not None and abs(obj - last_obj[0]) < 1e-6:
            return
        last_obj[0] = obj

        sol_count[0] += 1
        bound   = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        runtime = model.cbGet(GRB.Callback.RUNTIME)

        if abs(bound) > 1e10:
            gap_str = "no bound yet"
        else:
            gap     = abs(obj - bound) / (abs(obj) + 1e-10) * 100
            gap_str = f"{gap:.2f}%"

        print(f"\n[SOLUTION #{sol_count[0]}]  "
              f"obj={obj:.4f}  bound={bound:.4f}  "
              f"gap={gap_str}  t={runtime:.1f}s")

        lam_vals = model.cbGetSolution([lam[r] for r in range(len(routes))])
        vehicles_printed = 0
        for r_idx, val in enumerate(lam_vals):
            if val > 0.5:
                vehicles_printed += 1
                src      = "H" if r_idx < len(heuristic_routes) else "E"
                path_str = " -> ".join(str(nd) for nd in routes[r_idx]["path"])
                req_str  = ", ".join(str(i) for i in sorted(routes[r_idx]["requests"]))
                print(f"  Veh {vehicles_printed:>2} [{src}]: [{req_str}]  "
                      f"path: {path_str}")

        print(f"  ({vehicles_printed} vehicles active)")

    mip_start = _time.time()
    m.optimize(print_incumbent)
    mip_time  = _time.time() - mip_start

    # Results
    total_time = _time.time() - overall_start

    _status_labels = {
        GRB.OPTIMAL: "Optimal",
        GRB.TIME_LIMIT: "Time limit (best found)",
        GRB.INTERRUPTED: "SIGTERM checkpoint (best found)",
    }

    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and m.SolCount > 0:
        gap = m.MIPGap * 100
        print(f"\nObjective  : {m.objVal:.4f}")
        print(f"MIP gap    : {gap:.2f}%")
        print(f"Status     : {_status_labels.get(m.status, m.status)}")
        if not enumeration_complete:
            print(f"NOTE       : route enumeration did not finish (stopped after "
                  f"fully covering subset size {max_subset_size_reached} of "
                  f"configured max {max_subset_size}) -- this objective is "
                  f"optimal only relative to the resulting incomplete candidate "
                  f"pool, not a proven global optimum, even though MIP gap is "
                  f"{gap:.2f}%.")
        print(f"MIP time   : {mip_time:.2f}s")

        vehicles_used = 0
        for r in range(len(routes)):
            if lam[r].X > 0.5:
                vehicles_used += 1
                src = "H" if r < len(heuristic_routes) else "E"   # H=heuristic, E=enumerated
                path_str = " -> ".join(str(n) for n in routes[r]["path"])
                req_str  = ", ".join(str(i) for i in sorted(routes[r]["requests"]))
                print(f"\n  Vehicle {vehicles_used} [{src}]:")
                print(f"    Route    : {path_str}")
                print(f"    Requests : [{req_str}]")
                print(f"    Cost     : {routes[r]['cost']:.4f}")

        print(f"\nVehicles used : {vehicles_used} / {len(K)}")
        print(f"Total time    : {total_time:.2f}s")

    elif m.status == GRB.INFEASIBLE:
        print(f"\nModel is infeasible. Gurobi status: {m.status}")
    else:
        print(f"\nNo feasible solution found. Gurobi status: {m.status}")

    with open(instance_path) as f:
        raw_data = json.load(f)

    res = {
        "instance": raw_data.get("instance_name", instance_path),
        "base_instance": raw_data.get("base_instance_name"),
        "city": raw_data.get("real_world_inspiration"),
        "loc_distr": raw_data.get("location_distribution"),
        "dep_distr": raw_data.get("depot_location_distribution"),
        "restr": raw_data.get("variant", {}).get("node_restriction_percentage"),
        "n_requests": len(raw_data.get("requests", [])),
        "n_nodes": len(raw_data.get("nodes", [])),
        "n_vehicles": raw_data.get("vehicle", {}).get("fleet_size"),
        "capacity": raw_data.get("vehicle", {}).get("capacity"),
        "model": "1dpdptw_extended (no depot, non-loop penalty)",
        "non_loop_route_penalty_ratio": non_loop_penalty_ratio,
        "mip_status": m.status,
        "mip_time": mip_time if 'mip_time' in locals() else None,
        "total_time": total_time,
        "mip_obj": m.objVal if m.SolCount > 0 else None,
        "mip_bound": m.ObjBound if hasattr(m, "ObjBound") else None,
        "mip_gap": m.MIPGap if hasattr(m, "MIPGap") else None,
        "mip_setting_mode": mip_setting_mode,
        "single_tw": single_tw,
        "fleet_used": None,
        "valid_arcs": None,
        "ws_cost": ws_cost,
        "ws_fleet": ws_fleet,
        "ws_valid": ws_valid,
        "routes_generated": len(enum_routes) if 'enum_routes' in locals() else 0,
        "heuristic_routes": len(heuristic_routes) if 'heuristic_routes' in locals() else 0,
        "oropt_passes": total_oropt_passes,
        "oropt_moves": total_oropt_moves,
        "gen_time": gen_time if 'gen_time' in locals() else None,
        "enumeration_complete": enumeration_complete,
        "max_subset_size_configured": max_subset_size,
        "max_subset_size_reached": max_subset_size_reached,
        "routes": [],
    }

    # Extract routes and fleet count if a solution exists
    if m.SolCount > 0:
        fleet_count = 0
        routes_list = []
        for r in range(len(routes)):
            if lam[r].X > 0.5:
                fleet_count += 1
                rpath   = list(routes[r]["path"])
                req_ids = routes[r]["requests"]

                # Heuristic-derived columns already carry their exact event
                # sequence; enumerated columns only have a physical path and
                # need DFS reconstruction.
                if "events" in routes[r]:
                    raw_events = [
                        {"type": ev_type, "request": rid}
                        for ev_type, rid in routes[r]["events"]
                    ]
                else:
                    raw_events = _reconstruct_events_1d(rpath, req_ids, requests)

                grouped_events = _group_events_for_export(raw_events, requests, depot, t)

                route_str = _format_route_string(grouped_events, routes[r]["cost"])
                routes_list.append(route_str)
        res["fleet_used"] = fleet_count
        res["routes"] = routes_list

    # Save to JSON.

    run_date = datetime.now().strftime("%Y-%m-%d")
    base_name = os.path.basename(instance_path).replace('.json', '')

    output_dir = results_dir("out_1d_pdptw_ex")

    out_name = os.path.join(
        output_dir, f"out_1dpdptw_ex_{run_date}_{base_name}.json"
    )
    with open(out_name, "w") as f:
        json.dump(res, f, indent=4)

    print(f"\nResults written to: {out_name}")


# 7.  Warm-start hint for the MIP

def _inject_warm_start_hint(lam, all_routes, heuristic_routes, R):
    if not heuristic_routes:
        return

    covered   = set()
    hint_idxs = set()

    for r_idx, route in enumerate(all_routes):
        if r_idx >= len(heuristic_routes):
            break
        req_set = route["requests"]
        if req_set.isdisjoint(covered):
            covered   |= req_set
            hint_idxs.add(r_idx)
        if covered >= set(R):
            break

    if covered < set(R):
        return

    for r_idx in range(len(all_routes)):
        lam[r_idx].Start = 1 if r_idx in hint_idxs else 0

    print(f"  [Warm-start hint] {len(hint_idxs)} heuristic routes injected as MIP start.\n")


# 8.  Entry Point

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python 1d-pdptw_extended_v4.py <instance> <MIP time limit> "
              "<route generation algorithm time limit> <MIP setting mode> "
              "<number of heuristic runs> <noise scale> [--single-tw]")
        sys.exit(1)

    _argv = [a for a in sys.argv if not a.startswith("--")]
    _single_tw = "--single-tw" in sys.argv

    _instance           = _argv[1]
    _mip_time_limit     = float(_argv[2]) if len(_argv) >= 3 else None
    _gen_time_limit     = float(_argv[3]) if len(_argv) >= 4 else None
    _mip_setting_mode   = int(_argv[4])   if len(_argv) >= 5 else 0
    _n_heuristic_runs   = int(_argv[5])   if len(_argv) >= 6 else 15
    _noise_scale        = float(_argv[6]) if len(_argv) >= 7 else 0.15

    solve(_instance, _mip_time_limit, _gen_time_limit,
          mip_setting_mode=_mip_setting_mode,
          n_heuristic_runs=_n_heuristic_runs,
          noise_scale=_noise_scale,
          single_tw=_single_tw)
