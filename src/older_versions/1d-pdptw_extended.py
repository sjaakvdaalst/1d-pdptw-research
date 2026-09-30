"""
1D-PDPTW Extended Formulation

Solves the 1D-PDPTW using a set-partitioning formulation.
Designed for Sartori-Buriol PDPTW instances, in particular the node-restricted variants.

Brief overview of solution pipeline (see solve() ):
1. Parse JSON instance.
2. Generate a pool of heuristic routes via a number of randomized regret-2 construction runs.
3. Enumerate feasible route set using a best-first subset search.
4. Build and solve the set-partitioning MIP.
5. Export results to a JSON results file.

Usage:
python 1d-pdptw_extended.py <instance.json> [mip_time_limit] [gen_time_limit] [max_subset_size] [n_heuristic_runs] [noise_scale]
"""


import sys
import json
import os
import random
import time as _time
import heapq
import itertools
import gurobipy as gp

from gurobipy import GRB
from datetime import datetime


# 1.  PARSE INSTANCE

def parse_instance(filepath: str):

    with open(filepath) as f:
        data = json.load(f)

    Q            = data["vehicle"]["capacity"]
    K            = list(range(data["vehicle"]["fleet_size"]))
    depot        = data.get("depot_node", data.get("depot"))
    time_horizon = data.get("time_horizon", 10 ** 6)
    T_raw        = data["travel_times"]
    C_raw        = data.get("travel_costs", data["travel_times"])

    # travel time between physical nodes i and j
    def t(i: int, j: int):
        return T_raw[i][j]

    # travel cost between physical nodes i and j
    def c(i: int, j: int):
        return C_raw[i][j]

    requests: dict = {}
    for req in data["requests"]:
        rid = req["id"]
        requests[rid] = {
            "demand":     req["demand"],
            "p_node":     req["pickup"]["node"],
            "d_node":     req["delivery"]["node"],
            "p_earliest": req["pickup"]["earliest"],
            "p_latest":   req["pickup"]["latest"],
            "p_duration": req["pickup"]["duration"],
            "d_earliest": req["delivery"]["earliest"],
            "d_latest":   req["delivery"]["latest"],
            "d_duration": req["delivery"]["duration"],
        }

    return Q, K, depot, time_horizon, requests, t, c





# 2. REGRET-2 CONSTURCTION HEURISTIC

def _construction_heuristic(Q, K, depot, requests, t, c, R,
                             noise_scale=0.0, rng=None):
    route_events = [[] for _ in range(len(K))]

    # Per-vehicle mutable state
    vstate = [
        {
            "time": 0.0, "loc": depot, "load": 0, "dest": None,
            "batch": [], "batch_a": 0.0, "batch_b": float("inf"), "batch_s": 0.0,
        }
        for _ in range(len(K))
    ]

    def _deliver_sim(s):
        """
        Simulate sequential EDF delivery of current batch.
        """

        if s["load"] == 0:
            return s["time"], s["loc"]
        arr      = s["time"] + t(s["loc"], s["dest"])
        cur_time = arr
        for req in sorted(
            [requests[r] for r in s["batch"]],
            key=lambda r: (r["d_latest"], r["d_earliest"]),   # EDF; tie-break by earliest start
        ):
            start = max(cur_time, req["d_earliest"])
            if start > req["d_latest"]:
                return None                     # this request's window is missed
            cur_time = start + req["d_duration"]   # sequential: full duration each
        return cur_time, s["dest"]


    def _best_insert(s, rid):
        """
        Find cheapest feasible insertion of request 'rid' into state 's'.
        """

        req = requests[rid]
        p, d = req["p_node"], req["d_node"]
        best = None

        # Option A: extend current open batch (same destination)
        if (s["load"] > 0
                and s["dest"] == d
                and s["load"] + req["demand"] <= Q):
            arr_p = s["time"] + t(s["loc"], p)
            B_p   = max(arr_p, req["p_earliest"])
            if B_p <= req["p_latest"]:
                new_t  = B_p + req["p_duration"]
                new_a  = max(s["batch_a"], req["d_earliest"])
                new_b  = min(s["batch_b"], req["d_latest"])
                new_sd = s["batch_s"] + req["d_duration"]

                # Forward-feasibility: simulate sequential EDF delivery of the extended batch.
                # Note: with sequential delivery the TW intersection is not a sufficient condition
                # always delegate to _deliver_sim for the complete feasibility check.
                candidate_batch = s["batch"] + [rid]
                candidate_state = {
                    "time": new_t, "loc": p,
                    "load": s["load"] + req["demand"], "dest": d,
                    "batch": candidate_batch,
                    "batch_a": new_a, "batch_b": new_b, "batch_s": new_sd,
                }

                if _deliver_sim(candidate_state) is not None:
                    best = (c(s["loc"], p), "A", candidate_state)

        # Option B: close current batch first, then start new singleton batch
        res = _deliver_sim(s)
        if res is not None:
            after_t, after_loc = res
            arr_p = after_t + t(after_loc, p)
            B_p   = max(arr_p, req["p_earliest"])
            if B_p <= req["p_latest"]:
                new_t = B_p + req["p_duration"]
                arr_d = new_t + t(p, d)
                if max(arr_d, req["d_earliest"]) <= req["d_latest"]:
                    cost_B = c(after_loc, p)
                    ns_B = {
                        "time": new_t, "loc": p,
                        "load": req["demand"], "dest": d,
                        "batch": [rid],
                        "batch_a": req["d_earliest"],
                        "batch_b": req["d_latest"],
                        "batch_s": req["d_duration"],
                    }
                    if best is None or cost_B < best[0]:
                        best = (cost_B, "B", ns_B)

        return best


    # Regret-2 main loop
    unassigned = list(R)

    while unassigned:
        # Compute the two cheapest insertions per unassigned request
        best2: dict = {}
        for rid in unassigned:
            options = []
            for k_idx in range(len(K)):
                result = _best_insert(vstate[k_idx], rid)
                if result is not None:
                    cost, option, ns = result
                    options.append((cost, k_idx, option, ns))
            options.sort(key=lambda x: x[0])
            best2[rid] = options

        def _regret_key(rid):
            options = best2[rid]
            if not options:
                return (float("inf"), float("inf"))
            best_cost = options[0][0]
            if len(options) == 1:
                regret = float("inf")
            else:
                regret = options[1][0] - options[0][0]
            # Add small noise to diversify across runs
            if noise_scale > 0.0 and rng is not None and regret != float("inf"):
                regret += rng.uniform(-noise_scale, noise_scale)
            return (regret, best_cost)

        selection_rid = max(unassigned, key=_regret_key)
        options       = best2[selection_rid]

        if not options:
            print(f"  [HEURISTIC] No feasible vehicle for request {selection_rid}; skipping.")
            unassigned.remove(selection_rid)
            continue

        cost, k_idx, opt, ns = options[0]

        # Option B: close open batch on the chosen vehicle before inserting
        if opt == "B" and vstate[k_idx]["load"] > 0:
            for batch_request in vstate[k_idx]["batch"]:
                route_events[k_idx].append(("D", batch_request))
    
            after_t, after_loc = _deliver_sim(vstate[k_idx])
            vstate[k_idx].update({
                "time": after_t, "loc": after_loc,
                "load": 0, "dest": None,
                "batch": [], "batch_a": 0.0,
                "batch_b": float("inf"), "batch_s": 0.0,
            })

        route_events[k_idx].append(("P", selection_rid))
        vstate[k_idx].update(ns)
        unassigned.remove(selection_rid)

    # Close all still-open batches
    for k_idx in range(len(K)):
        s = vstate[k_idx]
        if s["load"] > 0:
            for batch_request in s["batch"]:
                route_events[k_idx].append(("D", batch_request))

    return route_events



# 3.  Convert heuristic route_events -> extended route format.

def _events_to_routes(route_events, requests, t, c, depot,
                       existing_best: dict):
    new_routes   = []
    seen_sets    = {}   # best cost seen among new heuristic routes.

    for events in route_events:
        if not events:
            continue

        req_set = frozenset(rid for _, rid in events if _ == "P")
        if not req_set:
            continue

        # Build physical node path.
        path = [depot]
        for ev_type, rid in events:
            node = (requests[rid]["p_node"] if ev_type == "P"
                    else requests[rid]["d_node"])
            path.append(node)
        path.append(depot)
        path = tuple(path)

        cost = 0.0
        for i in range(len(path) - 1):
            cost += c(path[i], path[i + 1])

        # Skip if a cheaper route for this request set already exists.
        if req_set in existing_best and existing_best[req_set] <= cost:
            continue

        # Among new heuristic routes, keep only the cheapest per request set.
        if req_set in seen_sets:
            if seen_sets[req_set] <= cost:
                continue
            new_routes = [r for r in new_routes if r["requests"] != req_set]

        seen_sets[req_set] = cost
        new_routes.append({"requests": req_set, "path": path, "cost": cost})

    return new_routes



# 4.  Heuristic route injection

def generate_heuristic_routes(Q, K, depot, time_horizon, requests, t, c, R,
                              n_runs=15, noise_scale=0.15,
                               existing_best=None, base_seed=42):
    if existing_best is None:
        existing_best = {}

    print(f"{'─'*60}")
    print(f"Phase 0: Heuristic route injection ({n_runs} regret-2 runs) ...")
    print(f"{'─'*60}")

    h_start      = _time.time()
    all_new      = []
    cumulative_best = dict(existing_best)      # track best cost per req-set across all runs

    for run_idx in range(n_runs):
        # Run 0: deterministic (noise_scale = 0)
        if run_idx == 0:
            rng_run      = None
            noise_run    = 0.0
            label        = "deterministic"
        # Run 1 and beyond: use a pseudo-random number generator
        else:
            rng_run      = random.Random(base_seed + run_idx)
            noise_run    = noise_scale
            label        = f"seed={base_seed + run_idx}"

        route_events = _construction_heuristic(
            Q, K, depot, requests, t, c, R,
            noise_scale=noise_run, rng=rng_run,
        )

        new_cols = _events_to_routes(
            route_events, requests, t, c, depot,
            existing_best=cumulative_best,
        )

        # Update cumulative registry so later runs can skip dominated columns
        for col in new_cols:
            s = col["requests"]
            if s not in cumulative_best or cumulative_best[s] > col["cost"]:
                cumulative_best[s] = col["cost"]

        all_new.extend(new_cols)
        print(f"  Run {run_idx + 1:2d} ({label:>20s}): "
              f"{len(new_cols):3d} new columns  "
              f"(cumulative: {len(all_new):4d})")

    h_time = _time.time() - h_start

    # Determine the largest route size generated by the heuristic
    if all_new:
        max_reqs = max(len(r["requests"]) for r in all_new)
        avg_reqs = sum(len(r["requests"]) for r in all_new) / len(all_new)
    else:
        max_reqs = avg_reqs = 0

    print(f"\nHeuristic columns added : {len(all_new):,}")
    print(f"Largest route (# reqs)  : {max_reqs}")
    print(f"Avg route size (# reqs) : {avg_reqs:.1f}")
    print(f"Heuristic time          : {h_time:.2f}s\n")

    return all_new, cumulative_best



# 5.  State-Space Route Generation (subset enumeration)

def generate_routes(
    Q,
    depot,
    time_horizon,
    requests,
    t,
    c,
    max_subset_size,
    gen_time_limit=None,
    existing_best=None,
):
    if existing_best is None:
        existing_best = {}

    R = list(requests.keys())
    routes = []
    gen_start_ts = _time.time()


    # Precompute single-request routes (initialize with heuristic)
    best_cost_for_set = dict(existing_best)

    for rid in R:
        req  = requests[rid]
        p, d = req["p_node"], req["d_node"]

        # Pickup time window
        start_p = max(t(depot, p), req["p_earliest"])
        if start_p > req["p_latest"]:
            continue

        # Delivery time window
        start_d = max(start_p + req["p_duration"] + t(p, d), req["d_earliest"])
        if start_d > req["d_latest"]:
            continue

        cost = c(depot, p) + c(p, d) + c(d, depot)
        s    = frozenset([rid])


        best_cost_for_set[s] = cost
        routes.append({
            "requests": s,
            "path":     (depot, p, d, depot),
            "cost":     cost,
        })
    
    # Dominance counter
    pruned_total = 0

    # Helper functions 
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
        # Add new point in frontier and remove any points now dominated by the new one
        dominance[state_key] = [(bt, bc) for (bt, bc) in frontier if not (bt >= time and bc >= cost)]
        dominance[state_key].append((time, cost))

    def deliver(node, time, cost, batch, dest):
        """
        Simulate sequential delivery of every request in batch at dest.
        """
        reqs     = sorted(
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
    

    # Iterate over subset sizes
    for k in range(2, max_subset_size + 1):

        for subset in itertools.combinations(R, k):      # creates all request combinations (size=k)

            # Generation time-limit check (between subsets)
            if gen_time_limit is not None:
                elapsed = _time.time() - gen_start_ts
                if elapsed >= gen_time_limit:
                    print(f"Generation time limit ({gen_time_limit:.0f}s) reached.")
                    print(f"All subsets up to size {k - 1} are completed.")
                    print(f"Routes found so far: {len(routes):,}.")
                    return routes

            S = frozenset(subset)
            UB = float("inf")
            UB_sum = sum(best_cost_for_set.get(frozenset([r]), float("inf")) for r in S)

            # Inherit any heuristic upper bound for this exact request set
            if S in best_cost_for_set:
                UB = best_cost_for_set[S]

            # INITIAL STATE
            # (cost, node, load, time, batch, active_dest, served, path)
            start = (0.0, depot, 0, 0.0, frozenset(), None, frozenset(), (depot,))
            queue = [start]
            best_route = None
            dominance = {}

            # Best-first search for subset S
            while queue:

                cost, node, load, time, batch, active_dest, served, path = heapq.heappop(queue)   # pop cheapest state
                state_key = (node, load, active_dest, batch, served)                              # used for dominance check

                # PRUNING STAGE
                # Bound 1: subset UB (tightened by any heuristic column for S)
                if cost >= UB:
                    break

                # Bound 2: subadditive single-request upper bound
                if cost > UB_sum:
                    break

                # Bound 3: Pareto-dominance check
                if is_dominated(state_key, time, cost):
                    pruned_total += 1
                    continue

                update_dominance(state_key, time, cost)


                # CASE 2: TRY ENDING ROUTE (T2)
                if served == S:
                    result = deliver(node, time, cost, batch, active_dest)
                    if not result:
                        continue
                    d_time, d_cost = result

                    end_time = d_time + t(active_dest, depot)
                    end_cost = d_cost + c(active_dest, depot)
                    end_path = path + (active_dest, depot)

                    if end_time <= time_horizon and end_cost < UB:
                        UB = end_cost
                        best_route = {
                            "requests": S,
                            "path": end_path,
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

                    # CASE 1: DEPOT -> PICKUP (T1)
                    if load == 0:
                        arrival = time + t(node, p)
                        start_t = max(arrival, req["p_earliest"])

                        if start_t <= req["p_latest"]:
                            heapq.heappush(queue, (
                                cost + c(node, p),
                                p,
                                demand,
                                start_t + req["p_duration"],
                                frozenset([rid]),
                                d,
                                served | {rid},
                                path + (p,)
                            ))
                    
                    # CASE 4 & 5: PICKUP -> PICKUP, SAME DEST 
                    elif active_dest == d:
                        arrival  = time + t(node, p)
                        start    = max(arrival, req["p_earliest"])
                        p_time   = start + req["p_duration"]
                        new_cost = cost + c(node, p)

                        # CASE 4: extend current batch (no intermediate delivery)
                        if load + demand <= Q and start <= req["p_latest"]:
                            if deliver(p, p_time, new_cost, batch | {rid}, active_dest) is not None:
                                heapq.heappush(queue, (
                                    new_cost, p, load + demand,
                                    start + req["p_duration"],
                                    batch | {rid}, active_dest,
                                    served | {rid}, path + (p,)
                                ))

                        # CASE 5: close current batch first, then pick up rid as new singleton
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
                                    served | {rid}, path + (active_dest, p)
                                ))
                    
                    # CASE 3: PICKUP -> PICKUP, DIFFERENT DEST (T3)
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
                                    path + (active_dest, p)
                                ))


            # store best route for subset once queue is empty
            if best_route:
                # Only keep if it improves on any existing route (heuristic or prior enumeration)
                S_key = best_route["requests"]
                if S_key not in best_cost_for_set or best_route["cost"] < best_cost_for_set[S_key]:
                    routes.append(best_route)
                    best_cost_for_set[S_key] = best_route["cost"]

    print(f"Dominance prunings: {pruned_total:,} routes")
    return routes



# Route-event reconstruction helper
def _reconstruct_events_1d(path, req_ids, requests):
    """
    Route enumeration algorithm only stores the physical path, not which
    request was picked up or delivered at each stop. This function recovers
    this information.

    Reconstruct the ordered event sequence using DFS to handle intra-node 
    requests and multiple visits to the same physical node correctly.
    """
    path_nodes = list(path)[1:-1]  # Exclude depots
    target_reqs = set(req_ids)

    def dfs(idx, curr_load, unserved, batch_dest):
        # Base case: processed all physical stops in the path
        if idx == len(path_nodes):
            return [] if not curr_load and not unserved else None

        curr_node = path_nodes[idx]
        
        # OPTION A: Treat this stop as a Batch Delivery
        if batch_dest is not None and curr_node == batch_dest:
            delivery_events = []
            # Deliver all in truck in EDF order
            for rid in sorted(curr_load, key=lambda r: (requests[r]["d_latest"], requests[r]["d_earliest"])):
                delivery_events.append({"node": curr_node, "type": "D", "request": rid})
            
            res = dfs(idx + 1, set(), unserved, None)
            if res is not None:
                return delivery_events + res

        # OPTION B: Treat this stop as a Single Pickup
        # Find unserved requests starting here that match the current batch destination
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
    instance_path:    str,
    mip_time_limit:   float | None = None,
    gen_time_limit:   float | None = None,
    max_subset_size:  int = 10,
    n_heuristic_runs: int = 10,
    noise_scale:      float = 0.15,
):

    print(f"\n{'='*60}")
    print(f"Instance : {instance_path}")
    print(f"{'='*60}\n")

    Q, K, depot, time_horizon, requests, t, c = parse_instance(instance_path)
    R = list(requests.keys())

    print(f"Requests          : {len(R)}")
    print(f"Vehicles          : {len(K)}")
    print(f"Capacity          : {Q}")
    print(f"Time horizon      : {time_horizon}")
    print(f"Max subset size   : {max_subset_size}")
    print(f"Heuristic runs    : {n_heuristic_runs}  (noise_scale={noise_scale})")
    print(f"Gen time limit    : {gen_time_limit if gen_time_limit else 'unlimited'}")
    print(f"MIP time limit    : {mip_time_limit if mip_time_limit else 'unlimited'}")
    print()

    overall_start = _time.time()

    heuristic_routes = []
    heuristic_best   = {}

    if n_heuristic_runs > 0:
        heuristic_routes, heuristic_best = generate_heuristic_routes(
            Q, K, depot, time_horizon, requests, t, c, R,
            n_runs=n_heuristic_runs,
            noise_scale=noise_scale,
        )

    print(f"{'─'*60}")
    print(f"Phase 1: Generating feasible 1D-PDPTW routes ...")
    print(f"Start clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}")
    print()

    gen_start = _time.time()
    enum_routes = generate_routes(
        Q, depot, time_horizon, requests, t, c,
        max_subset_size=max_subset_size,
        gen_time_limit=gen_time_limit,
        existing_best=heuristic_best,
    )
    gen_time = _time.time() - gen_start

    print(f"\nRoutes generated (enumeration) : {len(enum_routes):,}")
    print(f"Generation time                : {gen_time:.2f}s\n")

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

    m.setParam("Threads", 8) 
    m.setParam("NodefileStart", 20)

    if mip_time_limit is not None:
        print(f"Gurobi time limit : {mip_time_limit:.1f}s\n")
        m.setParam("TimeLimit", mip_time_limit)

    # Binary selection variable for each candidate route r
    lam = m.addVars(len(routes), vtype=GRB.BINARY, name="lambda")

    # Objective: minimize total travel cost
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

    # Callback: print every new incumbent solution.
    sol_count = [0]
    last_obj  = [None]

    def print_incumbent(model, where):
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

    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
        gap = m.MIPGap * 100
        print(f"\nObjective  : {m.objVal:.4f}")
        print(f"MIP gap    : {gap:.2f}%")
        print(f"Status     : "
              f"{'Optimal' if m.status == GRB.OPTIMAL else 'Time limit (best found)'}")
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
        print(f"Total time    : {total_time:.2f}s  "
              f"(heuristic={_time.time() - overall_start - gen_time - mip_time:.2f}s  "
              f"gen={gen_time:.2f}s  MIP={mip_time:.2f}s)")

    elif m.status == GRB.INFEASIBLE:
        print(f"\nModel is infeasible. Gurobi status: {m.status}")
    else:
        print(f"\nNo feasible solution found. Gurobi status: {m.status}")
    
    with open(instance_path) as f:
        raw_data = json.load(f)

    res = {
        "instance": raw_data.get("instance_name", instance_path),
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        "base_instance": raw_data.get("base_instance_name"),
        "city": raw_data.get("real_world_inspiration"),
        "loc_distr": raw_data.get("location_distribution"),
        "dep_distr": raw_data.get("depot_location_distribution"),
        "restr": raw_data.get("variant", {}).get("node_restriction_percentage"),
        "n_requests": len(raw_data.get("requests", [])),
        "n_nodes": len(raw_data.get("nodes", [])),
        "n_vehicles": raw_data.get("vehicle", {}).get("fleet_size"),
        "capacity": raw_data.get("vehicle", {}).get("capacity"),
        "model": "1dpdptw_extended",
        "mip_status": m.status,
        "mip_time": mip_time if 'mip_time' in locals() else None,
        "total_time": total_time if 'total_time' in locals() else None,
        "mip_obj": m.objVal if m.SolCount > 0 else None,
        "mip_bound": m.ObjBound if hasattr(m, "ObjBound") else None,
        "mip_gap": m.MIPGap if hasattr(m, "MIPGap") else None,
        "fleet_used": None,
        "routes": [],
        "routes_generated": len(enum_routes) if 'enum_routes' in locals() else 0,
        "heuristic_routes": len(heuristic_routes) if 'heuristic_routes' in locals() else 0,
        "gen_time": gen_time if 'gen_time' in locals() else None,
        "valid_arcs": None, "ws_cost": None, "ws_fleet": None, "ws_valid": None
    }

    # Extract routes and fleet count if a solution exists
    if m.SolCount > 0:
        fleet_count = 0
        routes_list = []
        for r in range(len(routes)):
            if lam[r].X > 0.5:
                fleet_count += 1
                rpath    = list(routes[r]["path"])
                req_ids  = routes[r]["requests"]
                events   = _reconstruct_events_1d(rpath, req_ids, requests)
                routes_list.append({
                    "path":     rpath,
                    "events":   events,
                    "requests": sorted(req_ids),
                    "cost":     routes[r]["cost"],
                })
        res["fleet_used"] = fleet_count
        res["routes"] = routes_list

    # Save to JSON
    output_dir = "res_1dextended_v2"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(instance_path).replace('.json', '')
    out_name = os.path.join(output_dir, f"res_1dextended_{base_name}.json")
    with open(out_name, "w") as f:
        json.dump(res, f, indent=4)



# 7.  Warm-start hint for the MIP

def _inject_warm_start_hint(lam, all_routes, heuristic_routes, R):
    if not heuristic_routes:
        return

    covered   = set()
    hint_idxs = set()

    for r_idx, route in enumerate(all_routes):
        if r_idx >= len(heuristic_routes):
            break                              # only consider heuristic columns
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
        print("python 1d-pdptw_extended_subsets.py <instance.json> [mip_time_limit] [gen_time_limit] [max_subset_size] [n_heuristic_runs] [noise_scale]")
        sys.exit(1)

    _instance          = sys.argv[1]
    _mip_time_limit    = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    _gen_time_limit    = float(sys.argv[3]) if len(sys.argv) >= 4 else None
    _max_subset_size   = int(sys.argv[4])   if len(sys.argv) >= 5 else 50
    _n_heuristic_runs  = int(sys.argv[5])   if len(sys.argv) >= 6 else 15
    _noise_scale       = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.15

    solve(_instance, _mip_time_limit, _gen_time_limit, _max_subset_size,
          _n_heuristic_runs, _noise_scale)