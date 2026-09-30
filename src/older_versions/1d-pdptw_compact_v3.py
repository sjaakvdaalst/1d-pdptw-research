"""
1D-PDPTW Compact MIP Formulation (no depot leg cost + non-loop route penalty)

Solves the 1D-PDPTW using a three-index compact formulation, with following specifications:
  - Depot leg costs (depot -> first stop, last stop -> depot) are excluded from the objective (arcs touching the fictitious depot_start/depot_end nodes cost 0).
  - A "non-loop route" penalty is added: for each vehicle whose route's first physical stop differs from its last physical stop, a
    penalty of `non_loop_route_penalty_ratio * cost(last_stop, first_stop)` is added to the objective. Loop routes (first == last) contribute 0 automatically since cost(x, x) = 0.

Designed for Sartori-Buriol PDPTW instances, in particular the node-restricted variants.

Brief overview of solution pipeline (see solve() ):
1. Parse JSON instance.
2. Generate valid arc set A enforcing standard and 1D-PDPTW specific rules (R1-R6).
3. Run a regret-2 construction heuristic to obtain a warm-start solution.
4. Build and solve the compact MIP.
5. Export results to a JSON results file.

Usage:
python 1d-pdptw_compact_v2_penalty.py <instance.json> [mip_time_limit_secs] [--no-warm-start]
"""

import sys
import json
import time as _time
import os
import signal
import gurobipy as gp

from gurobipy import GRB
from datetime import datetime


# SIGTERM handling: Slurm sends SIGTERM some grace period before the hard kill at wall-clock limit
_sigterm_received = [False]

def _handle_sigterm(signum, frame):
    _sigterm_received[0] = True
    print("\n[SIGTERM] Caught termination signal: will stop at next callback and export current best incumbent.", flush=True)

signal.signal(signal.SIGTERM, _handle_sigterm)



# 1. PARSE INSTANCE

def parse_instance(filepath: str):
    with open(filepath) as f:
        data = json.load(f)

    Q            = data["vehicle"]["capacity"]
    K            = list(range(data["vehicle"]["fleet_size"]))
    depot        = data.get("depot_node", data.get("depot"))
    time_horizon = data.get("time_horizon", 10 ** 6)
    T_raw        = data["travel_times"]
    C_raw        = data.get("travel_costs", data["travel_times"])
    non_loop_penalty_ratio = data.get("non_loop_route_penalty_ratio", 0.0)

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

    return Q, K, depot, time_horizon, requests, t, c, non_loop_penalty_ratio




# 2. REGRET-2 CONSTRUCTION HEURISTIC

def _route_cost(events, requests, depot, c, non_loop_penalty_ratio=0.0):
    """
    Total cost for an event sequence.
    """
    if not events:
        return 0.0

    total_cost = 0.0
    curr_loc = None
    first_loc = None

    for ev_type, rid in events:
        req = requests[rid]
        nxt_node = req["p_node"] if ev_type == "P" else req["d_node"]
        if curr_loc is not None:
            total_cost += c(curr_loc, nxt_node)
        else:
            first_loc = nxt_node
        curr_loc = nxt_node

    last_loc = curr_loc
    if first_loc != last_loc:
        total_cost += non_loop_penalty_ratio * c(last_loc, first_loc)
    return total_cost


def _is_1d_feasible(events, requests, depot, Q, time_horizon, t):
    """
    Validates following constraints:
    1. Capacity and time-window feasibility at every stop.
    2. No mixed destinations onboard.
    3. Must be empty after a delivery if the next stop is a pickup.
    4. Return to depot within time_horizon
    """

    time, loc, load = 0.0, depot, 0
    onboard_dest = None
    prev_type = None

    for ev_type, rid in events:
        req = requests[rid]
        p_node, d_node = req["p_node"], req["d_node"]

        if ev_type == "P":
            # Empty departure: must be empty before starting a new batch.
            if prev_type == "D" and load > 0:
                return False

            # 1D rule: Shared destination check
            if load > 0 and d_node != onboard_dest:
                return False

            arr = time + t(loc, p_node)
            if arr > req["p_latest"]: return False
            time = max(arr, req["p_earliest"]) + req["p_duration"]
            load += req["demand"]
            onboard_dest = d_node
            loc = p_node
        else:
            # Delivery must go to the current batch destination
            if onboard_dest is not None and d_node != onboard_dest:
                return False

            arr = time + t(loc, d_node)
            if arr > req["d_latest"]: return False
            time = max(arr, req["d_earliest"]) + req["d_duration"]
            load -= req["demand"]
            if load == 0: onboard_dest = None
            loc = d_node

        if load > Q: return False
        prev_type = ev_type

    return (time + t(loc, depot)) <= time_horizon


def _construction_heuristic(Q, K, depot, time_horizon, requests, t, c, R,
                             non_loop_penalty_ratio=0.0):
    route_events = [[] for _ in range(len(K))]

    def _route_cost_local(events, requests, depot):
        return _route_cost(events, requests, depot, c, non_loop_penalty_ratio)

    def _is_1d_feasible_local(events):
        return _is_1d_feasible(events, requests, depot, Q, time_horizon, t)


    def _evaluate_insertions(events, rid):
        """
        Find the cheapest feasible insertion of request 'rid' into 'events'.
        Returns marginal cost of the cheapest insertion and the resulting event list.
        
        This strategy differs from the extended models (see thesis Section 4.3.1)
        """
        best_cost = float('inf')
        best_route = None
        orig_cost = _route_cost_local(events, requests, depot)
        n = len(events)

        # Try all i (pickup index) and j (delivery index) where i < j
        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                test_route = events[:]
                test_route.insert(i, ("P", rid))
                test_route.insert(j, ("D", rid))

                if _is_1d_feasible_local(test_route):
                    new_cost = _route_cost_local(test_route, requests, depot)
                    cost = new_cost - orig_cost
                    if cost < best_cost:
                        best_cost, best_route = cost, test_route   # update best cost/route if cheaper
        return best_cost, best_route

    # Regret-2 main loop
    unassigned = list(R)
    while unassigned:
        best_insertions = {}
        for rid in unassigned:       # go through all unassigned requests
            options = []
            for k_idx in range(len(K)):     # go through all vehicles
                cost, b_route = _evaluate_insertions(route_events[k_idx], rid)
                if cost != float('inf'):
                    options.append((cost, k_idx, b_route))
            options.sort(key=lambda x: x[0])
            best_insertions[rid] = options

        feasible = [r for r in unassigned if best_insertions[r]]
        if not feasible:
            break 

        # select the request with the highest regret value
        selection_rid = max(feasible, key=lambda r:
            (best_insertions[r][1][0] - best_insertions[r][0][0], -best_insertions[r][0][0]) if len(best_insertions[r]) > 1
            else (float('inf'), -best_insertions[r][0][0]))
        
        # update route and remove request from unassigned set
        res = best_insertions[selection_rid][0]
        route_events[res[1]] = res[2]
        unassigned.remove(selection_rid)

    return route_events


def _heuristic_cost(route_events, requests, c, depot, non_loop_penalty_ratio=0.0):
    """
    Returns total cost of the heuristic solution
    """
    return sum(
        _route_cost(events, requests, depot, c, non_loop_penalty_ratio)
        for events in route_events
    )



# or-opt local search
# try to improve construction heuristic by trying to relocate request pairs to cheaper positions in the route post-construction

def _or_opt_pass(route_events, requests, depot, Q, time_horizon, t, c,
                  non_loop_penalty_ratio=0.0, max_passes=5):
    routes = [list(events) for events in route_events]
    n_moves = 0

    def route_cost(events):
        return _route_cost(events, requests, depot, c, non_loop_penalty_ratio)

    def feasible(events):
        return _is_1d_feasible(events, requests, depot, Q, time_horizon, t)

    for _pass in range(max_passes):
        improved_this_pass = False

        # Enumerate every currently-placed request as a (route_idx, rid) pair.
        placements = []
        for r_idx, events in enumerate(routes):
            seen = set()
            for ev_type, rid in events:
                if rid not in seen:
                    placements.append((r_idx, rid))
                    seen.add(rid)

        for src_idx, rid in placements:
            src_events = routes[src_idx]
            # Remove this request's P/D events as an atomic pair.
            stripped = [ev for ev in src_events if ev[1] != rid]
            if not feasible(stripped):
                continue  # shouldn't happen (removal only relaxes constraints), skip defensively

            src_cost_before = route_cost(src_events)
            src_cost_after_removal = route_cost(stripped)

            best_delta = 0.0  # only accept strictly improving moves
            best_dst_idx = None
            best_dst_events = None
            best_src_events_after = None

            for dst_idx, dst_events in enumerate(routes):
                base_events = stripped if dst_idx == src_idx else dst_events
                m_len = len(base_events)

                for i in range(m_len + 1):
                    for j in range(i + 1, m_len + 2):
                        candidate = base_events[:]
                        candidate.insert(i, ("P", rid))
                        candidate.insert(j, ("D", rid))

                        if not feasible(candidate):
                            continue

                        if dst_idx == src_idx:
                            # Whole-route delta vs. the original route.
                            delta = route_cost(candidate) - src_cost_before
                        else:
                            dst_cost_before = route_cost(dst_events)
                            delta = (
                                (src_cost_after_removal - src_cost_before)
                                + (route_cost(candidate) - dst_cost_before)
                            )

                        if delta < best_delta - 1e-9:
                            best_delta = delta
                            best_dst_idx = dst_idx
                            best_dst_events = candidate
                            best_src_events_after = stripped

            if best_dst_idx is not None:
                if best_dst_idx == src_idx:
                    routes[src_idx] = best_dst_events
                else:
                    routes[src_idx] = best_src_events_after
                    routes[best_dst_idx] = best_dst_events
                n_moves += 1
                improved_this_pass = True

        if not improved_this_pass:
            break

    total_cost = sum(route_cost(events) for events in routes)
    return routes, total_cost, n_moves


def _check_heuristic(route_events, R, Q, depot, time_horizon, requests, t):
    """
    Validate a heuristic solution and count 1D-PDPTW constraint violations (if applicable).
    """

    served = set()
    violations = 0

    for events in route_events:
        curr_t = 0.0
        curr_loc = depot
        curr_ld = 0
        onboard_dest = None
        prev_type = None

        for ev_type, rid in events:
            req = requests[rid]

            if ev_type == "P":
                node = req["p_node"]
                d_node = req["d_node"]

                # 1D: empty-departure
                if prev_type == "D" and curr_ld > 0:
                    violations += 1

                # 1D: destination consistency
                if curr_ld > 0 and d_node != onboard_dest:
                    violations += 1

                arr = curr_t + t(curr_loc, node)
                B = max(arr, req["p_earliest"])
                if B > req["p_latest"]:
                    violations += 1
                curr_t = B + req["p_duration"]
                curr_ld += req["demand"]
                onboard_dest = d_node
                curr_loc = node

            else:
                node = req["d_node"]

                # 1D: delivery must go to the current batch destination
                if onboard_dest is not None and node != onboard_dest:
                    violations += 1

                arr = curr_t + t(curr_loc, node)
                B = max(arr, req["d_earliest"])
                if B > req["d_latest"]:
                    violations += 1
                curr_t = B + req["d_duration"]
                curr_ld -= req["demand"]
                if curr_ld == 0:
                    onboard_dest = None
                served.add(rid)
                curr_loc = node
            
            if curr_ld < 0 or curr_ld > Q:
                violations += 1

            prev_type = ev_type
                
        if curr_t + t(curr_loc, depot) > time_horizon:
            violations += 1

    missing = set(R) - served
    return violations, missing




# 3. WARM START TRANSLATION & APPLICATION

def apply_warm_start(route_events, R, n, K, depot, requests, t,
                     A_list, x_var, B_var, L_var):
    """
    Inject a heuristic solution as a MIP warm start.
    Translates the event-sequence representation into Start values
    for the decision variables x, B, and L.
    """

    A_set = set(A_list)
    rid_to_pu = {rid: (i + 1) for i, rid in enumerate(R)}
    rid_to_de = {rid: (n + i + 1) for i, rid in enumerate(R)}
    all_valid = True

    for k_idx, k in enumerate(K):
        events = route_events[k_idx]

        if not events:
            # Empty route
            if (0, 2*n+1) in A_set:
                x_var[k, 0, 2*n+1].Start = 1
                B_var[k, 0].Start = 0
                B_var[k, 2*n+1].Start = 0
                L_var[k, 0].Start = 0
                L_var[k, 2*n+1].Start = 0
            continue

        # Build arc sequence
        curr_t = 0.0
        curr_loc = depot
        curr_ld = 0
        prev_model = 0

        B_var[k, 0].Start = 0
        L_var[k, 0].Start = 0

        for ev_type, rid in events:
            req = requests[rid]
            
            if ev_type == "P":
                model_node = rid_to_pu[rid]
                phys_node = req["p_node"]
                arr = curr_t + t(curr_loc, phys_node)
                B_start = max(arr, req["p_earliest"])
                curr_t = B_start + req["p_duration"]
                curr_ld += req["demand"]
                curr_loc = phys_node
            else:
                model_node = rid_to_de[rid]
                phys_node = req["d_node"]
                arr = curr_t + t(curr_loc, phys_node)
                B_start = max(arr, req["d_earliest"])
                curr_t = B_start + req["d_duration"]
                curr_ld -= req["demand"]
                curr_loc = phys_node

            arc = (prev_model, model_node)
            if arc in A_set:
                x_var[k, arc[0], arc[1]].Start = 1
            else:
                print(f"[WARM START] Arc {arc} not in model for vehicle {k}")
                all_valid = False

            B_var[k, model_node].Start = B_start
            L_var[k, model_node].Start = curr_ld

            prev_model = model_node

        # Final arc back to the depot.
        arc = (prev_model, 2*n+1)
        if arc in A_set:
            x_var[k, arc[0], arc[1]].Start = 1
        else:
            print(f"[WARM START] Final arc {arc} not in model for vehicle {k}")
            all_valid = False

        B_var[k, 2*n+1].Start = curr_t + t(curr_loc, depot)
        L_var[k, 2*n+1].Start = 0

    return all_valid





# 4. MAIN SOLVE FUNCTION

def solve(instance_path, mip_time_limit=None, warm_start=True):
    overall_start = _time.time()
    
    print(f"\n{'='*70}")
    print(f"1D-PDPTW COMPACT FORMULATION")
    print(f"Instance: {instance_path}")
    print(f"{'='*70}\n")

    Q, K, depot, time_horizon, requests, t, c, non_loop_penalty_ratio = parse_instance(instance_path)
    R = list(requests.keys())
    n = len(R)

    print(f"Requests               : {n}")
    print(f"Vehicles               : {len(K)}")
    print(f"Capacity               : {Q}")
    print(f"Time horizon           : {time_horizon}")
    print(f"Non-loop penalty ratio : {non_loop_penalty_ratio}")

    # Phase 1: Valid Arc Generation (see thesis Section 4.1.2)
    print(f"{'─'*60}")
    print("Phase 1: Valid arc generation (1D-PDPTW) ...")
    print(f"{'─'*60}")

    # Model node indices: 0 = depot_start, 1..n = pickups, n+1..2n = deliveries, 2n+1 = depot_end
    loc = ([depot]
           + [requests[rid]["p_node"] for rid in R]
           + [requests[rid]["d_node"] for rid in R]
           + [depot])

    a = ([0]
         + [requests[rid]["p_earliest"] for rid in R]
         + [requests[rid]["d_earliest"] for rid in R]
         + [0])

    b = ([time_horizon]
         + [requests[rid]["p_latest"] for rid in R]
         + [requests[rid]["d_latest"] for rid in R]
         + [time_horizon])

    dur = ([0]
           + [requests[rid]["p_duration"] for rid in R]
           + [requests[rid]["d_duration"] for rid in R]
           + [0])

    dem = ([0]
           + [requests[rid]["demand"] for rid in R]
           + [-requests[rid]["demand"] for rid in R]
           + [0])

    # Generate valid arcs by applying the 6 rules
    A = []
    for i in range(2 * n + 2):
        for j in range(2 * n + 2):
            # R1: No self-loops, no arcs to depot_start, no arcs from depot_end
            if i == j or j == 0 or i == 2*n+1:
                continue
            
            # R2: No depot_start -> delivery arcs
            if i == 0 and n+1 <= j <= 2*n:
                continue
            
            # R3: Pickup -> pickup with different destinations
            if 1 <= i <= n and 1 <= j <= n:
                if loc[n+i] != loc[n+j]:  # Different delivery destinations
                    continue
            
            # R4: Pickup -> delivery wrong destination
            if 1 <= i <= n and n+1 <= j <= 2*n:
                if loc[n+i] != loc[j]:  # Pickup i's destination != delivery j's location
                    continue
            
            # R5: Delivery -> delivery different locations
            if n+1 <= i <= 2*n and n+1 <= j <= 2*n:
                if loc[i] != loc[j]:  # Different physical locations
                    continue
            
            # R6: Time window reachability
            if a[i] + dur[i] + t(loc[i], loc[j]) > b[j]:
                continue
            
            A.append((i, j))

    print(f"Valid arcs: {len(A)}")


    # Phase 2: Regret-2 warm start heuristic (see thesis Section 4.3, 4.3.1)
    heur_routes = None
    heur_obj = None

    if warm_start:
        print(f"\n{'─'*60}")
        print("Phase 2: Regret-2 construction heuristic (warm start) ...")
        print(f"{'─'*60}")

        ws_start = _time.time()
        heur_routes = _construction_heuristic(
            Q, K, depot, time_horizon, requests, t, c, R,
            non_loop_penalty_ratio=non_loop_penalty_ratio,
        )
        heur_routes = sorted(heur_routes, key=lambda events: 0 if events else 1)
        ws_time = _time.time() - ws_start

        n_viol, missing = _check_heuristic(heur_routes, R, Q, depot, time_horizon, requests, t)  # possible violations or missing requests (for debugging)
        heur_obj = _heuristic_cost(heur_routes, requests, c, depot,
                                    non_loop_penalty_ratio=non_loop_penalty_ratio)  # warm start incumbent
        v_used = sum(1 for r in heur_routes if r)     # warm start fleet size

        print(f"Vehicles used                     : {v_used} / {len(K)}")
        print(f"Heuristic cost                     : {heur_obj:.4f}")
        print(f"Time window/capacity violations   : {n_viol}")
        print(f"Uncovered requests                : {len(missing)}"
              + (f"  {missing}" if missing else ""))
        print(f"Construction time                 : {ws_time:.2f}s")

        if n_viol > 0 or missing:
            print("[WARM START] Heuristic infeasible and not injected.")
        else:
            # Post-construction local search: or-opt pass that tries
            # relocating each pickup/delivery pair (within its own route or
            # to a different route) to a cheaper feasible position

            oropt_start = _time.time()
            heur_routes, oropt_obj, n_moves = _or_opt_pass(
                heur_routes, requests, depot, Q, time_horizon, t, c,
                non_loop_penalty_ratio,
            )
            oropt_time = _time.time() - oropt_start
            n_viol, missing = _check_heuristic(
                heur_routes, R, Q, depot, time_horizon, requests, t
            )
            heur_obj = oropt_obj
            v_used = sum(1 for r in heur_routes if r)

            print(f"Or-opt moves applied               : {n_moves}")
            print(f"Cost after or-opt                  : {heur_obj:.4f}")
            print(f"Or-opt time                        : {oropt_time:.2f}s")
            if n_viol > 0 or missing:
                print("[WARM START] Or-opt result infeasible -- discarding "
                      "local-search pass, reverting to pre-or-opt routes "
                      "is not implemented; treating as invalid warm start.")

    # Phase 3: MIP Formulation (see thesis Section 4.1.3)
    print(f"\n{'─'*60}")
    print("Phase 3: Solving 1D-PDPTW compact MIP ...")
    print(f"Start clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}\n")

    m = gp.Model("1D_PDPTW")
    if mip_time_limit:
        m.setParam("TimeLimit", mip_time_limit)
    
    # Gurobi Parameters
    m.setParam("PrePasses", -1)
    m.setParam("Aggregate", 1)
    m.setParam("MIPFocus", 2)
    m.setParam("CutPasses", -1)
    m.setParam("Cuts", 2)              # aggressive cut generation
    m.setParam("Symmetry", 2)          # aggressive symmetry detection (vehicle-index symmetry is the main one here)
    m.setParam("Presolve", 2)          # conservative presolve
    m.setParam("Method", 2)            # barrier algorithm for the root relaxation
    m.setParam("NodefileStart", 20)    # write MIP tree nodes to disk after exceeding 20 GB (sometimes I got an OOM crash)
    m.setParam("Threads", 8)


    # Decision variables
    x = m.addVars(K, A, vtype=GRB.BINARY, name="x")           # arc traversals
    B = m.addVars(K, range(2*n + 2), lb=0, ub=time_horizon,   # service start times
                  vtype=GRB.CONTINUOUS, name="B")
    L = m.addVars(K, range(2*n + 2), lb=0, ub=Q,
                  vtype=GRB.CONTINUOUS, name="L")             # vehicle loads

    # Tighten variable bounds
    for k in K:
        for i in range(2 * n + 2):
            B[k, i].LB = a[i]
            B[k, i].UB = b[i]
            if 1 <= i <= n:
                L[k, i].LB = dem[i]

    # Objective [4.1]

    def _arc_cost(i, j):
        if i == 0 or j == 2*n+1:     # depot arcs have zero cost
            return 0.0
        return c(loc[i], loc[j]) if j != 2*n+1 else c(loc[i], depot)

    obj = gp.quicksum(
        _arc_cost(i, j) * x[k, i, j] for k in K for (i, j) in A
    )

    # Non-loop route penalty
    # Candidate first-stop nodes (start_js) and last-stop nodes (end_is) are
    # taken from the depot arcs already present in A, same as before.
    # Exclude the trivial empty-route arc (0, 2n+1) from both sets: it
    # represents "vehicle unused", not a real first/last physical stop, and
    # must not be treated as either a start or an end candidate.
    start_arcs = [(i, j) for (i, j) in A if i == 0 and j != 2 * n + 1]        # (0, j)
    end_arcs   = [(i, j) for (i, j) in A if j == 2 * n + 1 and i != 0]        # (i, 2n+1)
    start_js = sorted({j for (_, j) in start_arcs})
    end_is   = sorted({i for (i, _) in end_arcs})


    # x[k,0,j] and x[k,i,2n+1] are both binary, so their product is
    # linearized with y[k,i,j] = x[k,0,j] * x[k,i,2n+1]. The lower-bound
    # constraint is required to force y up to 1 whenever both x-terms are 1:
    #     y[k,i,j] >= x[k,0,j] + x[k,i,2n+1] - 1

    penalty_pairs = [(i, j) for i in end_is for j in start_js if i != j]

    y = m.addVars(K, penalty_pairs, vtype=GRB.BINARY, name="y")

    for k in K:
        for (i, j) in penalty_pairs:
            m.addConstr(y[k, i, j] <= x[k, 0, j], name=f"nlp_ub1_{k}_{i}_{j}")
            m.addConstr(y[k, i, j] <= x[k, i, 2 * n + 1], name=f"nlp_ub2_{k}_{i}_{j}")
            m.addConstr(
                y[k, i, j] >= x[k, 0, j] + x[k, i, 2 * n + 1] - 1,
                name=f"nlp_lb_{k}_{i}_{j}",
            )

    penalty_obj = gp.quicksum(
        non_loop_penalty_ratio * c(loc[i], loc[j]) * y[k, i, j]
        for k in K for (i, j) in penalty_pairs
    )

    m.setObjective(obj + penalty_obj, GRB.MINIMIZE)


    # Constraints

    A_set = set(A)

    for k in K:
        # Flow conservation (every vehicle leaves depot_start exactly once) [4.2]
        m.addConstr(
            gp.quicksum(x[k, 0, j] for j in range(2*n+2) if (0, j) in A_set) == 1,
            name=f"start_{k}",
        )
        # Flow conservation (every vehicle returns to depot_end exactly once) [4.3]
        m.addConstr(
            gp.quicksum(x[k, i, 2*n+1] for i in range(2*n+2) if (i, 2*n+1) in A_set) == 1,
            name=f"end_{k}",
        )
        # Flow conservation for remaining nodes in between depot_start and depot_end [4.4]
        for v in range(1, 2 * n + 1):
            in_flow = gp.quicksum(x[k, i, v] for i in range(2*n+2) if (i, v) in A_set)
            out_flow = gp.quicksum(x[k, v, j] for j in range(2*n+2) if (v, j) in A_set)
            m.addConstr(in_flow == out_flow, name=f"flow_{k}_{v}")



    for i in range(1, n + 1):
        # Coverage (every model pickup node is visited exactly once across all vehicles) [4.5]
        m.addConstr(
            gp.quicksum(x[k, i, j] for k in K for j in range(2*n+2) if (i, j) in A_set) == 1,
            name=f"visit_{i}",
        )
        for k in K:
            visit_p = gp.quicksum(x[k, i, j] for j in range(2*n+2) if (i, j) in A_set)
            visit_d = gp.quicksum(x[k, n+i, j] for j in range(2*n+2) if (n+i, j) in A_set)

            # Pairing (pickup + delivery of the same request on the same vehicle) [4.6]
            m.addConstr(visit_p == visit_d, name=f"same_veh_{k}_{i}")

            # Precedence (pickup precedes delivery) [4.7]
            M_prec = b[i] + dur[i] + t(loc[i], loc[n+i]) - a[n+i]
            m.addConstr(
                B[k, i] + dur[i] + t(loc[i], loc[n+i])
                <= B[k, n+i] + M_prec * (1 - visit_p),
                name=f"prec_{k}_{i}",
            )

    for k in K:
        m.addConstr(L[k, 0] == 0, name=f"depot_load_{k}")
        for (i, j) in A:
            # Time propagation [4.8]
            M_time = b[i] + dur[i] + t(loc[i], loc[j]) - a[j]
            m.addConstr(
                B[k, i] + dur[i] + t(loc[i], loc[j])
                <= B[k, j] + M_time * (1 - x[k, i, j]),
                name=f"time_{k}_{i}_{j}",
            )
            # Load tracking [4.9]
            m.addConstr(
                L[k, i] + dem[j] <= L[k, j] + Q * (1 - x[k, i, j]),
                name=f"load_ub_{k}_{i}_{j}",
            )
            m.addConstr(
                L[k, i] + dem[j] >= L[k, j] - Q * (1 - x[k, i, j]),
                name=f"load_lb_{k}_{i}_{j}",
            )
            # Empty-departure constraint [4.11]
            if (n + 1 <= i <= 2 * n) and (j <= n or j == 2 * n + 1):
                m.addConstr(
                    L[k, i] <= Q * (1 - x[k, i, j]),
                    name=f"empty_dep_{k}_{i}_{j}",
                )


    # Inject warm-start values (only when the heuristic solution is fully valid)
    if warm_start and heur_routes is not None and n_viol == 0 and not missing:
        ws_ok = apply_warm_start(
            heur_routes, R, n, K, depot, requests, t, A, x, B, L
        )
        print(f"Warm start Injected. All arcs in A check: {ws_ok}")

        # Warm-start the non-loop-penalty linearization variables y[k,i,j]
        # to be consistent with the injected x-Start values: for each
        # vehicle, its actual first/last model node under the heuristic
        # route determines which single y[k,i,j] should start at 1.
        rid_to_pu = {rid: (idx + 1) for idx, rid in enumerate(R)}
        rid_to_de = {rid: (n + idx + 1) for idx, rid in enumerate(R)}
        for (i, j) in penalty_pairs:
            for k in K:
                y[k, i, j].Start = 0
        for k_idx, k in enumerate(K):
            events = heur_routes[k_idx]
            if not events:
                continue
            first_ev_type, first_rid = events[0]
            last_ev_type, last_rid = events[-1]
            start_model = rid_to_pu[first_rid] if first_ev_type == "P" else rid_to_de[first_rid]
            end_model = rid_to_pu[last_rid] if last_ev_type == "P" else rid_to_de[last_rid]
            if (end_model, start_model) in set(penalty_pairs):
                y[k, end_model, start_model].Start = 1


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

        gap     = abs(obj - bound) / (abs(obj) + 1e-10) * 100
        gap_str = f"{gap:.2f}%"

        print(f"\n[SOLUTION #{sol_count[0]}]  "
              f"obj={obj:.4f}  bound={bound:.4f}  "
              f"gap={gap_str}  t={runtime:.1f}s")

        x_vals = model.cbGetSolution([x[k, i, j] for k in K for (i, j) in A])
        x_sol  = {
            (k, i, j): v
            for (k, i, j), v in zip(
                [(k, i, j) for k in K for (i, j) in A], x_vals
            )
        }

        vehicles_printed = 0
        for k in K:
            arcs = {i: j for (i, j) in A if x_sol.get((k, i, j), 0) > 0.5}
            if not arcs or (len(arcs) == 1 and 0 in arcs and arcs[0] == 2*n+1):
                continue

            vehicles_printed += 1
            path = [0]
            while path[-1] in arcs:
                path.append(arcs[path[-1]])

            events = []
            for node in path[1:-1]:
                if 1 <= node <= n:
                    events.append(f"P{R[node - 1]}")
                elif n + 1 <= node <= 2*n:
                    events.append(f"D{R[node - n - 1]}")
            physical_path = " -> ".join(str(loc[nd]) for nd in path)
            print(f"  Veh {k}: [{', '.join(events)}]  path: {physical_path}")

        print(f"  ({vehicles_printed} vehicles active)")


    # Solve!
    mip_start = _time.time()
    m.optimize(print_incumbent)
    mip_time = _time.time() - mip_start


    # Print solution summary
    if m.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED) and m.SolCount > 0:
        gap = m.MIPGap * 100
        _status_labels = {
            GRB.OPTIMAL: "Optimal",
            GRB.TIME_LIMIT: "Time limit (best found)",
            GRB.INTERRUPTED: "SIGTERM checkpoint (best found)",
        }
        print(f"\nObjective  : {m.objVal:.4f}"
              + (f"  (heuristic: {heur_obj:.4f})" if heur_obj is not None else ""))
        print(f"MIP gap    : {gap:.2f}%")
        print(f"Status     : {_status_labels.get(m.status, m.status)}")
        print(f"MIP time   : {mip_time:.2f}s")

        vehicles_used = 0
        for k in K:
            active_arcs = [(i, j) for (i, j) in A if x[k, i, j].X > 0.5]
            if not active_arcs or (len(active_arcs) == 1
                                   and active_arcs[0] == (0, 2 * n + 1)):
                continue

            vehicles_used += 1

            # Reconstruct event sequence for output display
            curr = 0
            seq_nodes = [depot]
            seq_events = []
            while curr != 2 * n + 1:
                nxt = next(j for (i, j) in active_arcs if i == curr)
                seq_nodes.append(loc[nxt])
                if 1 <= nxt <= n:
                    seq_events.append(f"Pickup {R[nxt - 1]}")
                elif n + 1 <= nxt <= 2 * n:
                    seq_events.append(f"Dropoff {R[nxt - n - 1]}")
                else:
                    seq_events.append("End")
                curr = nxt

            path_str = " -> ".join(str(nd) for nd in seq_nodes)
            event_str = ", ".join(seq_events[:-1])

            print(f"\n  Vehicle {vehicles_used}:")
            print(f"    Path   : {path_str}")
            print(f"    Events : [{event_str}]")

        print(f"\nVehicles used : {vehicles_used} / {len(K)}")
        print(f"MIP time      : {mip_time:.2f}s")

    elif m.status == GRB.INFEASIBLE:
        print(f"\nModel is infeasible. Gurobi status: {m.status}")
    else:
        print(f"\nNo feasible solution found. Gurobi status: {m.status}")
    


    # Export results to JSON file
    overall_time = _time.time() - overall_start
    
    with open(instance_path) as f:
        raw_data = json.load(f)    # meta data

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
        "model": "1dpdptw_compact (no depot cost, non-loop penalty)",
        "non_loop_route_penalty_ratio": non_loop_penalty_ratio,
        "mip_status": m.status,
        "mip_time": mip_time if 'mip_time' in locals() else None,
        "total_time": overall_time,
        "mip_obj": m.objVal if m.SolCount > 0 else None,
        "mip_obj_penalty_component": (
            sum(non_loop_penalty_ratio * c(loc[i], loc[j]) * y[k, i, j].X
                for k in K for (i, j) in penalty_pairs)
            if m.SolCount > 0 else None
        ),
        "mip_bound": m.ObjBound if hasattr(m, "ObjBound") else None,
        "mip_gap": m.MIPGap if hasattr(m, "MIPGap") else None,
        "fleet_used": None,
        "valid_arcs": len(A) if 'A' in locals() else None,
        "ws_cost": heur_obj if 'heur_obj' in locals() else None,
        "ws_fleet": sum(1 for r in heur_routes if r) if ('heur_routes' in locals() and heur_routes is not None) else None,
        "ws_valid": (n_viol == 0) if 'n_viol' in locals() else None,
        "routes": [],
    }

    # Extract routes and fleet count if a solution exists
    if m.SolCount > 0:
        fleet_count = 0
        routes_list = []
        for k in K:
            active_arcs = [(i, j) for (i, j) in A if x[k, i, j].X > 0.5]
            if not active_arcs or (len(active_arcs) == 1 and active_arcs[0] == (0, 2 * n + 1)):
                continue
            fleet_count += 1

            # Go through route and record (type, physical node, service start time, request, deadline) for every pickup/delivery event in visiting order.
            curr = 0
            raw_events = []
            while curr != 2 * n + 1:
                nxt  = next(j for (i, j) in active_arcs if i == curr)
                phys = loc[nxt]
                if 1 <= nxt <= n:
                    rid = R[nxt - 1]
                    raw_events.append({
                        "type": "P", "node": phys, "time": B[k, nxt].X,
                        "request": rid, "deadline": requests[rid]["d_latest"],
                    })
                elif n + 1 <= nxt <= 2 * n:
                    rid = R[nxt - n - 1]
                    raw_events.append({
                        "type": "D", "node": phys, "time": B[k, nxt].X,
                        "request": rid, "deadline": requests[rid]["d_latest"],
                    })
                curr = nxt

            grouped_events = []
            for ev in raw_events:
                if ev["type"] == "P":
                    grouped_events.append({
                        "type": "P", "node": ev["node"], "time": ev["time"],
                        "requests": [ev["request"]], "deadline": ev["deadline"],
                    })
                else:
                    if grouped_events and grouped_events[-1]["type"] == "D":
                        grouped_events[-1]["requests"].append(ev["request"])
                        grouped_events[-1]["deadline"] = min(grouped_events[-1]["deadline"], ev["deadline"])
                    else:
                        grouped_events.append({
                            "type": "D", "node": ev["node"], "time": ev["time"],
                            "requests": [ev["request"]], "deadline": ev["deadline"],
                        })

            def _fmt(val):
                # Strip floating-point solver noise; print as int when whole-numbered.
                r = round(val, 4)
                return str(int(r)) if abs(r - round(r)) < 1e-6 else f"{r:g}"

            route_str = "".join(
                f"{ev['type']}([{', '.join(str(r) for r in ev['requests'])}] "
                f"@{_fmt(ev['node'])} t={_fmt(ev['time'])}/{_fmt(ev['deadline'])})"
                for ev in grouped_events
            )

            routes_list.append(route_str)
        res["fleet_used"] = fleet_count
        res["routes"] = routes_list

    # Save to JSON

    run_date = datetime.now().strftime("%Y-%m-%d")
    base_name = os.path.basename(instance_path).replace('.json', '')

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "out_1d_pdptw_cm",
        f"out_1d_pdptw_cm_{run_date}",
    )
    os.makedirs(output_dir, exist_ok=True)

    out_name = os.path.join(
        output_dir, f"out_1dpdptw_cm_{run_date}_{base_name}.json"
    )
    with open(out_name, "w") as f:
        json.dump(res, f, indent=4)

    print(f"\nResults written to: {out_name}")


# 5. ENTRY POINT
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python 1d-pdptw_compact_v2_penalty.py <instance.json> [mip_time_limit_secs] [--no-warm-start]")
        sys.exit(1)

    _instance = sys.argv[1]
    _mip_time_limit = float(sys.argv[2]) if len(sys.argv) >= 3 and not sys.argv[2].startswith("--") else None
    _warm_start = "--no-warm-start" not in sys.argv

    solve(_instance, _mip_time_limit, _warm_start)