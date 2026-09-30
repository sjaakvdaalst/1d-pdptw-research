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
4. Build and solve the set-partitioning MIP.
5. Export results to a JSON results file (same structure as the compact
   model's output).

Usage:
python 1d-pdptw_extended_v2.py <instance.json> [mip_time_limit] [gen_time_limit] [max_subset_size] [n_heuristic_runs] [noise_scale]
"""


import sys
import json
import os
import random
import signal
import time as _time
import heapq
import itertools
import gurobipy as gp

from gurobipy import GRB
from datetime import datetime


# SIGTERM handling: Slurm sends SIGTERM some grace period before the hard kill at the wall-clock limit.
_sigterm_received = [False]

def _handle_sigterm(signum, frame):
    _sigterm_received[0] = True
    print("\n[SIGTERM] Caught termination signal -- will stop at next "
          "callback poll and export current best incumbent.", flush=True)

signal.signal(signal.SIGTERM, _handle_sigterm)


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


# 2.  REGRET-2 CONSTRUCTION HEURISTIC (ported from the compact model)


def _route_cost(events, requests, depot, c, non_loop_penalty_ratio=0.0):
    """
    Total cost for an event sequence
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
    Shared by the construction heuristic and the or-opt local search pass.
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
                             non_loop_penalty_ratio=0.0,
                             noise_scale=0.0, rng=None):
    """
    Strong regret-2 construction
    """
    route_events = [[] for _ in range(len(K))]

    def _route_cost_local(events):
        return _route_cost(events, requests, depot, c, non_loop_penalty_ratio)

    def _is_1d_feasible_local(events):
        return _is_1d_feasible(events, requests, depot, Q, time_horizon, t)

    def _evaluate_insertions(events, rid):
        """
        Find the cheapest feasible insertion of request 'rid' into 'events'.
        Returns marginal cost of the cheapest insertion and the resulting
        event list.
        """
        best_cost = float('inf')
        best_route = None
        orig_cost = _route_cost_local(events)
        n = len(events)

        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                test_route = events[:]
                test_route.insert(i, ("P", rid))
                test_route.insert(j, ("D", rid))

                if _is_1d_feasible_local(test_route):
                    new_cost = _route_cost_local(test_route)
                    cost = new_cost - orig_cost
                    if cost < best_cost:
                        best_cost, best_route = cost, test_route
        return best_cost, best_route

    # Regret-2 main loop
    unassigned = list(R)
    while unassigned:
        best_insertions = {}
        for rid in unassigned:
            options = []
            for k_idx in range(len(K)):
                cost, b_route = _evaluate_insertions(route_events[k_idx], rid)
                if cost != float('inf'):
                    options.append((cost, k_idx, b_route))
            options.sort(key=lambda x: x[0])
            best_insertions[rid] = options

        feasible = [r for r in unassigned if best_insertions[r]]
        if not feasible:
            break

        def _regret_key(rid):
            options = best_insertions[rid]
            best_cost = options[0][0]
            if len(options) > 1:
                regret = options[1][0] - options[0][0]
            else:
                regret = float('inf')
            # Add small noise to diversify across runs
            if noise_scale > 0.0 and rng is not None and regret != float('inf'):
                regret += rng.uniform(-noise_scale, noise_scale)
            return (regret, -best_cost)

        selection_rid = max(feasible, key=_regret_key)

        # update route and remove request from unassigned set
        res = best_insertions[selection_rid][0]
        route_events[res[1]] = res[2]
        unassigned.remove(selection_rid)

    return route_events


def _heuristic_cost(route_events, requests, c, depot, non_loop_penalty_ratio=0.0):
    """
    Total cost of the heuristic solution, no depot legs + non-loop penalty.
    """
    return sum(
        _route_cost(events, requests, depot, c, non_loop_penalty_ratio)
        for events in route_events
    )


def _check_heuristic(route_events, R, Q, depot, time_horizon, requests, t):
    """
    Validate a heuristic solution and count 1D-PDPTW constraint violations.
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

                if prev_type == "D" and curr_ld > 0:
                    violations += 1
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


# or-opt pass
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

        placements = []
        for r_idx, events in enumerate(routes):
            seen = set()
            for ev_type, rid in events:
                if rid not in seen:
                    placements.append((r_idx, rid))
                    seen.add(rid)

        for src_idx, rid in placements:
            src_events = routes[src_idx]
            stripped = [ev for ev in src_events if ev[1] != rid]
            if not feasible(stripped):
                continue

            src_cost_before = route_cost(src_events)
            src_cost_after_removal = route_cost(stripped)

            best_delta = 0.0
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


# 3.  Convert heuristic route_events -> extended route (column) format.

def _events_to_routes(route_events, requests, depot, c, non_loop_penalty_ratio,
                       existing_best: dict):
    new_routes = []
    seen_sets  = {}   # best cost seen among new heuristic routes this run

    for events in route_events:
        if not events:
            continue

        req_set = frozenset(rid for _, rid in events if _ == "P")
        if not req_set:
            continue

        cost = _route_cost(events, requests, depot, c, non_loop_penalty_ratio)

        # Physical path (depot -> ... -> depot), kept for display/export
        # consistency with the enumerated routes' "path" field.
        path = [depot]
        for ev_type, rid in events:
            node = (requests[rid]["p_node"] if ev_type == "P"
                    else requests[rid]["d_node"])
            path.append(node)
        path.append(depot)
        path = tuple(path)

        # Skip if a cheaper route for this request set already exists.
        if req_set in existing_best and existing_best[req_set] <= cost:
            continue

        if req_set in seen_sets:
            if seen_sets[req_set] <= cost:
                continue
            new_routes = [r for r in new_routes if r["requests"] != req_set]

        seen_sets[req_set] = cost
        new_routes.append({
            "requests": req_set, "path": path, "cost": cost,
            "events": list(events),
        })

    return new_routes


# 4.  Heuristic route injection

def generate_heuristic_routes(Q, K, depot, time_horizon, requests, t, c, R,
                               non_loop_penalty_ratio=0.0,
                               n_runs=15, noise_scale=0.15,
                               existing_best=None, base_seed=42):
    if existing_best is None:
        existing_best = {}

    print(f"{'─'*60}")
    print(f"Phase 0: Heuristic route injection ({n_runs} regret-2 runs) ...")
    print(f"{'─'*60}")

    h_start = _time.time()
    all_new = []
    cumulative_best = dict(existing_best)

    for run_idx in range(n_runs):
        if run_idx == 0:
            rng_run   = None
            noise_run = 0.0
            label     = "deterministic"
        else:
            rng_run   = random.Random(base_seed + run_idx)
            noise_run = noise_scale
            label     = f"seed={base_seed + run_idx}"

        route_events = _construction_heuristic(
            Q, K, depot, time_horizon, requests, t, c, R,
            non_loop_penalty_ratio=non_loop_penalty_ratio,
            noise_scale=noise_run, rng=rng_run,
        )

        n_viol, missing = _check_heuristic(
            route_events, R, Q, depot, time_horizon, requests, t
        )
        if n_viol > 0 or missing:
            print(f"  Run {run_idx + 1:2d} ({label:>20s}): "
                  f"infeasible construction ({n_viol} violations, "
                  f"{len(missing)} missing) -- skipping or-opt, using raw "
                  f"construction output as-is for column extraction.")
        else:
            route_events, _oropt_cost, _n_moves = _or_opt_pass(
                route_events, requests, depot, Q, time_horizon, t, c,
                non_loop_penalty_ratio,
            )

        new_cols = _events_to_routes(
            route_events, requests, depot, c, non_loop_penalty_ratio,
            existing_best=cumulative_best,
        )

        for col in new_cols:
            s = col["requests"]
            if s not in cumulative_best or cumulative_best[s] > col["cost"]:
                cumulative_best[s] = col["cost"]

        all_new.extend(new_cols)
        print(f"  Run {run_idx + 1:2d} ({label:>20s}): "
              f"{len(new_cols):3d} new columns  "
              f"(cumulative: {len(all_new):4d})")

    h_time = _time.time() - h_start

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
    non_loop_penalty_ratio=0.0,
    gen_time_limit=None,
    existing_best=None,
):
    if existing_best is None:
        existing_best = {}

    R = list(requests.keys())
    routes = []
    gen_start_ts = _time.time()

    # Precompute single-request routes (initialize with heuristic).
    best_cost_for_set = dict(existing_best)

    for rid in R:
        req  = requests[rid]
        p, d = req["p_node"], req["d_node"]

        start_p = max(t(depot, p), req["p_earliest"])
        if start_p > req["p_latest"]:
            continue

        start_d = max(start_p + req["p_duration"] + t(p, d), req["d_earliest"])
        if start_d > req["d_latest"]:
            continue

        cost = c(p, d)
        if p != d:
            cost += non_loop_penalty_ratio * c(d, p)
        s = frozenset([rid])

        best_cost_for_set[s] = cost
        routes.append({
            "requests": s,
            "path":     (depot, p, d, depot),
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

            S = frozenset(subset)
            UB = float("inf")
            UB_sum = sum(best_cost_for_set.get(frozenset([r]), float("inf")) for r in S)

            if S in best_cost_for_set:
                UB = best_cost_for_set[S]

            # INITIAL STATE
            # (cost, node, load, time, batch, active_dest, served, path,
            #  first_loc)
            # first_loc tracks the route's first physical stop (the node
            # visited right after leaving the depot), needed at the
            # terminal state to decide whether the non-loop penalty
            # applies (first_loc != active_dest on close-out).
            start = (0.0, depot, 0, 0.0, frozenset(), None, frozenset(), (depot,), None)
            queue = [start]
            best_route = None
            dominance = {}

            while queue:

                cost, node, load, time, batch, active_dest, served, path, first_loc = heapq.heappop(queue)
                state_key = (node, load, active_dest, batch, served)

                if cost >= UB:
                    break
                if cost > UB_sum:
                    break
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
                    end_cost = d_cost
                    if first_loc is not None and first_loc != active_dest:
                        end_cost += non_loop_penalty_ratio * c(active_dest, first_loc)
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
                            # Depot -> first stop is free.
                            new_first_loc = first_loc if first_loc is not None else p
                            heapq.heappush(queue, (
                                cost,
                                p,
                                demand,
                                start_t + req["p_duration"],
                                frozenset([rid]),
                                d,
                                served | {rid},
                                path + (p,),
                                new_first_loc,
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
                                    served | {rid}, path + (p,), first_loc
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
                                    served | {rid}, path + (active_dest, p), first_loc
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
    path_nodes = list(path)[1:-1]  # Exclude depots
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

    overall_start = _time.time()

    Q, K, depot, time_horizon, requests, t, c, non_loop_penalty_ratio = parse_instance(instance_path)
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
    print()

    heuristic_routes = []
    heuristic_best   = {}

    ws_cost, ws_fleet, ws_valid = None, None, None

    if n_heuristic_runs > 0:
        heuristic_routes, heuristic_best = generate_heuristic_routes(
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
            det_events, _c, _m = _or_opt_pass(
                det_events, requests, depot, Q, time_horizon, t, c,
                non_loop_penalty_ratio,
            )
            ws_cost = _heuristic_cost(det_events, requests, c, depot, non_loop_penalty_ratio)
            ws_fleet = sum(1 for r in det_events if r)

    print(f"{'─'*60}")
    print(f"Phase 1: Generating feasible 1D-PDPTW routes ...")
    print(f"Start clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}")
    print()

    gen_start = _time.time()
    enum_routes, enumeration_complete, max_subset_size_reached = generate_routes(
        Q, depot, time_horizon, requests, t, c,
        max_subset_size=max_subset_size,
        non_loop_penalty_ratio=non_loop_penalty_ratio,
        gen_time_limit=gen_time_limit,
        existing_best=heuristic_best,
    )
    gen_time = _time.time() - gen_start

    print(f"\nRoutes generated (enumeration) : {len(enum_routes):,}")
    print(f"Generation time                 : {gen_time:.2f}s")
    print(f"Enumeration complete             : {enumeration_complete}"
          + ("" if enumeration_complete else
             f"  (only fully covered subset sizes up to {max_subset_size_reached} "
             f"of configured max {max_subset_size} -- objective is optimal only "
             f"relative to this incomplete candidate pool, NOT a proven global "
             f"optimum)"))
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

    m.setParam("Threads", 8)
    m.setParam("NodefileStart", 20)

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
        "model": "1dpdptw_extended (no depot cost, non-loop penalty)",
        "non_loop_route_penalty_ratio": non_loop_penalty_ratio,
        "mip_status": m.status,
        "mip_time": mip_time if 'mip_time' in locals() else None,
        "total_time": total_time,
        "mip_obj": m.objVal if m.SolCount > 0 else None,
        "mip_bound": m.ObjBound if hasattr(m, "ObjBound") else None,
        "mip_gap": m.MIPGap if hasattr(m, "MIPGap") else None,
        "fleet_used": None,
        "valid_arcs": None,
        "ws_cost": ws_cost,
        "ws_fleet": ws_fleet,
        "ws_valid": ws_valid,
        "routes_generated": len(enum_routes) if 'enum_routes' in locals() else 0,
        "heuristic_routes": len(heuristic_routes) if 'heuristic_routes' in locals() else 0,
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

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "out_1d_pdptw_ex",
        f"out_1d_pdptw_ex_{run_date}",
    )
    os.makedirs(output_dir, exist_ok=True)

    out_name = os.path.join(
        output_dir, f"out_1dpdptw_ex_{run_date}_{base_name}.json"
    )
    with open(out_name, "w") as f:
        json.dump(res, f, indent=4)

    print(f"\nResults written to: {out_name}")


def _simulate_route_times(raw_events, requests, depot, t):
    """
    Re-simulate a route's schedule to attach an actual feasible service
    start time to each event, using the exact same sequential logic as
    _is_1d_feasible (arrival = max(prev_time + travel, earliest), then
    + duration). The extended model's routes are pre-built columns with
    no MIP timing variables, so -- unlike the compact model, which reads
    B[k,i].X straight off the solved model -- these times are recomputed
    here purely for reporting; they are not decision variables and did
    not influence which route was chosen (that was already validated
    feasible during construction/enumeration).
    """
    time, loc = 0.0, depot
    timed_events = []
    for ev in raw_events:
        rid = ev["request"]
        req = requests[rid]
        ev_type = ev["type"]
        node = req["p_node"] if ev_type == "P" else req["d_node"]
        earliest = req["p_earliest"] if ev_type == "P" else req["d_earliest"]
        latest = req["p_latest"] if ev_type == "P" else req["d_latest"]
        duration = req["p_duration"] if ev_type == "P" else req["d_duration"]

        arr = time + t(loc, node)
        start = max(arr, earliest)
        time = start + duration
        loc = node

        timed_events.append({
            "type": ev_type, "node": node, "request": rid,
            "time": start, "deadline": latest,
        })
    return timed_events


def _group_events_for_export(raw_events, requests, depot, t):
    """
    Re-simulate the route's schedule (see _simulate_route_times), then
    group consecutive delivery events into a single batch entry, matching
    the compact model's route_str export exactly: same grouping logic,
    same @node t=start/deadline suffix format.
    """
    timed_events = _simulate_route_times(raw_events, requests, depot, t)

    grouped_events = []
    for ev in timed_events:
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
    return grouped_events


def _format_route_string(grouped_events, route_cost):

    def _fmt(val):
        r = round(val, 4)
        return str(int(r)) if abs(r - round(r)) < 1e-6 else f"{r:g}"

    body = "".join(
        f"{ev['type']}([{', '.join(str(r) for r in ev['requests'])}] "
        f"@{_fmt(ev['node'])} t={_fmt(ev['time'])}/{_fmt(ev['deadline'])})"
        for ev in grouped_events
    )
    return f"{body} (Cost: {_fmt(route_cost)})"


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
        print("python 1d-pdptw_extended_v2.py <instance.json> [mip_time_limit] [gen_time_limit] [max_subset_size] [n_heuristic_runs] [noise_scale]")
        sys.exit(1)

    _instance          = sys.argv[1]
    _mip_time_limit    = float(sys.argv[2]) if len(sys.argv) >= 3 else None
    _gen_time_limit    = float(sys.argv[3]) if len(sys.argv) >= 4 else None
    _max_subset_size   = int(sys.argv[4])   if len(sys.argv) >= 5 else 50
    _n_heuristic_runs  = int(sys.argv[5])   if len(sys.argv) >= 6 else 15
    _noise_scale       = float(sys.argv[6]) if len(sys.argv) >= 7 else 0.15

    solve(_instance, _mip_time_limit, _gen_time_limit, _max_subset_size,
          _n_heuristic_runs, _noise_scale)