"""
1D-PDPTW Compact MIP Formulation (no depot leg cost + non-loop route penalty)

Solves the 1D-PDPTW using a three-index compact formulation, with the following
specifications:
  - Depot leg costs (depot -> first stop, last stop -> depot) are excluded from
    the objective (arcs touching the fictitious depot_start/depot_end nodes cost 0).
  - A "non-loop route" penalty is added: for each vehicle whose route's first
    physical stop differs from its last physical stop, a penalty of
    `non_loop_route_penalty_ratio * cost(last_stop, first_stop)` is added to the
    objective. Loop routes (first == last) contribute 0 automatically since
    cost(x, x) = 0.
  - Vehicle-fleet symmetry is broken with a valid lexicographic ordering
    constraint (see `add_symmetry_breaking_constraints`).

Designed for Sartori-Buriol PDPTW instances, in particular the node-restricted
variants.

Brief overview of solution pipeline (see solve()):
1. Parse JSON instance.
2. Generate valid arc set A enforcing standard and 1D-PDPTW specific rules (R1-R6).
3. Run a regret-2 construction heuristic + or-opt local search to obtain a warm start.
4. Build and solve the compact MIP (optionally fleet-capped, with symmetry breaking
   and a chosen Gurobi MIP setting mode).
5. Export results to a JSON results file.

Usage:
    python 1d-pdptw_compact_v4.py <instance.json> <mip_time_limit_secs> [mip_setting_mode] [fleet_buffer] [--no-warm-start] [--single-tw]

    mip_time_limit_secs : MIP time limit in seconds. Pass 0 or a negative number
                           for no time limit.
    mip_setting_mode     : 0 = standard (default), 1 = focus on closing bounds,
                           2 = focus on finding new incumbents. See
                           `MIP_SETTING_MODES` below for the exact Gurobi
                           parameters each mode sets.
    fleet_buffer         : Non-negative integer. If given, the MIP's fleet size
                           is capped at (warm-start fleet size + fleet_buffer).
                           If omitted, no fleet cap is applied (any fleet size
                           up to len(K) remains possible). Requires a valid
                           warm start to be found; ignored (with a warning) if
                           none is available.

    A leading "--no-warm-start" flag may be passed anywhere on the command line
    to disable the construction heuristic / or-opt warm start (in which case
    fleet_buffer, if given, is also ignored).
"""

import sys
import json
import time as _time
import os
import signal
import gurobipy as gp

from gurobipy import GRB
from datetime import datetime
from common.instance import parse_instance
from common.paths import results_dir
from common.feasibility import route_cost as _route_cost, is_1d_feasible as _is_1d_feasible, check_heuristic as _check_heuristic
from common.local_search import or_opt_pass as _or_opt_pass
from common.mip_settings import COMPACT_MIP_MODES as MIP_SETTING_MODES, apply_mip_mode


# ──────────────────────────────────────────────────────────────────────────
# SIGTERM handling: Slurm sends SIGTERM some grace period before the hard
# kill at the wall-clock limit.
# ──────────────────────────────────────────────────────────────────────────
_sigterm_received = [False]


def _handle_sigterm(signum, frame):
    _sigterm_received[0] = True
    print("\n[SIGTERM] Caught termination signal: will stop at next callback "
          "and export current best incumbent.", flush=True)


signal.signal(signal.SIGTERM, _handle_sigterm)


# MIP setting modes, route feasibility/cost, or-opt and instance parsing live in
# src/common/ (shared with the extended solver).


# ──────────────────────────────────────────────────────────────────────────
# 2. REGRET-2 CONSTRUCTION HEURISTIC
# ──────────────────────────────────────────────────────────────────────────


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
        Returns the marginal cost of the cheapest insertion and the resulting
        event list.

        This strategy differs from the extended models (see thesis Section 4.3.1).
        """
        best_cost = float('inf')
        best_route = None
        orig_cost = _route_cost_local(events, requests, depot)
        n = len(events)

        # Try all i (pickup index) and j (delivery index) where i < j.
        for i in range(n + 1):
            for j in range(i + 1, n + 2):
                test_route = events[:]
                test_route.insert(i, ("P", rid))
                test_route.insert(j, ("D", rid))

                if _is_1d_feasible_local(test_route):
                    new_cost = _route_cost_local(test_route, requests, depot)
                    cost = new_cost - orig_cost
                    if cost < best_cost:
                        best_cost, best_route = cost, test_route
        return best_cost, best_route

    # Regret-2 main loop.
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

        # Select the request with the highest regret value.
        selection_rid = max(feasible, key=lambda r:
            (best_insertions[r][1][0] - best_insertions[r][0][0], -best_insertions[r][0][0]) if len(best_insertions[r]) > 1
            else (float('inf'), -best_insertions[r][0][0]))

        # Update route and remove request from unassigned set.
        res = best_insertions[selection_rid][0]
        route_events[res[1]] = res[2]
        unassigned.remove(selection_rid)

    return route_events


def _heuristic_cost(route_events, requests, c, depot, non_loop_penalty_ratio=0.0):
    """Returns total cost of the heuristic solution."""
    return sum(
        _route_cost(events, requests, depot, c, non_loop_penalty_ratio)
        for events in route_events
    )


# ──────────────────────────────────────────────────────────────────────────
# Or-opt local search: try to improve the construction heuristic by relocating
# request pairs to cheaper positions in the route (own route or another route).
# ──────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────
# 3. WARM START TRANSLATION & APPLICATION
# ──────────────────────────────────────────────────────────────────────────

def _route_sigma_key(events, rid_to_pu, n):
    """
    Sort key for a single vehicle's heuristic route, mirroring the MIP-side
    symmetry-breaking expression sigma_k exactly (see
    `add_symmetry_breaking_constraints`): the model-node index of the
    route's first pickup, or the sentinel n+1 if the route is empty or
    starts with something other than a pickup (shouldn't happen for a
    feasible 1D-PDPTW route, but guarded defensively).

    Used to reorder heuristic routes across vehicle slots before injecting
    them as a MIP start, so the resulting Start assignment satisfies the
    symmetry chain sigma_0 <= sigma_1 <= ... by construction. Without this,
    Gurobi silently rejects any MIP start that violates the symmetry
    constraints -- it does not partially accept or repair one.
    """
    if not events:
        return n + 1
    first_ev_type, first_rid = events[0]
    if first_ev_type != "P":
        # Should not occur for a feasible 1D-PDPTW route (every route must
        # start with a pickup), but fall back to the idle sentinel rather
        # than crashing if it ever does.
        return n + 1
    return rid_to_pu[first_rid]


def sort_routes_for_symmetry(route_events, R, n):
    """
    Reorder a list of per-vehicle heuristic routes (one list of (type, rid)
    events per vehicle slot) into the canonical order required by the
    vehicle symmetry-breaking chain: non-decreasing sigma_k, i.e. by first
    pickup's model-node index, with empty routes (sigma = n+1) sorted last.

    This must be applied to the FINAL route list (after any local search
    such as or-opt, which can change which route starts with which request)
    and immediately before `apply_warm_start` -- not just once after
    construction -- otherwise the injected start can violate the symmetry
    constraints and Gurobi will discard it outright ("User MIP start did
    not produce a new incumbent solution").
    """
    rid_to_pu = {rid: (i + 1) for i, rid in enumerate(R)}
    return sorted(route_events, key=lambda events: _route_sigma_key(events, rid_to_pu, n))


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
            # Empty route.
            if (0, 2 * n + 1) in A_set:
                x_var[k, 0, 2 * n + 1].Start = 1
                B_var[k, 0].Start = 0
                B_var[k, 2 * n + 1].Start = 0
                L_var[k, 0].Start = 0
                L_var[k, 2 * n + 1].Start = 0
            continue

        # Build arc sequence.
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
        arc = (prev_model, 2 * n + 1)
        if arc in A_set:
            x_var[k, arc[0], arc[1]].Start = 1
        else:
            print(f"[WARM START] Final arc {arc} not in model for vehicle {k}")
            all_valid = False

        B_var[k, 2 * n + 1].Start = curr_t + t(curr_loc, depot)
        L_var[k, 2 * n + 1].Start = 0

    return all_valid


# ──────────────────────────────────────────────────────────────────────────
# 4. SYMMETRY BREAKING (target #3)
# ──────────────────────────────────────────────────────────────────────────

def add_symmetry_breaking_constraints(m, K, A_set, x, n):
    """
    Break the full vehicle-index symmetry (S_|K|) of the compact formulation.

    Why it exists: every vehicle k in K has identical capacity, identical
    depot time window, and identical objective coefficients -- there is no
    per-vehicle data anywhere in the model (parse_instance / the arc set A
    depend only on the instance, never on k). Consequently, for ANY feasible
    solution and ANY permutation pi of K, relabeling every variable
    x[k,i,j] -> x[pi(k),i,j] (and B, L, and the non-loop penalty y variables
    along with it) produces another feasible solution of identical cost.
    With many structurally-interchangeable/unused vehicles (the case this
    symmetry-breaking mode is aimed at) this blows up the number of
    equal-cost incumbents Gurobi has to distinguish between, which is a
    well known cause of slow bound closure.

    Construction: define, for each vehicle k, the linear expression

        sigma_k = sum_{i=1..n} i * x[k,0,i]  +  (n+1) * x[k,0,2n+1]

    Every vehicle has exactly one out-arc from the depot-start node 0 (see
    constraint `start_{k}`), and by arc-generation rule R2 that arc goes
    either to some pickup node i in {1,...,n}, or directly to the
    depot-end sink 2n+1 (empty route) -- never to a delivery node. So
    sigma_k evaluates to exactly one value in {1, ..., n, n+1}: the index
    of the first pickup node vehicle k visits, or the sentinel n+1 if the
    vehicle is unused. sigma_k needs no new variables; it is already a
    linear combination of existing x variables.

    We add the chain of constraints

        sigma_0 <= sigma_1 <= ... <= sigma_{|K|-1}

    This is VALID: given any feasible solution, sort the vehicles by their
    sigma value (any total order refining sigma works, e.g. break ties by
    original index) and relabel. Because vehicles are fully interchangeable,
    the relabeled solution is feasible and has the same objective value, and
    it satisfies the chain by construction. Hence the optimal objective
    value is unchanged; we can only lose access to symmetric duplicates of
    optimal solutions, never to the optimum itself.

    As a side effect this single chain also forces every unused vehicle
    (sigma = n+1) to sort after every used vehicle, so the "which of the
    idle vehicle slots is unused" permutation symmetry (the dominant one on
    lightly-loaded / node-restricted instances) is eliminated too, without
    needing a separate "used" indicator or constraint set.

    Not implemented (documented here for future reference): a second,
    independent symmetry exists whenever two or more requests are pairwise
    identical in every model-relevant field (pickup node, delivery node,
    demand, and both time windows). Swapping such requests' pickup/delivery
    model-node labels everywhere is then also a symmetry of the MIP. This is
    not broken here because generated benchmark instances almost always
    differ in at least the time windows even when co-located, making the
    detection/benefit trade-off unfavourable; it would need an *exact*
    field-by-field duplicate check to stay valid.
    """
    n_constraints = 0
    for k in range(len(K) - 1):
        sigma_k = gp.quicksum(
            (i + 1) * x[K[k], 0, i + 1]
            for i in range(n) if (0, i + 1) in A_set
        )
        if (0, 2 * n + 1) in A_set:
            sigma_k += (n + 1) * x[K[k], 0, 2 * n + 1]

        sigma_k1 = gp.quicksum(
            (i + 1) * x[K[k + 1], 0, i + 1]
            for i in range(n) if (0, i + 1) in A_set
        )
        if (0, 2 * n + 1) in A_set:
            sigma_k1 += (n + 1) * x[K[k + 1], 0, 2 * n + 1]

        m.addConstr(sigma_k <= sigma_k1, name=f"symbreak_{K[k]}_{K[k + 1]}")
        n_constraints += 1

    return n_constraints


# ──────────────────────────────────────────────────────────────────────────
# 5. FLEET BUFFER (target #2)
# ──────────────────────────────────────────────────────────────────────────

def add_fleet_buffer_constraint(m, K, A_set, x, n, ws_fleet_size, fleet_buffer):
    """
    Cap the MIP's usable fleet size at (ws_fleet_size + fleet_buffer).

    Vehicles are pre-ordered by the symmetry-breaking chain (see
    `add_symmetry_breaking_constraints`) so that all unused vehicles
    (sigma_k = n+1, i.e. the trivial (0, 2n+1) arc) sort to the back of K.
    Capping the fleet size is therefore just: force each vehicle beyond
    position `max_fleet` in that fixed order to take the empty-route arc.
    This is a genuine model tightening (it can cut off solutions that use a
    larger fleet than max_fleet), so it is only ever applied when explicitly
    requested via the fleet_buffer CLI argument, and only when a valid warm
    start establishes a credible baseline fleet size to buffer from -- the
    buffer is added on top precisely so the cap doesn't bind at the warm
    start's exact fleet size and accidentally exclude a cheaper solution
    that needs a couple more vehicles.
    """
    max_fleet = ws_fleet_size + fleet_buffer
    if max_fleet >= len(K):
        print(f"[FLEET BUFFER] max_fleet ({max_fleet}) >= |K| ({len(K)}); "
              "constraint would be vacuous, skipping.")
        return 0

    forced_idle = K[max_fleet:]
    if (0, 2 * n + 1) not in A_set:
        print("[FLEET BUFFER] Empty-route arc (0, 2n+1) not in A; "
              "cannot force idle vehicles, skipping.")
        return 0

    for k in forced_idle:
        m.addConstr(x[k, 0, 2 * n + 1] == 1, name=f"fleet_buffer_idle_{k}")

    print(f"Fleet buffer            : warm start fleet {ws_fleet_size} + "
          f"buffer {fleet_buffer} -> max fleet size {max_fleet} "
          f"(vehicles {forced_idle[0]}..{forced_idle[-1]} forced idle)")
    return len(forced_idle)


# ──────────────────────────────────────────────────────────────────────────
# 6. ROUTE STRING FORMATTING (target #5)
# ──────────────────────────────────────────────────────────────────────────

def _fmt_num(val):
    """Strip floating-point solver noise; print as int when whole-numbered."""
    r = round(val, 4)
    return str(int(r)) if abs(r - round(r)) < 1e-6 else f"{r:g}"


def _format_route_string(grouped_events, route_cost):
    """
    Render a route as a compact event string, ending with its cost -- same
    format as the extended model's `_format_route_string` (target #5: route
    costs must match the extended model's output exactly).
    """
    body = "".join(
        f"{ev['type']}([{', '.join(str(r) for r in ev['requests'])}] "
        f"@{_fmt_num(ev['node'])} t={_fmt_num(ev['time'])}/{_fmt_num(ev['deadline'])})"
        for ev in grouped_events
    )
    return f"{body} (Cost: {_fmt_num(route_cost)})"


# ──────────────────────────────────────────────────────────────────────────
# 7. MAIN SOLVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────

def solve(instance_path, mip_time_limit=None, mip_setting_mode=0,
          fleet_buffer=None, warm_start=True, threads=8, single_tw=False):
    overall_start = _time.time()

    print(f"\n{'='*70}")
    print("1D-PDPTW COMPACT FORMULATION")
    print(f"Instance: {instance_path}")
    print(f"{'='*70}\n")

    Q, K, depot, time_horizon, requests, t, c, non_loop_penalty_ratio = parse_instance(instance_path, single_tw=single_tw)
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

    # Generate valid arcs by applying the 6 rules.
    A = []
    for i in range(2 * n + 2):
        for j in range(2 * n + 2):
            # R1: No self-loops, no arcs to depot_start, no arcs from depot_end.
            if i == j or j == 0 or i == 2 * n + 1:
                continue

            # R2: No depot_start -> delivery arcs.
            if i == 0 and n + 1 <= j <= 2 * n:
                continue

            # R3: Pickup -> pickup with different destinations.
            if 1 <= i <= n and 1 <= j <= n:
                if loc[n + i] != loc[n + j]:
                    continue

            # R4: Pickup -> delivery wrong destination.
            if 1 <= i <= n and n + 1 <= j <= 2 * n:
                if loc[n + i] != loc[j]:
                    continue

            # R5: Delivery -> delivery different locations.
            if n + 1 <= i <= 2 * n and n + 1 <= j <= 2 * n:
                if loc[i] != loc[j]:
                    continue

            # R6: Time window reachability.
            if a[i] + dur[i] + t(loc[i], loc[j]) > b[j]:
                continue

            A.append((i, j))

    A_set = set(A)
    print(f"Valid arcs: {len(A)}")

    # Phase 2: Regret-2 warm start heuristic (see thesis Section 4.3, 4.3.1)
    heur_routes = None
    heur_obj = None
    n_viol, missing = None, None
    ws_fleet_size = None
    n_oropt_moves = 0
    n_oropt_passes = 0

    if warm_start:
        print(f"\n{'─'*60}")
        print("Phase 2: Regret-2 construction heuristic (warm start) ...")
        print(f"{'─'*60}")

        ws_start = _time.time()
        heur_routes = _construction_heuristic(
            Q, K, depot, time_horizon, requests, t, c, R,
            non_loop_penalty_ratio=non_loop_penalty_ratio,
        )
        heur_routes = sort_routes_for_symmetry(heur_routes, R, n)
        ws_time = _time.time() - ws_start

        n_viol, missing = _check_heuristic(heur_routes, R, Q, depot, time_horizon, requests, t)
        heur_obj = _heuristic_cost(heur_routes, requests, c, depot,
                                    non_loop_penalty_ratio=non_loop_penalty_ratio)
        v_used = sum(1 for r in heur_routes if r)

        print(f"Vehicles used                      : {v_used} / {len(K)}")
        print(f"Heuristic cost                      : {heur_obj:.4f}")
        print(f"Time window/capacity violations     : {n_viol}")
        print(f"Uncovered requests                  : {len(missing)}"
              + (f"  {missing}" if missing else ""))
        print(f"Construction time                   : {ws_time:.2f}s")

        if n_viol > 0 or missing:
            print("[WARM START] Heuristic infeasible and not injected.")
        else:
            # Post-construction local search: or-opt pass that tries
            # relocating each pickup/delivery pair (within its own route or
            # to a different route) to a cheaper feasible position.
            oropt_start = _time.time()
            heur_routes, oropt_obj, n_oropt_moves, n_oropt_passes = _or_opt_pass(
                heur_routes, requests, depot, Q, time_horizon, t, c,
                non_loop_penalty_ratio,
            )
            oropt_time = _time.time() - oropt_start
            n_viol, missing = _check_heuristic(
                heur_routes, R, Q, depot, time_horizon, requests, t
            )
            heur_obj = oropt_obj
            v_used = sum(1 for r in heur_routes if r)

            print(f"Or-opt moves applied                : {n_oropt_moves}")
            print(f"Or-opt passes run                   : {n_oropt_passes}")
            print(f"Cost after or-opt                   : {heur_obj:.4f}")
            print(f"Or-opt time                         : {oropt_time:.2f}s")
            if n_viol > 0 or missing:
                print("[WARM START] Or-opt result infeasible -- discarding "
                      "local-search pass, reverting to pre-or-opt routes "
                      "is not implemented; treating as invalid warm start.")
            else:
                ws_fleet_size = v_used
                # Or-opt can move requests between routes, so re-derive the
                # canonical (symmetry-consistent) ordering from the final
                # routes -- the ordering from right after construction is
                # no longer guaranteed to hold. See `sort_routes_for_symmetry`.
                heur_routes = sort_routes_for_symmetry(heur_routes, R, n)

    # Phase 3: MIP Formulation (see thesis Section 4.1.3)
    print(f"\n{'─'*60}")
    print("Phase 3: Solving 1D-PDPTW compact MIP ...")
    print(f"Start clock: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'─'*60}\n")

    m = gp.Model("1D_PDPTW")
    if mip_time_limit:
        m.setParam("TimeLimit", mip_time_limit)

    # Gurobi Parameters (target #1: MIP setting modes)
    apply_mip_mode(m, mip_setting_mode, MIP_SETTING_MODES, threads=threads)
    print(f"MIP setting mode        : {mip_setting_mode} ({MIP_SETTING_MODES[mip_setting_mode]['label']})")

    # Decision variables
    x = m.addVars(K, A, vtype=GRB.BINARY, name="x")           # arc traversals
    B = m.addVars(K, range(2 * n + 2), lb=0, ub=time_horizon,  # service start times
                  vtype=GRB.CONTINUOUS, name="B")
    L = m.addVars(K, range(2 * n + 2), lb=0, ub=Q,
                  vtype=GRB.CONTINUOUS, name="L")               # vehicle loads

    # Tighten variable bounds.
    for k in K:
        for i in range(2 * n + 2):
            B[k, i].LB = a[i]
            B[k, i].UB = b[i]
            if 1 <= i <= n:
                L[k, i].LB = dem[i]

    # Objective [4.1]

    def _arc_cost(i, j):
        if i == 0 or j == 2 * n + 1:  # depot arcs have zero cost
            return 0.0
        return c(loc[i], loc[j]) if j != 2 * n + 1 else c(loc[i], depot)

    obj = gp.quicksum(
        _arc_cost(i, j) * x[k, i, j] for k in K for (i, j) in A
    )

    # Non-loop route penalty.
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

    for k in K:
        # Flow conservation (every vehicle leaves depot_start exactly once) [4.2]
        m.addConstr(
            gp.quicksum(x[k, 0, j] for j in range(2 * n + 2) if (0, j) in A_set) == 1,
            name=f"start_{k}",
        )
        # Flow conservation (every vehicle returns to depot_end exactly once) [4.3]
        m.addConstr(
            gp.quicksum(x[k, i, 2 * n + 1] for i in range(2 * n + 2) if (i, 2 * n + 1) in A_set) == 1,
            name=f"end_{k}",
        )
        # Flow conservation for remaining nodes in between depot_start and depot_end [4.4]
        for v in range(1, 2 * n + 1):
            in_flow = gp.quicksum(x[k, i, v] for i in range(2 * n + 2) if (i, v) in A_set)
            out_flow = gp.quicksum(x[k, v, j] for j in range(2 * n + 2) if (v, j) in A_set)
            m.addConstr(in_flow == out_flow, name=f"flow_{k}_{v}")

    for i in range(1, n + 1):
        # Coverage (every model pickup node is visited exactly once across all vehicles) [4.5]
        m.addConstr(
            gp.quicksum(x[k, i, j] for k in K for j in range(2 * n + 2) if (i, j) in A_set) == 1,
            name=f"visit_{i}",
        )
        for k in K:
            visit_p = gp.quicksum(x[k, i, j] for j in range(2 * n + 2) if (i, j) in A_set)
            visit_d = gp.quicksum(x[k, n + i, j] for j in range(2 * n + 2) if (n + i, j) in A_set)

            # Pairing (pickup + delivery of the same request on the same vehicle) [4.6]
            m.addConstr(visit_p == visit_d, name=f"same_veh_{k}_{i}")

            # Precedence (pickup precedes delivery) [4.7]
            M_prec = b[i] + dur[i] + t(loc[i], loc[n + i]) - a[n + i]
            m.addConstr(
                B[k, i] + dur[i] + t(loc[i], loc[n + i])
                <= B[k, n + i] + M_prec * (1 - visit_p),
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

    # Target #3: vehicle-fleet symmetry breaking.
    n_symbreak = add_symmetry_breaking_constraints(m, K, A_set, x, n)
    print(f"Symmetry-breaking constraints added : {n_symbreak}")

    # Target #2: optional fleet buffer (requires a valid warm start).
    n_fleet_buffer = 0
    if fleet_buffer is not None:
        if ws_fleet_size is None:
            print("[FLEET BUFFER] No valid warm start available; "
                  "fleet_buffer requested but not applied.")
        else:
            n_fleet_buffer = add_fleet_buffer_constraint(
                m, K, A_set, x, n, ws_fleet_size, fleet_buffer
            )

    # Inject warm-start values (only when the heuristic solution is fully valid).
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
    last_obj = [None]

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
        bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        runtime = model.cbGet(GRB.Callback.RUNTIME)

        gap = abs(obj - bound) / (abs(obj) + 1e-10) * 100
        gap_str = f"{gap:.2f}%"

        print(f"\n[SOLUTION #{sol_count[0]}]  "
              f"obj={obj:.4f}  bound={bound:.4f}  "
              f"gap={gap_str}  t={runtime:.1f}s")

        x_vals = model.cbGetSolution([x[k, i, j] for k in K for (i, j) in A])
        x_sol = {
            (k, i, j): v
            for (k, i, j), v in zip(
                [(k, i, j) for k in K for (i, j) in A], x_vals
            )
        }

        vehicles_printed = 0
        for k in K:
            arcs = {i: j for (i, j) in A if x_sol.get((k, i, j), 0) > 0.5}
            if not arcs or (len(arcs) == 1 and 0 in arcs and arcs[0] == 2 * n + 1):
                continue

            vehicles_printed += 1
            path = [0]
            while path[-1] in arcs:
                path.append(arcs[path[-1]])

            events = []
            for node in path[1:-1]:
                if 1 <= node <= n:
                    events.append(f"P{R[node - 1]}")
                elif n + 1 <= node <= 2 * n:
                    events.append(f"D{R[node - n - 1]}")
            physical_path = " -> ".join(str(loc[nd]) for nd in path)
            print(f"  Veh {k}: [{', '.join(events)}]  path: {physical_path}")

        print(f"  ({vehicles_printed} vehicles active)")

    # Solve!
    mip_start = _time.time()
    m.optimize(print_incumbent)
    mip_time = _time.time() - mip_start

    # Print solution summary.
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

            # Reconstruct event sequence for display.
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

    # Export results to JSON file.
    overall_time = _time.time() - overall_start

    with open(instance_path) as f:
        raw_data = json.load(f)  # meta data

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
        "model": "1dpdptw_compact (no depot cost, non-loop penalty, symmetry breaking)",
        "non_loop_route_penalty_ratio": non_loop_penalty_ratio,
        "mip_setting_mode": mip_setting_mode,
        "mip_setting_mode_label": MIP_SETTING_MODES[mip_setting_mode]["label"],
        "fleet_buffer": fleet_buffer,
        "fleet_buffer_forced_idle": n_fleet_buffer,
        "symmetry_breaking_constraints": n_symbreak,
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
        "ws_fleet": ws_fleet_size,
        "ws_valid": (n_viol == 0 and not missing) if n_viol is not None else None,
        "ws_oropt_moves": n_oropt_moves,
        "ws_oropt_passes": n_oropt_passes,
        "routes": [],
    }

    # Extract routes and fleet count if a solution exists.
    if m.SolCount > 0:
        fleet_count = 0
        routes_list = []
        for k in K:
            active_arcs = [(i, j) for (i, j) in A if x[k, i, j].X > 0.5]
            if not active_arcs or (len(active_arcs) == 1 and active_arcs[0] == (0, 2 * n + 1)):
                continue
            fleet_count += 1

            # Go through route and record (type, physical node, service start
            # time, request, deadline) for every pickup/delivery event in
            # visiting order.
            curr = 0
            raw_events = []
            while curr != 2 * n + 1:
                nxt = next(j for (i, j) in active_arcs if i == curr)
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

            # Route cost, computed exactly like the objective treats this
            # vehicle's arcs (zero-cost depot legs + non-loop penalty),
            # matching the extended model's per-route cost figure (target #5).
            route_arc_cost = sum(
                _arc_cost(i, j) for (i, j) in active_arcs
            )
            # y[k,i,j] is 1 iff vehicle k's route both ends at i and starts at
            # j; at most one such pair can be active per vehicle.
            route_penalty = sum(
                non_loop_penalty_ratio * c(loc[i], loc[j]) * y[k, i, j].X
                for (i, j) in penalty_pairs
            )
            route_cost = route_arc_cost + route_penalty

            route_str = _format_route_string(grouped_events, route_cost)
            routes_list.append(route_str)
        res["fleet_used"] = fleet_count
        res["routes"] = routes_list

    # Save to JSON.
    run_date = datetime.now().strftime("%Y-%m-%d")
    base_name = os.path.basename(instance_path).replace('.json', '')

    output_dir = results_dir("out_1d_pdptw_cm")

    out_name = os.path.join(
        output_dir, f"out_1dpdptw_cm_{run_date}_{base_name}.json"
    )
    with open(out_name, "w") as f:
        json.dump(res, f, indent=4)

    print(f"\nResults written to: {out_name}")


# ──────────────────────────────────────────────────────────────────────────
# 8. ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

def _usage_and_exit():
    print("python 1d-pdptw_compact_v4.py <instance.json> <mip_time_limit_secs> "
          "[mip_setting_mode] [fleet_buffer] [--no-warm-start] [--single-tw]")
    print()
    print("  mip_time_limit_secs : seconds (0 or negative = no time limit)")
    print("  mip_setting_mode    : 0 = standard (default), 1 = focus on closing "
          "bounds, 2 = focus on finding new incumbents")
    print("  fleet_buffer        : non-negative integer; caps the MIP fleet "
          "size at (warm-start fleet size + fleet_buffer). Omit for no cap.")
    print("  --single-tw         : one time window per request (earliest "
          "pickup, latest delivery)")
    sys.exit(1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    _warm_start = "--no-warm-start" not in sys.argv
    _single_tw = "--single-tw" in sys.argv

    if len(args) < 2:
        _usage_and_exit()

    _instance = args[0]

    try:
        _mip_time_limit = float(args[1])
    except ValueError:
        _usage_and_exit()
    if _mip_time_limit <= 0:
        _mip_time_limit = None

    _mip_setting_mode = 0
    if len(args) >= 3:
        try:
            _mip_setting_mode = int(args[2])
        except ValueError:
            _usage_and_exit()
        if _mip_setting_mode not in MIP_SETTING_MODES:
            print(f"Invalid mip_setting_mode {args[2]!r}; must be one of "
                  f"{sorted(MIP_SETTING_MODES)}.")
            sys.exit(1)

    _fleet_buffer = None
    if len(args) >= 4:
        try:
            _fleet_buffer = int(args[3])
        except ValueError:
            _usage_and_exit()
        if _fleet_buffer < 0:
            print("fleet_buffer must be a non-negative integer.")
            sys.exit(1)

    if not _warm_start and _fleet_buffer is not None:
        print("[FLEET BUFFER] --no-warm-start given; fleet_buffer requires a "
              "warm start and will be ignored.")
        _fleet_buffer = None

    solve(_instance, _mip_time_limit, _mip_setting_mode, _fleet_buffer, _warm_start,
          single_tw=_single_tw)
