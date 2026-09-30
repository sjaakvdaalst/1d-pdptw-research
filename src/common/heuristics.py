"""Regret-2 construction + or-opt heuristic and column extraction (shared)."""
import random
import time as _time

from common.feasibility import route_cost, is_1d_feasible, check_heuristic
from common.local_search import or_opt_pass


def construction_heuristic(Q, K, depot, time_horizon, requests, t, c, R,
                             non_loop_penalty_ratio=0.0,
                             noise_scale=0.0, rng=None):
    """
    Strong regret-2 construction
    """
    route_events = [[] for _ in range(len(K))]

    def _route_cost_local(events):
        return route_cost(events, requests, depot, c, non_loop_penalty_ratio)

    def _is_1d_feasible_local(events):
        return is_1d_feasible(events, requests, depot, Q, time_horizon, t)

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


def heuristic_cost(route_events, requests, c, depot, non_loop_penalty_ratio=0.0):
    """
    Total cost of the heuristic solution, no depot legs + non-loop penalty.
    """
    return sum(
        route_cost(events, requests, depot, c, non_loop_penalty_ratio)
        for events in route_events
    )


def events_to_routes(route_events, requests, depot, c, non_loop_penalty_ratio,
                       existing_best: dict):
    new_routes = []
    seen_sets  = {}   # best cost seen among new heuristic routes this run

    for events in route_events:
        if not events:
            continue

        req_set = frozenset(rid for _, rid in events if _ == "P")
        if not req_set:
            continue

        cost = route_cost(events, requests, depot, c, non_loop_penalty_ratio)

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
    total_oropt_passes = 0
    total_oropt_moves = 0

    for run_idx in range(n_runs):
        if run_idx == 0:
            rng_run   = None
            noise_run = 0.0
            label     = "deterministic"
        else:
            rng_run   = random.Random(base_seed + run_idx)
            noise_run = noise_scale
            label     = f"seed={base_seed + run_idx}"

        route_events = construction_heuristic(
            Q, K, depot, time_horizon, requests, t, c, R,
            non_loop_penalty_ratio=non_loop_penalty_ratio,
            noise_scale=noise_run, rng=rng_run,
        )

        n_viol, missing = check_heuristic(
            route_events, R, Q, depot, time_horizon, requests, t
        )
        if n_viol > 0 or missing:
            print(f"  Run {run_idx + 1:2d} ({label:>20s}): "
                  f"infeasible construction ({n_viol} violations, "
                  f"{len(missing)} missing) -- skipping or-opt, using raw "
                  f"construction output as-is for column extraction.")
        else:
            route_events, _oropt_cost, _n_moves, _n_passes = or_opt_pass(
                route_events, requests, depot, Q, time_horizon, t, c,
                non_loop_penalty_ratio,
            )
            total_oropt_passes += _n_passes
            total_oropt_moves += _n_moves

        new_cols = events_to_routes(
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
    print(f"Or-opt passes executed  : {total_oropt_passes}  (moves applied: {total_oropt_moves})")
    print(f"Heuristic time          : {h_time:.2f}s\n")

    return all_new, cumulative_best, total_oropt_passes, total_oropt_moves


def run_regret_oropt(Q, K, depot, time_horizon, requests, t, c, R,
                     non_loop_penalty_ratio=0.0, noise_scale=0.0, rng=None):
    """
    One regret-2 construction followed by or-opt (only if the construction is
    feasible and covers every request).

    Returns dict(events, cost, fleet, valid, oropt_moves, oropt_passes).
    `cost` / `fleet` are None when the construction was not valid.
    """
    events = construction_heuristic(
        Q, K, depot, time_horizon, requests, t, c, R,
        non_loop_penalty_ratio=non_loop_penalty_ratio,
        noise_scale=noise_scale, rng=rng,
    )
    n_viol, missing = check_heuristic(events, R, Q, depot, time_horizon, requests, t)
    valid = (n_viol == 0 and not missing)
    moves = passes = 0
    cost = fleet = None
    if valid:
        events, _, moves, passes = or_opt_pass(
            events, requests, depot, Q, time_horizon, t, c, non_loop_penalty_ratio,
        )
        cost = heuristic_cost(events, requests, c, depot, non_loop_penalty_ratio)
        fleet = sum(1 for r in events if r)
    return {"events": events, "cost": cost, "fleet": fleet, "valid": valid,
            "oropt_moves": moves, "oropt_passes": passes,
            "violations": n_viol, "missing": missing}
