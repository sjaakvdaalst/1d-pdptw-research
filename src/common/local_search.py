"""Or-opt local search shared by both 1D-PDPTW solvers."""
from common.feasibility import route_cost as _route_cost, is_1d_feasible as _is_1d_feasible


def or_opt_pass(route_events, requests, depot, Q, time_horizon, t, c,
                  non_loop_penalty_ratio=0.0, max_passes=5):
    routes = [list(events) for events in route_events]
    n_moves = 0
    n_passes_run = 0

    def route_cost(events):
        return _route_cost(events, requests, depot, c, non_loop_penalty_ratio)

    def feasible(events):
        return _is_1d_feasible(events, requests, depot, Q, time_horizon, t)

    for _pass in range(max_passes):
        n_passes_run += 1
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
    return routes, total_cost, n_moves, n_passes_run
