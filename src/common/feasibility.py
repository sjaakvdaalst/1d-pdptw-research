"""Route cost / feasibility checks shared by both 1D-PDPTW solvers."""


def route_cost(events, requests, depot, c, non_loop_penalty_ratio=0.0):
    """Total cost for an event sequence."""
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


def is_1d_feasible(events, requests, depot, Q, time_horizon, t):
    """
    Validates the following constraints:
    1. Capacity and time-window feasibility at every stop.
    2. No mixed destinations onboard.
    3. Must be empty after a delivery if the next stop is a pickup.
    4. Return to depot within time_horizon.
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

            # 1D rule: shared destination check.
            if load > 0 and d_node != onboard_dest:
                return False

            arr = time + t(loc, p_node)
            if arr > req["p_latest"]:
                return False
            time = max(arr, req["p_earliest"]) + req["p_duration"]
            load += req["demand"]
            onboard_dest = d_node
            loc = p_node
        else:
            # Delivery must go to the current batch destination.
            if onboard_dest is not None and d_node != onboard_dest:
                return False

            arr = time + t(loc, d_node)
            if arr > req["d_latest"]:
                return False
            time = max(arr, req["d_earliest"]) + req["d_duration"]
            load -= req["demand"]
            if load == 0:
                onboard_dest = None
            loc = d_node

        if load > Q:
            return False
        prev_type = ev_type

    return (time + t(loc, depot)) <= time_horizon


def check_heuristic(route_events, R, Q, depot, time_horizon, requests, t):
    """Validate a heuristic solution and count 1D-PDPTW constraint violations."""

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

                # 1D: empty-departure.
                if prev_type == "D" and curr_ld > 0:
                    violations += 1

                # 1D: destination consistency.
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

                # 1D: delivery must go to the current batch destination.
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
