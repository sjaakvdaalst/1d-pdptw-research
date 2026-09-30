"""Route export helpers shared by the 1D-PDPTW solvers."""


def simulate_route_times(raw_events, requests, depot, t):
    """
    Re-simulate a route's schedule to attach an actual feasible service
    start time to each event, using the exact same sequential logic as
    is_1d_feasible (arrival = max(prev_time + travel, earliest), then
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


def group_events_for_export(raw_events, requests, depot, t):
    """
    Re-simulate the route's schedule (see simulate_route_times), then
    group consecutive delivery events into a single batch entry, matching
    the compact model's route_str export exactly: same grouping logic,
    same @node t=start/deadline suffix format.
    """
    timed_events = simulate_route_times(raw_events, requests, depot, t)

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


def format_route_string(grouped_events, route_cost):

    def _fmt(val):
        r = round(val, 4)
        return str(int(r)) if abs(r - round(r)) < 1e-6 else f"{r:g}"

    body = "".join(
        f"{ev['type']}([{', '.join(str(r) for r in ev['requests'])}] "
        f"@{_fmt(ev['node'])} t={_fmt(ev['time'])}/{_fmt(ev['deadline'])})"
        for ev in grouped_events
    )
    return f"{body} (Cost: {_fmt(route_cost)})"


def build_result_base(raw_data, instance_path, model_label, non_loop_penalty_ratio):
    """
    Instance-metadata part of the result JSON, identical to the extended
    solver's, so downstream analysis can read all solvers' files alike.
    """
    return {
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
        "model": model_label,
        "non_loop_route_penalty_ratio": non_loop_penalty_ratio,
    }


def route_strings(route_events_list, requests, depot, t, c, cost_fn):
    """Format each non-empty event list as the standard route string."""
    out = []
    for events in route_events_list:
        if not events:
            continue
        raw = [{"type": ty, "request": rid} for ty, rid in events]
        grouped = group_events_for_export(raw, requests, depot, t)
        out.append(format_route_string(grouped, cost_fn(events)))
    return out
