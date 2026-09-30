"""Instance parsing shared by the compact and extended 1D-PDPTW solvers."""
import json


def parse_instance(filepath, single_tw=False):
    """
    Read a 1D-PDPTW instance JSON.

    single_tw=False : two time windows per request (pickup and delivery each
                      keep their own [earliest, latest]).
    single_tw=True  : one window per request, earliest pickup / latest delivery:
                      p_latest := delivery.latest, d_earliest := pickup.earliest.

    Returns (Q, K, depot, time_horizon, requests, t, c, non_loop_penalty_ratio).
    """
    with open(filepath) as f:
        data = json.load(f)

    Q = data["vehicle"]["capacity"]
    K = list(range(data["vehicle"]["fleet_size"]))
    depot = data.get("depot_node", data.get("depot"))
    time_horizon = data.get("time_horizon", 10 ** 6)
    T_raw = data["travel_times"]
    C_raw = data.get("travel_costs", data["travel_times"])
    non_loop_penalty_ratio = data.get("non_loop_route_penalty_ratio", 0.0)

    def t(i, j):
        return T_raw[i][j]

    def c(i, j):
        return C_raw[i][j]

    requests = {}
    for req in data["requests"]:
        pu, de = req["pickup"], req["delivery"]
        requests[req["id"]] = {
            "demand":     req["demand"],
            "p_node":     pu["node"],
            "d_node":     de["node"],
            "p_earliest": pu["earliest"],
            "p_latest":   de["latest"] if single_tw else pu["latest"],
            "p_duration": pu["duration"],
            "d_earliest": pu["earliest"] if single_tw else de["earliest"],
            "d_latest":   de["latest"],
            "d_duration": de["duration"],
        }

    return Q, K, depot, time_horizon, requests, t, c, non_loop_penalty_ratio
