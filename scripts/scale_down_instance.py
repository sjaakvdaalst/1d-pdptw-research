"""
Purpose: Scale down 1D-PDPTW instance by randomly sampling a subset of requests.

Usage:
    python scale_down_instance.py <instance.json> <requests> <vehicles>
"""

import sys
import json
import os
import random
import copy


def scale_down(instance_path, n_requests, vehicles):

    with open(instance_path) as f:
        data = json.load(f)

    depot = data.get("depot_node", data.get("depot"))
    all_n_requests = data["requests"]

    seed = 42

    # Sample a subset of requests (random)
    rng = random.Random(seed)
    sampled = rng.sample(all_n_requests, n_requests)

    # Physical nodes still needed: depot + every pickup/delivery node referenced by the sampled n_requests
    used_nodes = {depot}  # this is the new set of nodes for the scaled down version
    for req in sampled:
        used_nodes.add(req["pickup"]["node"])
        used_nodes.add(req["delivery"]["node"])

    # Node IDs
    old_ids_sorted = [depot] + sorted(n for n in used_nodes if n != depot)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(old_ids_sorted)}
    new_depot = old_to_new[depot]




    # Only keep the relevant slice of the travel matrix.
    # Arcs connecting nodes that are not in the sampled requests are removed from the travel matrix.
    def slice_matrix(matrix):
        return [
            [matrix[old_i][old_j] for old_j in old_ids_sorted] for old_i in old_ids_sorted
        ]

    new_travel_times = slice_matrix(data["travel_times"])




    # Build dictionary where the key is the old node ID and the value is the complete node object (for easy lookups)
    old_node_by_id = {node["id"]: node for node in data.get("nodes", [])}

    # Remap request ids
    sampled_sorted = sorted(sampled, key=lambda r: r["id"])  # make sure new request ids are assigned in ascending order of old request ids
    old_rid_to_new_rid = {req["id"]: new_id for new_id, req in enumerate(sampled_sorted)} # create a dictionarymapping the old request IDs to the new request IDs (new IDs are consecutive starting from 0)

    new_request_list = []
    for req in sampled_sorted:
        new_req = copy.deepcopy(req) # create a deep copy of the request to avoid modifying the original
        new_req["id"] = old_rid_to_new_rid[req["id"]]
        new_req["pickup"]["node"] = old_to_new[req["pickup"]["node"]]
        new_req["delivery"]["node"] = old_to_new[req["delivery"]["node"]]
        new_request_list.append(new_req)

    new_nodes = []
    for old_id in old_ids_sorted:
        old_node = old_node_by_id.get(old_id, {"id": old_id})
        new_node = copy.deepcopy(old_node)
        new_node["id"] = old_to_new[old_id]

        # Only keep pickups and deliveries that are still present in the new request list
        kept_pickups = [
            old_rid_to_new_rid[rid]
            for rid in old_node.get("requests", {}).get("pickups", [])
            if rid in old_rid_to_new_rid
        ]
        kept_deliveries = [
            old_rid_to_new_rid[rid]
            for rid in old_node.get("requests", {}).get("deliveries", [])
            if rid in old_rid_to_new_rid
        ]
        new_node["requests"] = {"pickups": kept_pickups, "deliveries": kept_deliveries}
        new_nodes.append(new_node)

    # Assemble the new instance dict
    new_data = copy.deepcopy(data)
    new_data["nodes"] = new_nodes
    new_data["requests"] = new_request_list
    new_data["travel_times"] = new_travel_times
    new_data["depot"] = new_depot
    new_data["depot_node"] = new_depot
    new_data["vehicle"]["fleet_size"] = vehicles






    base_name = data.get("base_instance_name", os.path.basename(instance_path).replace(".json", ""))
    
    # Retrieve node restriction percentage for file name
    restr = data.get("variant", {}).get("node_restriction_percentage")
    restr_str = f"_node_restriction_percentage_{restr}%" if restr is not None else ""
    
    new_data["instance_name"] = f"{base_name}_{n_requests}req_{vehicles}veh{restr_str}"

    out_dir = "modified_instances"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{new_data['instance_name']}.json")

    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"Original : {len(all_n_requests)} requests, {len(data.get('nodes', []))} nodes, "
          f"fleet {data['vehicle']['fleet_size']}")
    print(f"Scaled   : {n_requests} requests, {len(new_nodes)} nodes, fleet {vehicles}")
    print(f"Saved to : {out_path}")

    return new_data, out_path


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("invalid usage")
        sys.exit(1)

    _instance    = sys.argv[1]
    n_requests  = int(sys.argv[2])
    _vehicles  = int(sys.argv[3])

    scale_down(_instance, n_requests, _vehicles)