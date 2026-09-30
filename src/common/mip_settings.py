"""
Gurobi parameter presets ("MIP setting modes") for the 1D-PDPTW solvers.

The compact and extended solvers keep SEPARATE preset tables on purpose: their
parameter sets differ (the compact model additionally sets PrePasses, Aggregate,
CutPasses and Method, and uses Heuristics=0.3 in mode 2 whereas the extended
model uses 0.5). Merging them would silently change one solver's behaviour.

Mode meaning (same idea in both tables):
  0 - standard / balanced (MIPFocus=2)
  1 - focus on closing bounds (MIPFocus=3, heuristics and RINS off, Symmetry=0)
  2 - focus on finding new incumbents (MIPFocus=1, heuristics and RINS up)
"""

EXTENDED_MIP_MODES = {  # plain {param: value} dicts
    0: {
        "MIPFocus":   2,
        "Cuts":       2,
        "Symmetry":   2,
        "Presolve":   2,
    },
    1: {
        "MIPFocus":    3,
        "Cuts":        2,
        "Symmetry":    0,
        "Presolve":    2,
        "Heuristics":  0.0,
        "RINS":        0,
    },
    2: {
        "MIPFocus":    1,
        "Cuts":        2,
        "Symmetry":    2,
        "Presolve":    2,
        "Heuristics":  0.5,
        "RINS":        10,
    },
}


COMPACT_MIP_MODES = {
    0: {
        "label": "standard",
        "params": {
            "PrePasses": -1,
            "Aggregate": 1,
            "MIPFocus": 2,
            "CutPasses": -1,
            "Cuts": 2,
            "Symmetry": 2,
            "Presolve": 2,
            "Method": 2,
            "NodefileStart": 20,
        },
    },
    1: {
        "label": "focus on closing bounds",
        "params": {
            "PrePasses": -1,
            "Aggregate": 1,
            "MIPFocus": 3,       # focus on the bound
            "CutPasses": -1,
            "Cuts": 2,           # very aggressive cut generation
            "Symmetry": 0,
            "Presolve": 2,
            "Method": 2,
            "NodefileStart": 20,
            "Heuristics": 0,   # don't burn B&B time on incumbent heuristics
            "RINS": 0,
        },
    },
    2: {
        "label": "focus on finding new incumbents",
        "params": {
            "PrePasses": -1,
            "Aggregate": 1,
            "MIPFocus": 1,       # focus on feasibility / incumbents
            "CutPasses": -1,
            "Cuts": 2,
            "Symmetry": 2,
            "Presolve": 2,
            "Method": 2,
            "NodefileStart": 20,
            "Heuristics": 0.3,   # spend more time in heuristics
            "RINS": 10,          # run RINS every 10 nodes
        },
    },
}



def apply_mip_mode(model, mode, modes, threads=8):
    """
    Apply preset `mode` from the table `modes` (COMPACT_MIP_MODES or
    EXTENDED_MIP_MODES) to a Gurobi model and set Threads.
    Returns the parameter dict that was applied. Raises ValueError on an
    unknown mode.
    """
    if mode not in modes:
        raise ValueError(
            f"Unknown mip_setting_mode {mode!r}; valid choices are {sorted(modes)}."
        )
    spec = modes[mode]
    params = spec["params"] if "params" in spec else spec
    for name, value in params.items():
        model.setParam(name, value)
    model.setParam("Threads", threads)
    return params


def mode_label(mode, modes):
    spec = modes[mode]
    return spec.get("label", f"mode {mode}") if "params" in spec else f"mode {mode}"
