from typing import Callable, FrozenSet, Dict
from collections import defaultdict

from .structures import TransitionSystemData, States

ViolationCostFn = Callable[[TransitionSystemData, str, States], float]

# ============================================================
# Violation cost function
# ============================================================

def violation_cost_violating_freq(
    ts: TransitionSystemData,
    label: str,
    region: States,
    *,
    target_weight: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """
    Normalized violation cost for an approximate region.

    Cost =
        target_weight * (main-label violating freq / main-label total freq)
        + (1 - target_weight) * (other-labels violating freq / other-labels total freq)

    For each label l:
        total_freq[l] = sum of frequencies of arcs with label l
        violating_freq[l] = total_freq[l] - dominant crossing-type frequency of l
    """

    if not (0.0 <= target_weight <= 1.0):
        raise ValueError("target_weight must be in [0, 1]")

    total_freq: Dict[str, int] = defaultdict(int)
    crossing_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # --- collect statistics per label / crossing type ---
    for arc in ts.arcs:
        if not arc.label.startswith("__END__"):
            init_ctype = arc.crossing_type(region)
            if init_ctype == 'enter':
                ctype = 'enter'
            elif init_ctype == 'exit':
                ctype = 'exit'
            else:
                ctype = 'no cross'
            total_freq[arc.label] += arc.freq
            crossing_freq[arc.label][ctype] += arc.freq

    # --- main label cost ---
    main_total = total_freq.get(label, 0)
    main_dominant = max(crossing_freq[label].values(), default=0)
    main_violation = main_total - main_dominant

    if main_total > 0:
        main_cost = main_violation / (main_total + eps)
    else:
        main_cost = 0.0

    # --- other labels pooled cost ---
    other_total = 0
    other_violation = 0

    for lab, F in total_freq.items():
        if lab == label:
            continue

        dominant = max(crossing_freq[lab].values(), default=0)
        violation = F - dominant

        other_total += F
        other_violation += violation

    if other_total > 0:
        other_cost = other_violation / (other_total + eps)
    else:
        other_cost = 0.0

    # --- weighted final score ---
    score = target_weight * main_cost + (1.0 - target_weight) * other_cost

    return max(0.0, min(1.0, score))/0.66


def violation_cost_violating_freq_no_main(
    ts: TransitionSystemData,
    label: str,
    region: States,
    *,
    target_weight: float = 0.5,
    eps: float = 1e-12,
) -> float:
    """
    Normalized violation cost for an approximate region.

    Cost =
        target_weight * (main-label violating freq / main-label total freq)
        + (1 - target_weight) * (other-labels violating freq / other-labels total freq)

    For each label l:
        total_freq[l] = sum of frequencies of arcs with label l
        violating_freq[l] = total_freq[l] - dominant crossing-type frequency of l
    """

    if not (0.0 <= target_weight <= 1.0):
        raise ValueError("target_weight must be in [0, 1]")

    freq_dict: Dict[str, int] = defaultdict(int)
    crossing_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # --- collect statistics per label / crossing type ---
    for arc in ts.arcs:
        if not arc.label.startswith("__END__"):
            init_ctype = arc.crossing_type(region)
            if init_ctype == 'enter':
                ctype = 'enter'
            elif init_ctype == 'exit':
                ctype = 'exit'
            else:
                ctype = 'no cross'
            freq_dict[arc.label] += arc.freq
            crossing_freq[arc.label][ctype] += arc.freq

    # --- other labels pooled cost ---
    total_freq = 0
    total_violation = 0

    for lab, F in freq_dict.items():

        dominant = max(crossing_freq[lab].values(), default=0)
        violation = F - dominant

        total_freq += F
        total_violation += violation

    if total_freq > 0:
        violation_cost = total_violation / (total_freq + eps)
    else:
        violation_cost = 0.0

    # --- weighted final score ---
    score = violation_cost

    return max(0.0, min(1.0, score))/0.66



def region_violation_score(
    ts: TransitionSystemData,
    region: FrozenSet[str],
    *,
    eps: float = 1e-12,
) -> float:
    total_freq: Dict[str, int] = defaultdict(int)
    crossing_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for arc in ts.arcs:
        ctype = arc.crossing_type(region)

        # normalize in/out -> no_cross
        if ctype in {"in", "out"}:
            ctype = "no_cross"

        w = getattr(arc, "freq", 1)
        total_freq[arc.label] += w
        crossing_freq[arc.label][ctype] += w

    numerator = 0.0
    denominator = 0.0

    for lab, F in total_freq.items():
        dominant = max(crossing_freq[lab].values(), default=0)
        violation = F - dominant

        numerator += violation
        denominator += F

    return numerator / (denominator + eps)


def get_violation_cost_fn(name: str):
    fn = globals().get(name)

    if fn is None:
        raise ValueError(f"Unknown violation cost function: {name!r}")

    if not callable(fn):
        raise TypeError(f"{name!r} exists but is not callable")

    return fn