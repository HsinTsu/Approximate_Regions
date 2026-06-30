from __future__ import annotations
from collections import defaultdict, deque
from typing import Deque, Dict, FrozenSet, List, Set, Tuple
from .structures import TransitionSystemData, TSArc, States, Regions, RegionsDiscoveryResult

# ============================================================
# Core TS / region utilities
# ============================================================

def excitation_region(ts: TransitionSystemData, label: str) -> States:
    return {arc.source for arc in ts.arcs if arc.label == label}

def switching_region(ts: TransitionSystemData, label: str) -> States:
    return {arc.target for arc in ts.arcs if arc.label == label}

def extract_main_behaviour_of_label(
    ts: TransitionSystemData,
    label: str,
    min_main_bahaviour_coverage_ratio: float = 0.8,
) -> States:
    """
    Identify the main arcs of a label based on accumulate frequency ratio by frequency ordering.
    """
    if not (0.0 <= min_main_bahaviour_coverage_ratio <= 1.0):
        raise ValueError("main_behaviour_ratio must be in [0, 1]")
    
    label_arcs: List[TSArc] = [arc for arc in ts.arcs if arc.label == label]
    if not label_arcs:
        return frozenset()

    total_freq = sum(arc.freq for arc in label_arcs)
    if total_freq <= 0:
        return frozenset()
    
    arcs_sorted = sorted(
        label_arcs,
        key=lambda arc: arc.freq,
        reverse=True,
    )

    main_arcs:List[TSArc] = []
    
    accum_freq = 0 
    accum_ratio = 0
    for arc in arcs_sorted:
        if accum_ratio >= min_main_bahaviour_coverage_ratio:
            break
        
        main_arcs.append(arc)
        accum_freq += arc.freq
        accum_ratio += arc.freq/total_freq

    er_main: States = frozenset(arc.source for arc in main_arcs)
    sr_main: States = frozenset(arc.target for arc in main_arcs)

    return er_main, sr_main, accum_freq

def _is_region(ts: TransitionSystemData, states: States) -> bool:
    """
    Check if a state is a strict region to the TS
    """
    return len(_violating_labels(ts, states)) == 0

def _canonical_region(states: States) -> States:
    return frozenset(states)

def _sort_regions(regions: Regions) -> List[States]:
    return sorted(set(regions), key=lambda r: (len(r), tuple(sorted(r))))

# ============================================================
# Region searching
# ============================================================
def _crossing_signature(ts: TransitionSystemData, states: States, label: str) -> FrozenSet[str]:
    """
    Check all the crossing types of a label to a states set.
    """
    signatures: Set[str] = set()
    for arc in ts.arcs:
        if arc.label == label:
            signatures.add(arc.crossing_type(states))
    return frozenset(signatures)

def _normalized_signature(signatures: FrozenSet[str]) -> FrozenSet[str]:
    """
    Normalize crossing type "in" and "out" --> "no_cross"
    """
    norm: Set[str] = set()
    for signature in signatures:
        if signature in {"in", "out"}:
            norm.add("no_cross")
        else:
            norm.add(signature)
    return frozenset(norm)

def _violating_labels(ts: TransitionSystemData, states: States) -> Dict[str, FrozenSet[str]]:
    """
    Collect all the violating labels and their violating types of a states set.
    """
    violating_labels: Dict[str, FrozenSet[str]] = {}
    for label in ts.labels():
        original_sign = _crossing_signature(ts, states, label)
        norm_sign = _normalized_signature(original_sign)

        if len(norm_sign) > 1: # label is violating if normalised sign > 1
            violating_labels[label] = original_sign

    return violating_labels

def _legalizations_for_label(ts: TransitionSystemData, states: States, label: str) -> List[States]:
    """
    Based on violation of the label to the states, return 1 or 2 expansions states.
    """
    signatures = _crossing_signature(ts, states, label)
    # Guard Condition: make sure there're not less than 2 crossing types
    if len(signatures) <= 1:
        return []

    label_arcs = [arc for arc in ts.arcs if arc.label == label]
    if not label_arcs:
        return []

    expansions: List[States] = []

    # if "in" and ("enter" or "exit") --> expand label's source and target states to make the label "in"
    # if "enter" and "exit" --> expand label's source and target states to make the label "in"
    if ("in" in signatures and ("enter" in signatures or "exit" in signatures)) or ("enter" in signatures and "exit" in signatures):
        add: Set[str] = set()
        for arc in label_arcs:
            add.add(arc.source)
            add.add(arc.target)
        expansions.append(_canonical_region(set(states) | add))
        return _sort_regions(expansions)

    # if "out" and "enter" --> 
    # 1. expand "enter" label's source to make the label "in" or "out" --> "no cross"
    # 2. expand "out" label's target to make the label "enter"
    if "out" in signatures and "enter" in signatures:
        add1: Set[str] = set()
        for arc in label_arcs:
            if arc.target in states:
                add1.add(arc.source)

        add2: Set[str] = set()
        for arc in label_arcs:
            if arc.source not in states:
                add2.add(arc.target)

        expansions.append(_canonical_region(set(states) | add1))
        expansions.append(_canonical_region(set(states) | add2))
        return _sort_regions(expansions)

    # if "out" and "exit" --> 
    # 1. expand "exit" label's target to make the label "in" or "out" --> "no cross"
    # 2. expand "out" label's source to make the label "enter"
    if "out" in signatures and "exit" in signatures:
        add1: Set[str] = set()
        for arc in label_arcs:
            if arc.source in states:
                add1.add(arc.target)

        add2: Set[str] = set()
        for arc in label_arcs:
            if arc.target not in states:
                add2.add(arc.source)

        expansions.append(_canonical_region(set(states) | add1))
        expansions.append(_canonical_region(set(states) | add2))
        return _sort_regions(expansions)

    return []

def search_minimal_regions_from_seed(
        ts: TransitionSystemData, 
        seed: States,
        *,
        max_iterations: int = 100) -> Tuple[Regions, Regions]:
    """
    Search minimal strict regions from given states as seed, and expand the region according to the violation conditions.
    A branch stops expanding when region is found, then check if the found region is minimal.
    Return all the found minimal strict regions and non-regions states that have been seen.
    """
    if not seed:
        return set(), set()

    queue: deque[States] = deque([seed]) # queuing states set to be checked and expanded
    seen: Set[States] = {frozenset(seed)}
    minimal_strict_regions: Regions = set()
    non_region_seen: Regions = set()

    c = 0
    while queue:
        c += 1
        current = queue.popleft()

        if _is_region(ts, set(current)):
            minimal_strict_regions = {r for r in minimal_strict_regions if not current.issubset(r)} # remove all the superset of current region
            if not any(r.issubset(current) for r in minimal_strict_regions): # no other region is subset of current region
                minimal_strict_regions.add(current)
            continue

        non_region_seen.add(current)

        violating_labels = _violating_labels(ts, set(current)) # check the violating labels and violating crossing types
        next_sets: List[States] = [] # the expansion states to legalize the violates from current state
        for label in sorted(violating_labels):
            next_sets.extend(_legalizations_for_label(ts, set(current), label))

        next_sets = _sort_regions([r for r in next_sets if r != current])

        # if no further expansion(reaching universal), continue
        if not next_sets:
            universal = frozenset(ts.states)
            if universal not in seen:
                seen.add(universal)
                queue.append(universal)
            continue

        # add the unseen states to seen and next checking queue
        for nxt in next_sets:
            if frozenset(nxt) not in seen:
                seen.add(nxt)
                queue.append(nxt)
        
        # limit the total number of processed nodes(states) to prevent long computation times
        if c > max_iterations:
            break

    return minimal_strict_regions, non_region_seen

def non_region_violation_cost_check(
    ts: TransitionSystemData,
    label: str,
    candidates: Regions,
    violation_cost_fn: ViolationCostFn,
    approx_cost_threshold: float,
) -> Regions:
    """
    Check the violation cost of non-region sets and keep the sets with costs within approx_cost_threshold.
    """
    kept: Regions = set()

    for region in candidates:
        cost = violation_cost_fn(ts, label, region)
        if cost <= approx_cost_threshold:
            kept.add(region)

    return kept

# ============================================================
# Label splitting
# ============================================================
def _intersect_regions(regions: Regions) -> States:
    if not regions:
        return frozenset()

    it = iter(regions)
    acc = set(next(it))
    for r in it:
        acc.intersection_update(r)
    return frozenset(acc)

def _normalize_ctype(ctype: str) -> str:
    """
    Normalize crossing type "in" and "out" --> "no_cross"
    """
    if ctype in ("in", "out"):
        return "no_cross"
    return ctype


def split_labels_from_signatures(
    current_ts: TransitionSystemData,
    split_target_state_sets: List[States],
) -> Tuple[TransitionSystemData, Dict[str, str]]:
    """
    Split labels according to arc signatures.

    Parameters
    ----------
    current_ts :
        Current transition system.
    split_target_state_sets :
        Format:
            violating_label -> { arc -> [ctype1, ctype2, ...] }

    Returns
    -------
    new_ts :
        Transition system after applying signature-based splitting.
    local_split_map :
        new_label -> original_label
    """
    # -------------------------------------------------
    # 0) Transfrom the violation of each arc to each target states sets into a signature
    # -------------------------------------------------
    violating_signatures: Dict[str, Dict[TSArc, List[str]]] = {}
    for candidate in split_target_state_sets:
        bad = _violating_labels(current_ts, candidate) # violating label of the states

        for bad_label in bad:
            arcs = [a for a in current_ts.arcs if a.label == bad_label] # all the arcs of the label
            
            if bad_label not in violating_signatures: # create dict for each violating arc
                violating_signatures[bad_label] = {arc: [] for arc in arcs}

            for arc in arcs: # note the normalized crossing type of the arc to the states
                violating_signatures[bad_label][arc].append(_normalize_ctype(arc.crossing_type(candidate)))

    # -------------------------------------------------
    # 1) Group arcs by identical signature per label
    # -------------------------------------------------
    split_groups: Dict[str, List[Set[TSArc]]] = {}

    # split each label into groups based on signatures
    for label, arc_sig_map in violating_signatures.items():
        if not arc_sig_map:
            continue

        # defensive check: all arcs of the same label should have equal signature length
        sig_lengths = {len(sig) for sig in arc_sig_map.values()}
        if len(sig_lengths) > 1:
            raise ValueError(
                f"Inconsistent signature lengths for label {label}: {sorted(sig_lengths)}"
            )

        grouped: Dict[Tuple[str, ...], Set[TSArc]] = defaultdict(set)

        # group the arcs with same signature
        for arc, sig_list in arc_sig_map.items():
            signature = tuple(sig_list)
            grouped[signature].add(arc)

        # generate the list of each group
        groups = list(grouped.values())

        # only split when more than one distinct signature exists
        if len(groups) >= 2:
            # deterministic ordering for reproducibility
            groups.sort(
                key=lambda g: sorted((a.source, a.target, a.freq) for a in g)
            )
            # store the splitting group of each label
            split_groups[label] = groups

    # nothing to split
    if not split_groups:
        return current_ts.copy(), {}

    # -------------------------------------------------
    # 2) Generate fresh labels for each split group
    # -------------------------------------------------
    existing_labels: Set[str] = set(current_ts.labels())
    label_group_names: Dict[str, List[str]] = {}
    label_split_map: Dict[str, str] = {}

    for label, groups in split_groups.items():
        fresh_labels: List[str] = []
        idx = 1
        while len(fresh_labels) < len(groups):
            candidate = f"{label}#open{idx}"    # candidate of splited label's name
            if candidate not in existing_labels:  # add the candidate if it's not exist
                fresh_labels.append(candidate)
                existing_labels.add(candidate)
            idx += 1

        label_group_names[label] = fresh_labels  # store the candidate new label names of each label
        for new_label in fresh_labels:
            label_split_map[new_label] = label # label split record. map new label -> previous label

    # -------------------------------------------------
    # 3) Build arc -> new_label mapping
    # -------------------------------------------------
    arc_relabel: Dict[TSArc, str] = {}

    for label, groups in split_groups.items():
        fresh_labels = label_group_names[label]
        for i, group in enumerate(groups):
            for arc in group:
                arc_relabel[arc] = fresh_labels[i] # assign new name to each label according to the group

    # -------------------------------------------------
    # 4) Rebuild TS with relabeled arcs
    # -------------------------------------------------
    new_arcs: Set[TSArc] = set()

    for arc in current_ts.arcs:
        if arc in arc_relabel:
            new_arcs.add(
                TSArc(
                    source=arc.source,
                    label=arc_relabel[arc],
                    target=arc.target,
                    freq=arc.freq,
                )
            )
        else:
            new_arcs.add(arc)

    new_ts = TransitionSystemData(
        states=set(current_ts.states),
        arcs=new_arcs,
        initial_states=set(current_ts.initial_states),
        sink_states=set(current_ts.sink_states),
    )

    return new_ts, label_split_map


# ============================================================
# Final result cleansing
# ============================================================
def _prune_non_minimal(regions: Regions) -> Regions:
    """
    Remove non-minimal regions
    """
    sorted_regions = sorted(regions, key=len)
    result: List[States] = []

    for r in sorted_regions:
        if not any(prev.issubset(r) for prev in result):
            result.append(r)

    return set(result)

def labels_rename(
    ts: TransitionSystemData,
    split_map: Dict[str, str],
) -> Tuple[TransitionSystemData, Dict[str, str]]:
    """
    Rename labels into canonical names of the form:
        a, a#1, a#2, ...

    Rules
    -----
    1. split_map stores direct parent only: child -> parent
    2. Canonical renaming is grouped by base label, defined as:
           base(label) = label.split("#", 1)[0]
    3. Base/original label is preserved when present:
           a -> a
    4. All other labels in the same family are renamed to:
           a#1, a#2, ...
    5. Returned split_map remains direct-parent mapping after renaming

    Returns
    -------
    renamed_ts, renamed_split_map
    """

    def _base_label(label: str) -> str:
        return label.split("#", 1)[0]

    def _label_sort_key(label: str) -> Tuple[int, str]:
        # base label first, then the others in lexical order
        if "#" not in label:
            return (0, label)
        return (1, label)

    def _rename_arc_label(arc: TSArc, new_label: str) -> TSArc:
        # support both TSArc(source, label, target)
        # and TSArc(source, label, target, freq)
        if hasattr(arc, "freq"):
            return TSArc(
                source=arc.source,
                label=new_label,
                target=arc.target,
                freq=arc.freq,
            )
        return TSArc(
            source=arc.source,
            label=new_label,
            target=arc.target,
        )

    # -------------------------------------------------
    # 1) labels that actually exist in final TS
    # -------------------------------------------------
    existing_labels: Set[str] = {arc.label for arc in ts.arcs}

    # -------------------------------------------------
    # 2) keep only split relations whose child still exists
    #    in final TS
    # -------------------------------------------------
    effective_split_map: Dict[str, str] = {
        child: parent
        for child, parent in split_map.items()
        if child in existing_labels
    }

    # -------------------------------------------------
    # 3) build rename universe
    #
    # Main idea:
    # - all split children should be re-encoded
    # - parents referenced by those children must also be included
    # - existing labels in final TS must also be included
    # -------------------------------------------------
    labels_to_rename: Set[str] = set(existing_labels)
    labels_to_rename |= set(effective_split_map.keys())
    labels_to_rename |= set(effective_split_map.values())

    # -------------------------------------------------
    # 4) group labels by base label
    # -------------------------------------------------
    by_base: Dict[str, List[str]] = defaultdict(list)
    for label in labels_to_rename:
        by_base[_base_label(label)].append(label)

    # -------------------------------------------------
    # 5) build old_label -> canonical_label
    # -------------------------------------------------
    rename: Dict[str, str] = {}

    for base in sorted(by_base.keys()):
        family = sorted(set(by_base[base]), key=_label_sort_key)

        # preserve base label if it exists in this family
        has_base = base in family
        if has_base:
            rename[base] = base

        # everything except the base label gets canonical numbering
        split_labels = [lab for lab in family if lab != base]

        for idx, old_label in enumerate(split_labels, start=1):
            rename[old_label] = f"{base}#{idx}"

    # defensive fallback
    for label in labels_to_rename:
        if label not in rename:
            rename[label] = label

    # -------------------------------------------------
    # 6) rename TS arcs
    # -------------------------------------------------
    renamed_arcs: Set[TSArc] = {
        _rename_arc_label(arc, rename[arc.label])
        for arc in ts.arcs
    }

    renamed_ts = TransitionSystemData(
        states=ts.states.copy(),
        arcs=renamed_arcs,
        initial_states=ts.initial_states.copy(),
        sink_states=ts.sink_states.copy(),
    )

    # -------------------------------------------------
    # 7) rename split_map, preserving direct-parent mapping
    # -------------------------------------------------
    renamed_split_map: Dict[str, str] = {}

    for child, parent in effective_split_map.items():
        new_child = rename[child]
        new_parent = rename[parent]
        renamed_split_map[new_child] = new_parent

    return renamed_ts, renamed_split_map


# ============================================================
# Approximate region method main pipeline
# ============================================================

def approximate_label_splitting_algo(
    ts: TransitionSystemData,
    violation_cost_fn: ViolationCostFn,
    approx_cost_threshold: float = 0,
    min_main_bahaviour_coverage_ratio: float = 0.8,
    min_freq_to_trigger_label_splitting: int = 10
) -> RegionsDiscoveryResult:
    """
    Discover strict and approximate state regions with frequency-guided label splitting.

    For each label, the algorithm first searches strict pre-regions from its
    main behaviour excitation region. If strict excitation closure cannot be achieved,
    explored non-region candidates are evaluated by the given violation cost function and
    accepted as approximate regions when their cost is below the threshold.

    If neither strict nor approximate closure is sufficient, the label may be
    split using main behaviour excitation seed. The main behaviour excitation region is
    obtained by accumulating the most frequent arcs of the label until the given
    frequency ratio is reached.

    Parameters
    ----------
    ts:
        Transition system used as the input for region discovery.

    violation_cost_fn:
        Function used to measure how much a candidate state set violates the
        region condition.

    approx_cost_threshold:
        Maximum allowed violation cost for accepting an explored state set as an
        approximate region. A value of 0 only accepts candidates without
        violations.

    min_accumulate_frequency_ratio:
        Minimum accumulated frequency ratio used to construct the high-frequency
        excitation seed of a label.

    min_freq_to_trigger_label_splitting:
        Minimum label frequency required before frequency-based label
        splitting is attempted.

    Returns
    -------
    RegionsDiscoveryResult
        Result containing the final split transition system, strict regions,
        approximate regions, split map, and split history.
    """
    current_ts = ts.copy()
    label_split_map: Dict[str, str] = {}
    strict_regions: Regions = set()
    approximate_regions: Regions = set()

    # initialize the labels to be checked and ignore __END__ kind labels
    next_round_labels: Set[str] = set([label for label in current_ts.labels() if not label.startswith("__END__")])
    # initialize strict regions and include __END__ state as it's always a minimal strict region
    strict_regions |= {frozenset(current_ts.sink_states)}

    while next_round_labels:
        queue: Deque[str] = deque(sorted(next_round_labels)) # initialize this round's checking label
        next_round_labels = set() # initialize the next round labels
        split_target_state_sets = [] # initialize target state sets for label splitting

        while queue:
            label = queue.popleft() # label check
            
            # create main behaviour excitation region, switch region and frequency of label
            er_main, sr_main, main_behaviour_freq = extract_main_behaviour_of_label(current_ts, label, min_main_bahaviour_coverage_ratio)

            # if main behaviour frequency of the label is too low, label splitting won't be triggered
            if main_behaviour_freq < min_freq_to_trigger_label_splitting:
                continue
            # create full behaviour excitation region
            er = excitation_region(current_ts, label)
            if not er:
                continue

            # regions searching from er_main
            strict_regions_er, seen_non_regions_er = search_minimal_regions_from_seed(current_ts, frozenset(er_main))
            # regions searching from sr_main
            strict_regions_sr, seen_non_regions_sr = search_minimal_regions_from_seed(current_ts, frozenset(sr_main))

            # save found strict regions
            strict_regions |= strict_regions_er
            strict_regions |= strict_regions_sr

            # accept non-region states as approximate region by examining violation cost
            approximate_regions_sr = non_region_violation_cost_check(
                current_ts,
                label,
                seen_non_regions_sr,
                violation_cost_fn,
                approx_cost_threshold,
            )
            approximate_regions_er = non_region_violation_cost_check(
                current_ts,
                label,
                seen_non_regions_er,
                violation_cost_fn,
                approx_cost_threshold,
            )
            
            # save found approximate regions
            approximate_regions |= approximate_regions_er
            approximate_regions |= approximate_regions_sr

            # calculate pre-region intersection
            if not approximate_regions_er: 
                # pre-region intersection calculated by strict regions
                pre_region_intersection = _intersect_regions(strict_regions_er) 
            else:
                # pre-region intersection calculated by strict regions & approximate regions
                pre_region_intersection = _intersect_regions(strict_regions_er | approximate_regions_er)
            
            # if intersection is subset of er, there're sufficient regions to restrict main behaviour or the label
            # loosed EC satisfied
            if pre_region_intersection.issubset(er):
                continue
            
            split_target_state_sets.append(er_main) # add er_main as target states set for label splitting

        # loop break if no label splitting and labels checks for next round needed
        if not split_target_state_sets:
            if not next_round_labels:
                break
            continue
        
        # label splitting based on target state sets and update TS
        current_ts, local_split_map = split_labels_from_signatures(current_ts, split_target_state_sets)
        # add splitted labels to next checking list
        next_round_labels |= set(local_split_map.keys())
        # update label splitting record
        label_split_map.update(local_split_map)

    # promote approximate regions to strict regions if it's a strict region after label splitting
    promoted_to_strict: Regions = {
        r for r in approximate_regions
        if _is_region(current_ts, set(r))
    }

    if promoted_to_strict:
        approximate_regions -= promoted_to_strict
        strict_regions |= promoted_to_strict

    strict_regions = _prune_non_minimal(strict_regions) # remove non-minimal strict regions
    current_ts, label_split_map = labels_rename(current_ts, label_split_map) # rename labels for readability

    return RegionsDiscoveryResult(
        split_ts=current_ts,
        minimal_strict_regions=strict_regions,
        approximate_regions=approximate_regions,
        split_map=label_split_map,
    )


# ============================================================
# Classical label splitting method main pipeline
# ============================================================

def classical_label_splitting_algo(
    ts: TransitionSystemData,
) -> RegionsDiscoveryResult:
    """
    Discover strict state regions by searching pre-regions from each label's
    excitation region(er). If excitation closure cannot be achieved, split the label 
    using er.

    Parameters
    ----------
    ts:
        Transition system used as the input for region discovery.

    Returns
    -------
    RegionsDiscoveryResult
        Result containing the final split transition system, strict regions,
        approximate regions, split map, and split history.
    """
    current_ts = ts.copy()
    label_split_map: Dict[str, str] = {}
    regions: Regions = set()

    # initialize the labels to be checked and ignore __END__ kind labels
    next_round_labels: Set[str] = set([label for label in current_ts.labels() if not label.startswith("__END__")])
    # initialize strict regions and include __END__ state as it's always a minimal strict region
    regions |= {frozenset(current_ts.sink_states)}

    while next_round_labels:
        queue: Deque[str] = deque(sorted(next_round_labels)) # initialize this round's checking label
        next_round_labels = set() # initialize the next round labels
        split_target_state_sets = [] # initialize target state sets for label splitting

        while queue:
            label = queue.popleft() # label check
            
            # create full behaviour excitation region
            er = excitation_region(current_ts, label)
            sr = switching_region(current_ts, label)
            if not er:
                continue

            # regions searching from er
            regions_er, seen_non_regions_er = search_minimal_regions_from_seed(current_ts, frozenset(er))
            # regions searching from sr
            regions_sr, seen_non_regions_sr = search_minimal_regions_from_seed(current_ts, frozenset(sr))

            # save found strict regions
            regions |= regions_er
            regions |= regions_sr
            
            # calculate pre-region intersection
            pre_region_intersection = _intersect_regions(regions_er) 
            
            # EC property check
            if pre_region_intersection == er:
                continue
            
            split_target_state_sets.append(er) # add er as target states set for label splitting

        # loop break if no label splitting and labels checks for next round needed
        if not split_target_state_sets:
            if not next_round_labels:
                break
            continue
        
        # label splitting based on target state sets and update TS
        current_ts, local_split_map = split_labels_from_signatures(current_ts, split_target_state_sets)
        # add splitted labels to next checking list
        next_round_labels |= set(local_split_map.keys())
        # update label splitting record
        label_split_map.update(local_split_map)

    regions = _prune_non_minimal(regions) # remove non-minimal strict regions
    current_ts, label_split_map = labels_rename(current_ts, label_split_map) # rename labels for readability

    return RegionsDiscoveryResult(
        split_ts=current_ts,
        minimal_strict_regions=regions,
        approximate_regions=set(),
        split_map=label_split_map,
    )



if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.transition_system_builder import read_xes_to_ts
    from violation_cost_fn import *

    load_dotenv()

    XES_PATH = os.environ["XES_PATH"]
    VIEW = os.getenv("VIEW", "sequence")
    WINDOW = int(os.getenv("WINDOW", "2"))
    ACTIVITY_KEY = os.getenv("ACTIVITY_KEY", "concept:name")
    DIRECTION = os.getenv("DIRECTION", "backward")

    ts, self_loop_arcs = read_xes_to_ts(
        XES_PATH,
        view=VIEW,                      # type: ignore[arg-type]
        window=WINDOW,
        activity_key=ACTIVITY_KEY,
        include_data=1,
        direction="backward",            # type: ignore[arg-type]
    )

    # --- Step 2: run approximate splitting
    approx_cost_threshold = 0.02
    min_pattern_ratio = 0
    min_accumulate_frequency_ratio = 0.8
    min_freq_to_trigger_label_splitting = 10
    result = approximate_label_splitting_algo(
        ts,
        violation_cost_violating_freq,
        approx_cost_threshold=approx_cost_threshold,
        min_main_bahaviour_coverage_ratio = min_accumulate_frequency_ratio,
        min_freq_to_trigger_label_splitting = min_freq_to_trigger_label_splitting
    )
    # --- Step 3: unpack (depending on your implementation)
    try:
        final_ts = result.split_ts
        minimal_regions = result.minimal_strict_regions
        approximate_regions = result.approximate_regions
        split_map = result.split_map
    except AttributeError:
        final_ts, minimal_regions, split_map = result

    print("-----------------------------------")
    print("Origin TS")
    print("-----------------------------------")
    print("States:", len(ts.states))
    print("Labels:", len(ts.labels()))
    print("Arcs:", len(ts.arcs))

    print("-----------------------------------")
    print("Final TS")
    print("-----------------------------------")
    print("States:", len(final_ts.states))
    print("Labels:", len(final_ts.labels()))
    print("Arcs:", len(final_ts.arcs))
    print(final_ts.labels())
    print("Strict regions:", len(minimal_regions))
    print("Approximate regions:", len(approximate_regions))