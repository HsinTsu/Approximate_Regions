from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Tuple
from collections import defaultdict
from .structures import TransitionSystemData, States, Regions, PlaceSignature, PlaceCandidate, Place


def _normalize_ctype(ctype: str) -> str:
    if ctype in ("in", "out"):
        return "no_cross"
    return ctype

def _dominant_crossing_type(
    ts: TransitionSystemData,
    region: States,
    label: str,
    *,
    tie_policy: bool = True,
) -> Optional[str]:
    """
    Return the frequency-weighted dominant normalized crossing type of a label
    with respect to a region.

    For all arcs with the given label, their crossing types are computed against
    the region and normalized into one of:

    - "enter"
    - "exit"
    - "no_cross"

    The frequency of each arc contributes to the total weight of its normalized
    crossing type. The crossing type with the largest accumulated frequency is
    returned as the dominant behavior of the label on this region.

    Parameters
    ----------
    ts:
        Transition system containing labeled arcs with frequency information.

    region:
        Candidate state set used to evaluate crossing behavior.

    label:
        Label whose arcs are evaluated.

    tie_policy:
        Policy for resolving ties between accumulated crossing-type frequencies.

        - If False, return None when multiple crossing types have the same
          maximum frequency.
        - If True, resolve ties by preferring "exit" over
          "no_cross" over "enter". 

    Returns
    -------
    Optional[str]
        The dominant normalized crossing type, or None if the label has no arcs
        in the transition system or if a tie cannot be resolved.
    """
    counts: Dict[str, int] = defaultdict(int)

    for arc in ts.arcs:
        if arc.label != label:
            continue
        norm = _normalize_ctype(arc.crossing_type(region))
        counts[norm] += int(getattr(arc, "freq", 1))

    if not counts:
        return None, None

    max_count = max(counts.values())
    winners = [ctype for ctype, c in counts.items() if c == max_count]
    if len(winners) == 1:
        return winners[0], counts

    if not tie_policy:
        return None, None

    for preferred in ("exit", "enter", "no_cross"):
        if preferred in winners:
            return preferred, counts
    return None, None


def _prune_minimal(regions: Regions) -> Regions:
    """
    Remove non-minimal regions
    """
    sorted_regions = sorted(regions, key=len)
    result: List[States] = []

    for r in sorted_regions:
        if not any(prev.issubset(r) for prev in result):
            result.append(r)

    return set(result)


def region_to_place(
    ts: TransitionSystemData,
    state_region: States,
    *,
    place_id: str,
    tie_policy: bool = True,
    min_crossing_freq_ratio:float = 0.05,
    min_crossing_freq_amount:int = 10
) -> PlaceCandidate:
    """
    Convert a state-based region into a Petri-net place candidate.

    The function interprets the crossing behavior of each label with respect to
    the given state set and derives the corresponding place arcs. Labels whose
    behavior enters the state region are mapped to incoming arcs of the place,
    while labels whose behavior exits the state region are mapped to outgoing
    arcs of the place. Labels that do not cross the region boundary are ignored.

    For strict regions, each label has a unique normalized crossing type. For
    approximate regions, the crossing type is chosen by frequency-weighted
    dominance, so that the most frequent behavior determines how the label is
    connected to the place.

    Parameters
    ----------
    ts:
        Transition system containing states, labeled arcs, and arc frequencies.

    state_region:
        State set to be converted into a place candidate.

    place_id:
        Identifier assigned to the generated place candidate.

    tie_policy:
        Policy used when multiple normalized crossing types have the same
        dominant frequency.

        - If False, unresolved ties are ignored.
        - If True, ties prefer boundary-crossing behavior ("enter" or "exit")
          over "no_cross".

    min_crossing_freq_ratio:
        Minimum relative frequency ratio required for a crossing behavior to be
        considered reliable. Crossing behavior below this ratio is treated as
        insufficiently supported.

    min_crossing_freq_amount:
        Minimum absolute frequency required for a crossing behavior to be
        considered reliable. Crossing behavior below this amount is treated as
        insufficiently supported.

    Returns
    -------
    PlaceCandidate
        A place candidate derived from the state region, including its place ID,
        connected input/output labels, initial marking information, and any
        metadata needed for later filtering or Petri-net construction.
    """
    enter: Dict[str, int] = {}
    exit: Dict[str, int] = {}
    label_dom_c_type: Dict[str, str] = {}
    pre: Dict[str, int] = {}
    post: Dict[str, int] = {}

    for label in ts.labels():
        dom, counts = _dominant_crossing_type(
            ts,
            state_region,
            label,
            tie_policy=tie_policy,
        )
        if dom is None:
            continue

        label_dom_c_type[label] = dom

        for c_type, freq in counts.items():
            if c_type == "enter":
                enter[label] = freq
            elif c_type == "exit":
                exit[label] = freq
    total_enter_freq = sum(enter.values())
    total_exit_freq = sum(exit.values())

    for label, dom_c_type in label_dom_c_type.items():
        if dom_c_type == "enter":
            freq = enter[label]
            if freq/total_enter_freq >= min_crossing_freq_ratio and freq >= min_crossing_freq_amount:
                pre[label] = 1
        elif dom_c_type == "exit":
            freq = exit[label]
            if freq/total_exit_freq >= min_crossing_freq_ratio and freq >= min_crossing_freq_amount:
                post[label] = 1

    m0 = 1 if any(s in state_region for s in ts.initial_states) else 0
    fm = 1 if any(s in state_region for s in ts.sink_states) else 0

    p = PlaceCandidate(
        place_id=place_id,
        states=frozenset(state_region),
        pre=pre,
        post=post,
        m0=m0,
        fm=fm,
        score=0,
        origin="approximate",
    )
    p.validate()
    return p


def build_place_candidates(
    ts: TransitionSystemData,
    #strict_regions: Regions,
    regions: Regions,
    *,
    approx_tie_policy: bool = True,
    min_crossing_freq_ratio: float = 0.05,
    min_crossing_freq_amount: int = 10
) -> List[PlaceCandidate]:
    out: List[PlaceCandidate] = []
    approx_sorted = sorted(regions, key=lambda r: (len(r), tuple(sorted(r))))

    for idx, region in enumerate(approx_sorted):
        out.append(region_to_place(
                ts,
                region,
                place_id=f"p_a{idx}",
                tie_policy=approx_tie_policy,
                min_crossing_freq_ratio = min_crossing_freq_ratio,
                min_crossing_freq_amount = min_crossing_freq_amount
            ))

    return out


def deduplicate_identical_places(
    places: Iterable[PlaceCandidate]
) -> List[PlaceCandidate]:
    """
    Deduplication identical places on (pre, post, m0).
    """
    best: Dict[PlaceSignature, PlaceCandidate] = {}

    def rank(p: PlaceCandidate) -> Tuple[float, int, str]:
        return (p.score, 0 if p.origin == "strict" else 1, p.place_id)

    for p in places:
        key = p.signature()
        cur = best.get(key)
        if cur is None or rank(p) < rank(cur):
            best[key] = p

    deduped = sorted(
        best.values(),
        key=lambda p: (len(p.pre), len(p.post), p.m0, tuple(sorted(p.pre)), tuple(sorted(p.post)), p.score),
    )

    return deduped


def remove_places_with_no_arcs(places: Iterable[PlaceCandidate]) -> List[PlaceCandidate]:
    return [p for p in places if len(p.pre) > 0 or len(p.post) > 0]


def reindex_place_candidates(
    places: list[PlaceCandidate],
) -> list[PlaceCandidate]:
    return [
        replace(p, place_id=f"p_{idx}")
        for idx, p in enumerate(places)
    ]

# -----------------------------------------------------------------------------
# Regions to places pipeline
# -----------------------------------------------------------------------------
def convert_regions_to_places(
    ts: TransitionSystemData,
    strict_regions: Regions,
    approximate_regions: Regions,
    *,
    approx_tie_policy: bool = True,
    build_complement_regions: bool = False,
    min_crossing_freq_ratio: float = 0.05,
    min_crossing_freq_amount: int = 10
):
    """
    Convert discovered regions into Petri-net places.

    The function takes both strict regions and approximate regions found on the
    transition system and maps each accepted state region to a place candidate.
    For each region, labels are classified by their dominant crossing behavior,
    determined using transition frequency, with respect to the region:
      - exit  labels become input arcs of the place,
      - enter labels become output arcs of the place,
      - non-crossing labels are ignored.
    Weak crossing evidence can be ignored using the minimum frequency ratio and 
    amount thresholds.

    Parameters
    ----------
    ts : TransitionSystemData

    strict_regions : Regions

    approximate_regions : Regions

    approx_tie_policy : bool, default=True
        Tie-breaking policy used when determining the dominant crossing type
        for approximate regions. If True, ties prefer exit over no-cross over 
        enter behavior. If False, unresolved ties return no-cross.

    build_complement_regions : bool, default=False
        If True, also generate complement regions from the given state regions
        before converting them to place candidates.

    min_crossing_freq_ratio : float, default=0.05
        Minimum relative frequency required for a crossing type to be considered
        meaningful when converting approximate regions.

    min_crossing_freq_amount : float, default=10
        Minimum absolute frequency required for a crossing type to be considered
        meaningful when converting approximate regions.

    Returns
    -------
    list[PlaceCandidate]
        Place candidates derived from the given strict and approximate regions.
    """
    approximate_regions = _prune_minimal(approximate_regions) # prune non minimal approximate region

    if build_complement_regions == True:
        temp = set()
        for approximate_region in approximate_regions:
            for strict_region in strict_regions:
                # build complement regions of approximate regions to strict regions
                if approximate_region < strict_region:
                    temp.add(frozenset(strict_region - approximate_region))
        
        approximate_regions |= temp # add complement regions to approximate regions
        approximate_regions = _prune_minimal(approximate_regions) # prune non minimal approximate region

    regions = strict_regions | approximate_regions # Undifferentiate strict regions and approximate regions 

    regions = _prune_minimal(regions) # overall non minimal regions pruning

    # build place candidates from regions
    places = build_place_candidates(
        ts=ts,
        regions=regions,
        approx_tie_policy=approx_tie_policy,
        min_crossing_freq_ratio = min_crossing_freq_ratio,
        min_crossing_freq_amount = min_crossing_freq_amount
    )

    # reduplicate identical places candidates
    places = deduplicate_identical_places(places)

    # remove isolated places candidates
    places = remove_places_with_no_arcs(places)

    # rename places candidates for readability
    places = reindex_place_candidates(places)

    # final places
    places = [
        Place(
            place_id=place.place_id,
            states=place.states,
            pre=dict(place.pre),
            post=dict(place.post),
            m0=int(place.m0),
            fm=int(place.fm),
        )
        for place in places
    ]

    return places


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.transition_system_builder import read_xes_to_ts
    from src.regions_discovery import approximate_label_splitting_algo
    from violation_cost_fn import *

    load_dotenv()

    # run ts
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

    # run approximate splitting
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

    strict_regions = result.minimal_strict_regions
    approx_regions = result.approximate_regions

    # run regions to places
    places = convert_regions_to_places(
        result.split_ts, 
        strict_regions, 
        approx_regions, 
        min_crossing_freq_ratio = 0.05,
        min_crossing_freq_amount = 2
    )
    print(places)