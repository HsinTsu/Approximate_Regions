from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Set
from collections import defaultdict

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

from .structures import Place, TSArc

def _remove_transition(net: PetriNet, transition: PetriNet.Transition) -> None:
    for arc in list(transition.in_arcs):
        net.arcs.remove(arc)
        arc.source.out_arcs.remove(arc)
        transition.in_arcs.remove(arc)

    for arc in list(transition.out_arcs):
        net.arcs.remove(arc)
        arc.target.in_arcs.remove(arc)
        transition.out_arcs.remove(arc)

    net.transitions.remove(transition)

def _remove_place(net: PetriNet, place: PetriNet.Place, im: Marking, fm: Marking) -> None:
    for arc in list(place.in_arcs):
        net.arcs.remove(arc)
        arc.source.out_arcs.remove(arc)
        place.in_arcs.remove(arc)

    for arc in list(place.out_arcs):
        net.arcs.remove(arc)
        arc.target.in_arcs.remove(arc)
        place.out_arcs.remove(arc)

    if place in im:
        del im[place]
    if place in fm:
        del fm[place]

    net.places.remove(place)


def _cleanup_petri_net(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    *,
    remove_isolated_transitions: bool = True,
    remove_no_input_zero_marking_places: bool = True,
    remove_no_output_fm_marking_places: bool = True,
    max_iteration = 1000
) -> Tuple[PetriNet, Marking, Marking]:
    """
    Structural cleanup for region-derived Petri nets.

    Rules:
    1. remove isolated transitions
    2. remove places with no incoming arcs and not initial places
    3. remove places with no outgoing arcs and not end places
    Repeats until fixed point.
    """
    iterations = 0

    while True:
        iterations += 1
        changed = False

        # ---- transitions ----
        for t in list(net.transitions):
            no_preset = len(t.in_arcs) == 0
            no_postset = len(t.out_arcs) == 0
            is_isolated = no_preset or no_postset

            if remove_isolated_transitions and is_isolated:
                _remove_transition(net, t)
                changed = True
                continue

        # ---- places ----
        for p in list(net.places):
            no_input = len(p.in_arcs) == 0
            no_output = len(p.out_arcs) == 0
            m0 = int(im.get(p, 0))
            fm_val = int(fm.get(p, 0))

            if remove_no_input_zero_marking_places and no_input and m0 == 0:
                _remove_place(net, p, im, fm)
                changed = True
                continue

            if remove_no_output_fm_marking_places and no_output and fm_val == 0:
                _remove_place(net, p, im, fm)
                changed = True
                continue

        if not changed:
            break

        if iterations > max_iteration:
            break

    return net, im, fm


# ------------------------------------------------------------
# Regions -> pm4py Petri net
# ------------------------------------------------------------

def places_to_petri_net(
    places: Iterable[Place],
    net_name: str = "RegionNet",
    labels: Optional[Iterable[str]] = None,
    self_loop_arcs: Set[TSArc] = None, 
    *,
    invalid_cleanup: bool = True,
) -> Tuple[PetriNet, Marking, Marking]:
    # ------------------------------------------------------------
    # Prepare places and labels for P/N construction
    # ------------------------------------------------------------

    places = list(places)
    # validate all the places if it's legalize
    for p in places:
        p.validate()

    # generate transitions list if labels are not given
    label_set: Set[str] = set()
    if labels is None:
        for p in places:
            label_set.update(p.pre.keys())
            label_set.update(p.post.keys())
        labels_list = sorted(label_set)
    else:
        labels_list = sorted(set(labels))

    # ------------------------------------------------------------
    # Create net, places and transitions objects
    # ------------------------------------------------------------
    # create pm4py net object
    net = PetriNet(net_name)

    # generate pm4py transitions object in net according to labels, and store the label -> transition object mapping
    t_by_label: Dict[str, PetriNet.Transition] = {}
    for label in labels_list:
        base_label = label.split("#", 1)[0] # extract the original label name
        t = PetriNet.Transition(name=f"t_{label}", label=base_label) # create pm4py transition object of the label
        net.transitions.add(t) # add transition object to net
        t_by_label[label] = t # save label to transition mapping

    t_start = PetriNet.Transition(name=f"t___START__", label="__START__") # create "starting" transition object
    net.transitions.add(t_start) # add start transition to net

    # generate pm4py places object in net, and store the place -> place object mapping
    p_by_region: Dict[str, PetriNet.Place] = {}
    for r in places:
        p = PetriNet.Place(name=r.place_id) # create pm4py place object of the place
        net.places.add(p) # add place object to net
        p_by_region[r.place_id] = p # save place to place object mapping
    
    p_start = PetriNet.Place(name="p_-1") # create an unique "starting" place object to simplify P/N
    net.places.add(p_start) # add start place to net

    # ------------------------------------------------------------
    # Connect places and transitions objects
    # ------------------------------------------------------------
    
    # add the arc to the transitions for each place
    # w default = 1
    for r in places:
        p = p_by_region[r.place_id] # get the matched place object

        for label, w in r.pre.items(): # preset of the place. Giving label and weight of arc
            if w > 0: 
                petri_utils.add_arc_from_to(t_by_label[label], p, net, weight=w) # add transition to place arc with weight w

        for label, w in r.post.items(): # postset of the place. Giving label and weight of arc
            if w > 0:
                petri_utils.add_arc_from_to(p, t_by_label[label], net, weight=w) # add place to transition arc with weight w
    
    petri_utils.add_arc_from_to(p_start, t_start, net, weight=1) # add start place to start transition arc

    # create pm4py marking object for initial marking
    im = Marking()
    for r in places:
        # add start transition to place arc if the place shall be a start place
        if r.m0 > 0:
            petri_utils.add_arc_from_to(t_start, p_by_region[r.place_id], net, weight=1)  

    im[p_start] = 1 # add the unique start place to be the only initial marking

    # create pm4py marking object for final marking
    fm = Marking()
    for r in places:
        # add the place(the only one with fm > 0) represent __END__ state to fm
        if r.fm > 0:
            fm[p_by_region[r.place_id]] = r.fm

    # clean up isolate transitions and invalid places
    if invalid_cleanup:
        net, im, fm = _cleanup_petri_net(net, im, fm)

    # ------------------------------------------------------------
    # self-loop transitions
    # one TS self-loop arc -> one duplicate visible transition
    # ------------------------------------------------------------
    if self_loop_arcs:
        # counter of each self-loop transition from different states
        # treat each self-loop transition as unique, use counter to assign label idx
        sl_count_by_label: Dict[str, int] = defaultdict(int)

        # store self-loop transitions located at each state. state -> list of self-loop transitions
        t_self_loop_by_state: Dict[str, List[PetriNet.Transition]] = defaultdict(list)

        for arc in sorted(
            self_loop_arcs,
            key=lambda a: (a.label, a.source),
        ):
            # check arc is self-loop arc
            if arc.source != arc.target:
                raise ValueError(
                    f"Expected self-loop arc, got {arc.source} --{arc.label}--> {arc.target}"
                )

            base_label = arc.label # original label of self-loop arc
            state = arc.source  # source state of self-loop arc

            sl_count_by_label[base_label] += 1 # self-loop arcs count of each label
            idx = sl_count_by_label[base_label] # increasing idx for each label

            # create self-loop transition object
            t = PetriNet.Transition(
                name=f"t_{base_label}_sl#{idx}",
                label=base_label,
            )

            net.transitions.add(t) # add transition object to net
            t_self_loop_by_state[state].append(t) # record self-loop transitions of each state
        
        cleanup_p = net.places # places objects in net
        check_p = [r for r in places if p_by_region[r.place_id] in cleanup_p] # regions to be checked for self-loop add up

        for r in check_p:
            p = p_by_region[r.place_id] # get place object
            states_in_region = r.states # get states of the region
            
            for state in states_in_region: # for each state in the region
                for t_sl in t_self_loop_by_state.get(state, []): # for each self-loop arcs to the state
                    petri_utils.add_arc_from_to(p, t_sl, net, weight=1) # add place to self-loop transition arc
                    petri_utils.add_arc_from_to(t_sl, p, net, weight=1) # add self-loop transition to place arc
    
    # cleanup invalid self-loop transitions
    if invalid_cleanup:
        net, im, fm = _cleanup_petri_net(net, im, fm)

    return net, im, fm


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from src.transition_system_builder import read_xes_to_ts
    from src.regions_discovery import approximate_label_splitting_algo
    from src.regions_selection import convert_regions_to_places
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
    print(self_loop_arcs)
    # run approximate splitting
    approx_cost_threshold = 0.14
    min_accumulate_frequency_ratio = 1
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
    final_ts = result.split_ts
    final_labels = final_ts.labels()

    # run regions to places
    places = convert_regions_to_places(
        result.split_ts, 
        strict_regions, 
        approx_regions, 
        min_crossing_freq_ratio = 0.05,
        min_crossing_freq_amount = 10
    )
    
    net, im, fm = places_to_petri_net(
        places,
        net_name="Net",
        labels = final_labels,
        self_loop_arcs = self_loop_arcs
    )

    gviz = visualize_petri_net(net, im, fm, view=False)
    gviz.view()