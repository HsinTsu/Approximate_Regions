from __future__ import annotations

from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.transition_system import algorithm as ts_discovery
from typing import Any, Literal, Set, Tuple, Dict
from .structures import TSArc, TransitionSystemData, ViewType
from collections import defaultdict, Counter
import ast
import re

def read_xes(xes_path: str):
    """
    Tries multiple PM4Py APIs for XES import to be robust across versions.
    Returns an EventLog/EventStream compatible with transition system discovery.
    """
    try:
        return xes_importer.apply(xes_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to read XES from path: {xes_path}\n"
            f"Make sure pm4py is installed and the file exists.\n"
            f"Original error: {e}"
        ) from e

def _state_name(state: Any) -> str:
    """
    Read and formalize state name into str type.
    """
    name = getattr(state, "name", None)
    if name is None:
        raise ValueError("TS has state without state name.")
    return str(name)

def _transition_source(tr: Any) -> Any:
    """
    Read from_state/source/src of the transition from TS object
    """
    for attr in ("from_state", "source", "src"):
        if hasattr(tr, attr):
            return getattr(tr, attr)
    raise ValueError("TS transition has no source/from_state attribute")

def _transition_target(tr: Any) -> Any:
    """
    Read to_state/target/tgt of the transition from TS object
    """
    for attr in ("to_state", "target", "tgt"):
        if hasattr(tr, attr):
            return getattr(tr, attr)
    raise ValueError("TS transition has no target/to_state attribute")

def _transition_label(tr: Any) -> str:
    """
    Read label of the transition from TS object
    """
    for attr in ("name", "label"):
        if hasattr(tr, attr):
            value = getattr(tr, attr)
            if value is not None:
                return str(value)
    raise ValueError("TS transition has no label/name attribute")

def _transition_freq(tr: Any) -> int:
    data = getattr(tr, "data", {}) or {}
    events = data.get("events")
    if events is None:
        return 1
    if not hasattr(events, "__len__"):
        return 1
    freq = len(events)
    if freq < 1:
        raise ValueError("Transition frequency must be >= 1")
    return int(freq)

def pm4py_ts_to_data(ts: Any) -> TransitionSystemData:
    """
    Reconstruct pm4py transition object to own defined TransitionSystem data type.
    """
    states_attr = getattr(ts, "states", None)   #get states from pm4py ts
    trans_attr = getattr(ts, "transitions", None) # get transitions from pm4py ts
    
    if states_attr is None or trans_attr is None:  # check if states or transitions are empty
        raise TypeError("Expected a PM4Py-like TS object with .states and .transitions")

    states: Set[str] = set() # store all the states in ts
    initial_states: Set[str] = set() # store all the initial states in ts
    sink_states: Set[str] = set() # store all the sink states in ts

    # identify all the states, initial states, sink states for TS
    for s in states_attr:
        s_name = _state_name(s) # read and formalize state name to str type
        states.add(s_name)  # add state into states bucket
        if not getattr(s, "incoming", None):  # if there's no incoming arc to the state => state is an initial state
            initial_states.add(s_name)
        if not getattr(s, "outgoing", None):  # if there's no outgoing arc from the state => state is a sink state
            sink_states.add(s_name)

    arcs: Set[TSArc] = set() # store all the transitions in ts by own defined TSArc type
    # transform all the arcs to TSArc type for TS, keeping source, target, label and frequency information
    for tr in trans_attr:
        src = _state_name(_transition_source(tr)) # read the source state of the transition and formalize by str type
        tgt = _state_name(_transition_target(tr)) # read the target state of the transition and formalize by str type
        label = _transition_label(tr) # read the label of the transition and formalize by str type
        freq = _transition_freq(tr) # read the frequency of the transition
        arcs.add(TSArc(src, label, tgt, freq)) # add the transition by TSArc type

    return TransitionSystemData(
        states=states,
        arcs=arcs,
        initial_states=initial_states,
        sink_states=sink_states,
    )


def _trace_labels(trace, activity_key: str = "concept:name") -> list[str]:
    """
    Extract events sequence of a trace from pm4py trace object.
    """
    return [event[activity_key] for event in trace]


def _abstract_trace_state(events: list[str], view: str):
    """
    Return a state of the given event sequence according to abstraction rule(view type).
    """
    if view == "set":
        return frozenset(events)

    if view == "sequence":
        return tuple(events)

    if view == "multiset":
        return tuple(sorted(Counter(events).items()))

    raise ValueError(f"Unsupported view: {view}")


def _get_trace_terminal_key(events: list[str], view: str, window: int, direction: str):
    """
    Construct the final state of a trace
    """
    if direction == "backward":
        terminal_events = events[-window:] # look back final {window} events when it looks backward at the final
    elif direction == "forward":
        terminal_events = [] # emtpy set when it looks forward at the final
    else:
        raise ValueError(f"Unsupported direction: {direction}") # only accept backward or forward for view direction

    return _abstract_trace_state(terminal_events, view) # return the final state of the trace


def _parse_ts_state_name(state_name: str, view: str):
    """
    Transform state from str type to other data type according to abstraction view type.
        view = set -> frozenset[str]
        view = sequence -> tuple[str]
        view = multiset -> tuple[[str, int]]
    """
    if view == "set":
        if state_name == "set()":
            return frozenset()

        value = ast.literal_eval(state_name)
        if not isinstance(value, set):
            raise ValueError(f"Expected set state, got {state_name!r}")
        return frozenset(value)

    if view == "sequence":
        value = ast.literal_eval(state_name)
        if not isinstance(value, list):
            raise ValueError(f"Expected sequence/list state, got {state_name!r}")
        return tuple(value)

    if view == "multiset":
        if state_name == "Counter()":
            return tuple()

        match = re.fullmatch(r"Counter\((.*)\)", state_name)
        if not match:
            raise ValueError(f"Expected Counter state, got {state_name!r}")

        inner = match.group(1)
        value = ast.literal_eval(inner)
        if not isinstance(value, dict):
            raise ValueError(f"Expected Counter dict body, got {state_name!r}")

        return tuple(sorted(value.items()))

    raise ValueError(f"Unsupported view: {view}")


def add_end_arcs(
    ts: TransitionSystemData,
    log,
    *,
    view: str,
    window: int,
    direction: str,
    activity_key: str = "concept:name",
) -> TransitionSystemData:
    """
    Add __END__ transition as an outgoing transition to states which is end state for some traces and connects to a singular __END__ state.
    The frequency of the transition will be the amount of traces ends at the state.
    """
    END_LABEL = "__END__"
    END_STATE = "__END__"

    # 1. Build state index for mapping own abstract state name to ts state name
    state_index = {
        _parse_ts_state_name(state_name, view): state_name
        for state_name in ts.states
    }

    # 2. Collect frequency of each terminal state
    end_freq = defaultdict(int)

    for trace in log:
        # extract event sequenct from pm4py trace object
        events = _trace_labels(trace, activity_key=activity_key)
        # get the final state name of the trace
        terminal_key = _get_trace_terminal_key(
            events,
            view=view,
            window=window,
            direction=direction,
        )
        # final state shall include in ts
        if terminal_key not in state_index:
            raise KeyError(
                f"Terminal state not found in TS: "
                f"{terminal_key!r}, view={view}, window={window}, direction={direction}"
            )

        terminal_state = state_index[terminal_key] # get the final state name in ts
        end_freq[terminal_state] += 1 # frequency of the state is considered as final state

    # 3. Extract TArc data for TS reconstruction
    arc_freq = defaultdict(int)

    for arc in ts.arcs:
        arc_freq[(arc.source, arc.label, arc.target)] += arc.freq

    # 4. Add arcs from terminal state to __END__state
    for idx, (terminal_state, freq) in enumerate(end_freq.items()):
        arc_freq[(terminal_state, END_LABEL+"#"+str(idx), END_STATE)] += freq

    # 5. Rebuild new TArcs
    new_arcs = {
        TSArc(source=src, label=label, target=tgt, freq=freq)
        for (src, label, tgt), freq in arc_freq.items()
    }

    # 6. Extract states from original TS
    new_states = set(ts.states)
    new_states.add(END_STATE)

    # 7. Set __END__ state as only sink state
    outgoing_sources = {arc.source for arc in new_arcs}
    
    # error prevention, check end states of the new ts structure
    computed_sink_states = {
        state for state in new_states
        if state not in outgoing_sources
    }
    # make sure END_STATE is the only end state
    if computed_sink_states != {END_STATE}:
        raise ValueError(
            f"Unexpected sink states after adding END arcs: {computed_sink_states}"
        )

    new_sink_states = {END_STATE}

    # 8. Return updated TS
    return TransitionSystemData(
        states=new_states,
        arcs=new_arcs,
        initial_states=set(ts.initial_states),
        sink_states=new_sink_states,
    )

def isolate_self_loop_arcs(
    ts:TransitionSystemData,
    *,
    min_self_loop_freq_ratio_to_state: float = 0.50,
    min_self_loop_freq_amount_to_state: int = 10
    ) -> Tuple[TransitionSystemData, Set[TSArc]]:
    """
    Isolate self loop arcs from TS and return a no self loop TS and a list of frequency filtered self loop arcs.
    min_self_loop_freq_ratio_to_state - a minimum outgoing(=incoming) frequency filtering ratio of self loop arcs from each state.
    min_self_loop_freq_amount_to_state - a minimum outgoing(=incoming) frequency filtering amount of self loop arcs from each state.
    """
    non_self_loop_arcs: Set[TSArc] = set()
    self_loop_arcs: Set[TSArc] = set()
    total_freq_of_each_state: Dict[str, int] = defaultdict(int)

    # calculate the total frequency of each state in TS
    for arc in ts.arcs:
        total_freq_of_each_state[arc.source] += arc.freq

    # isolate self-loop arcs and waive arcs by frequency ratio and amount filtering
    for arc in ts.arcs:
        src = arc.source
        tgt = arc.target
        label = arc.label
        freq = arc.freq

        if arc.source == arc.target:
            state_total = total_freq_of_each_state[src]
            state_ratio = freq/state_total if state_total > 0 else 0.0
            if state_ratio >= min_self_loop_freq_ratio_to_state or freq >= min_self_loop_freq_amount_to_state:
                self_loop_arcs.add(TSArc(source=src, label=label, target=tgt, freq=freq))
            else:
                pass
        else:
            non_self_loop_arcs.add(TSArc(source=src, label=label, target=tgt, freq=freq))

    # re-constructure non self-loop arcs into TS
    ts_no_self_loop = TransitionSystemData(
        states=ts.states,
        arcs=non_self_loop_arcs,
        initial_states=ts.initial_states,
        sink_states=ts.sink_states,
    )
    return ts_no_self_loop, self_loop_arcs

def rename_states(ts:TransitionSystemData) -> TransitionSystemData:
    """
    Rename states except __END__ state into S0, S1...
    """

    # read states and arcs exclude __END__ state
    states_without_end = ts.states - ts.sink_states
    original_arcs = ts.arcs
    initial_states = ts.initial_states

    # states mapping from orignal to new state name
    states_map: dict[str, str] = {}

    state_id_counter = 0

    # initial states get early ids
    for s_name in sorted(initial_states):
        states_map[s_name] = f"S{state_id_counter}"
        state_id_counter += 1

    # sink states don't change
    for s_name in sorted(ts.sink_states):
        states_map[s_name] = s_name

    # normal states get middle ids
    for s_name in sorted(states_without_end):
        if s_name not in states_map:
            states_map[s_name] = f"S{state_id_counter}"
            state_id_counter += 1

    arcs: Set[TSArc] = set()
    # rebuild arcs with rename states
    for tr in original_arcs:
        src = states_map[tr.source]
        tgt = states_map[tr.target]
        label = tr.label
        freq = tr.freq
        arcs.add(TSArc(src, label, tgt, freq))

    return TransitionSystemData(
        states=set(states_map.values()),
        arcs=arcs,
        initial_states={states_map[s] for s in initial_states},
        sink_states=ts.sink_states,
    )

def xes_to_ts(
    xes_path: str,
    view: ViewType = "sequence",
    window: int = 2,
    *,
    activity_key: str = "concept:name",
    include_data: bool = True,
    direction: Literal["forward", "backward"] = "backward",
    min_self_loop_freq_ratio_to_state: float = 0.2,
    min_self_loop_freq_amount_to_state: int = 10,
    isolate_self_loop: bool = True
) -> TransitionSystemData:
    """
    Convert an XES event log into a TransitionSystemData object using PM4Py
    view-based transition system discovery.

    The discovered transition system abstracts log prefixes according to the
    selected view and window size. By default, the TS is constructed in backward
    direction, which is useful when final behavior and suffix-based structure are
    relevant for region discovery.

    Optionally, self-loop arcs can be isolated during preprocessing. This removes
    self-loop behavior from the main transition system used for region discovery,
    while allowing relevant self-loops to be handled separately.

    Parameters
    ----------
    xes_path:
        Path to the input `.xes` event log.

    view:
        State abstraction used by PM4Py VIEW_BASED transition system discovery.

        - "sequence": states are represented by the ordered last-k activities.
        - "set": states are represented by the set of the last-k activities.
        - "multiset": states are represented by the bag/multiset of the last-k
          activities.

    window:
        Window size `k` used for state abstraction. Must be greater than or
        equal to 1.

    activity_key:
        Event attribute used as the activity label. The default is
        `"concept:name"`.

    include_data:
        Whether to include PM4Py transition/state data in the discovered TS.
        This should usually be enabled when arc frequencies are needed for
        approximate region discovery.

    direction:
        Direction used for transition system discovery.

        - "forward": construct states from prefixes in normal trace order.
        - "backward": construct states from reversed traces.

    min_state_ratio_of_self_loop:
        Minimum frequency ratio required for a self-loop arc to be considered
        relevant with respect to its source state. Self-loops below this ratio
        are treated as low-frequency behavior during self-loop preprocessing.

    isolate_self_loop:
        If True, self-loop arcs are separated from the main transition system
        before region discovery. If False, self-loop arcs remain in the returned
        transition system.

    Returns
    -------
    TransitionSystemData
        The converted transition system, optionally with self-loop arcs isolated
        according to the given preprocessing parameters.
    """
    if view not in ("sequence", "set", "multiset"):
        raise ValueError(f"Invalid view='{view}'. Must be one of: sequence|set|multiset.")
    if not isinstance(window, int) or window < 1:
        raise ValueError(f"Invalid window='{window}'. Must be an integer >= 1.")

    # read xes log
    log = read_xes(xes_path)

    # convert log to pm4py transition system object
    pm4py_ts = ts_discovery.apply(
        log,
        variant=ts_discovery.Variants.VIEW_BASED,
        parameters={
            "view": view,
            "window": window,
            "include_data": include_data,
            "activity_key": activity_key,
            "direction": direction,
        },
    )

    # convert pm4py transition system object to own defined TS data type
    ts = pm4py_ts_to_data(pm4py_ts)

    # add __END__ transitions and __END__ state to TS
    ts = add_end_arcs(ts, log, view = view, window = window, direction = direction, activity_key= activity_key)

    # rename states to S0, S1...
    ts = rename_states(ts)

    # isolate self-loop arcs if isolate_self_loop == True
    if isolate_self_loop == True:
        ts, self_loop_arcs = isolate_self_loop_arcs(
            ts, 
            min_self_loop_freq_ratio_to_state = min_self_loop_freq_ratio_to_state, 
            min_self_loop_freq_amount_to_state = min_self_loop_freq_amount_to_state)

        return ts, self_loop_arcs
    else:
        return ts, set()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    XES_PATH = os.environ["XES_PATH"]
    VIEW = os.getenv("VIEW", "sequence")
    WINDOW = int(os.getenv("WINDOW", "2"))
    ACTIVITY_KEY = os.getenv("ACTIVITY_KEY", "concept:name")
    DIRECTION = os.getenv("DIRECTION", "backward")

    ts, self_loop_arcs = xes_to_ts(
        XES_PATH,
        view=VIEW,                      # type: ignore[arg-type]
        window=WINDOW,
        activity_key=ACTIVITY_KEY,
        include_data=1,
        direction="backward",            # type: ignore[arg-type]
    )

    ts.summarize()
    ts.save_plot()
