from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional, Literal, Dict, List, Set, FrozenSet, Tuple, TypeAlias
from datetime import datetime
import graphviz

ViewType = Literal["sequence", "set", "multiset"]

@dataclass(frozen=True)
class XesToTSConfig:
    xes_path: str
    view: ViewType = "sequence"    # sequence | set | multiset
    window: int = 2
    activity_key: str = "concept:name"
    include_data: bool = True
    direction: Literal["forward", "backward"] = "forward"   # forward | backward

@dataclass(frozen=True, slots=True)
class TSArc:
    source: str
    label: str
    target: str
    freq: int
    
    def crossing_type(self, region: Set[str]) -> str:
        src_in = self.source in region
        tgt_in = self.target in region
        if src_in and tgt_in:
            return "in"
        if (not src_in) and (not tgt_in):
            return "out"
        if (not src_in) and tgt_in:
            return "enter"
        return "exit"


@dataclass(frozen=True, slots=True)
class TransitionSystemData:
    states: Set[str]
    arcs: Set[TSArc]
    initial_states: Set[str] = field(default_factory=set)
    sink_states: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # --- invariant checks ---
        if not self.initial_states.issubset(self.states):
            raise ValueError("initial_states must be subset of states")
        
        if not self.sink_states.issubset(self.states):
            raise ValueError("sink_states must be subset of states")

        for arc in self.arcs:
            if arc.source not in self.states:
                raise ValueError(f"arc source {arc.source} not in states")
            if arc.target not in self.states:
                raise ValueError(f"arc target {arc.target} not in states")

    def copy(self) -> "TransitionSystemData":
        return TransitionSystemData(
            states=self.states.copy(),
            arcs=self.arcs.copy(),
            initial_states=self.initial_states.copy(),
            sink_states=self.sink_states.copy(),
        )

    def labels(self) -> List[str]:
        return sorted({a.label for a in self.arcs})
    
    def summarize(self, title="TS"):
        print(f"========={title}=========")
        print("States:", len(self.states))
        print("Arcs:", len(self.arcs))
        print("Labels:", len(self.labels()))
        if hasattr(self, "initial_states"):
            print("Initial states:", self.initial_states)
        else:
            print("Initial state:", None)
        if hasattr(self, "sink_states"):
            print("sink states:", self.sink_states)
        else:
            print("sink state:", None)
        print(f"========={'=' * len(title)}=========")

    def _plot(self, *, show_freq=True, title="TS"):
        graph = graphviz.Digraph(name=title)
        graph.attr(rankdir="LR")

        initial_states = self.initial_states
        sink_states = self.sink_states
        middle_states = self.states - initial_states - sink_states

        for s in initial_states:
            attrs = {"peripheries": "2"}
            graph.node(str(s), **attrs)

        for s in sorted(middle_states):
            attrs = {"shape": "circle"}
            graph.node(str(s), **attrs)
        
        for s in sink_states:
            attrs = {"peripheries": "2"}
            graph.node(str(s), **attrs)

        for arc in sorted(self.arcs, key=lambda a: (a.source, a.label, a.target)):
            label = arc.label
            if show_freq and getattr(arc, "freq", None) is not None:
                label = f"{label} ({arc.freq})"
            graph.edge(str(arc.source), str(arc.target), label=label)

        return graph
    
    def visualize(self, *, show_freq=True, title="TS"):
        graph = self._plot(show_freq = show_freq, title = title)
        graph.view()
        return

    def save_plot(
        self,
        *,
        path: Optional[Union[str, Path]] = None,
        file_name: Optional[str] = None,
        show_freq: bool = True,
        title: str = "TS",
        default_format: str = "png",
        cleanup: bool = True,
    ) -> Path:
        """
        Save the transition system plot.

        If path is None:
            Save to ../out relative to this file's src folder.
            Example: project/src/file.py -> project/out/

        If file_name is None:
            Use current date and time.
            Example: transition_system_20260628_143522.png
        """

        # ----- decide output folder -----
        if path is None:
            try:
                # Expected structure:
                # project/
                #   src/
                #     your_file.py
                #   out/
                src_dir = Path(__file__).resolve().parent
                output_dir = src_dir.parent / "out"
            except NameError:
                # Fallback for notebooks / interactive execution
                output_dir = Path.cwd() / "out"
        else:
            output_dir = Path(path)

        output_dir.mkdir(parents=True, exist_ok=True)

        # ----- decide file name -----
        if file_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"transition_system_{timestamp}.{default_format}"

        output_path = output_dir / file_name
        suffix = output_path.suffix.lower().lstrip(".")

        if not suffix:
            suffix = default_format.lower().lstrip(".")
            output_path = output_path.with_suffix(f".{suffix}")

        # ----- create graph -----
        dot = self._plot(show_freq=show_freq, title=title)

        # Save raw Graphviz source
        if suffix in {"dot", "gv"}:
            dot.save(str(output_path))
            return output_path

        # Render png/pdf/svg/etc.
        rendered_path = dot.render(
            filename=output_path.stem,
            directory=str(output_dir),
            format=suffix,
            cleanup=cleanup,
        )

        return Path(rendered_path)

States: TypeAlias = FrozenSet[str]
Regions: TypeAlias = Set[States]

@dataclass(slots=True)
class RegionsDiscoveryResult:
    split_ts: TransitionSystemData
    minimal_strict_regions: Regions
    approximate_regions: Regions
    split_map: Dict[str, str]        

PlaceSignature: TypeAlias = Tuple[
    Tuple[Tuple[str, int], ...],
    Tuple[Tuple[str, int], ...],
    int
]

@dataclass(frozen=True, slots=True)
class PlaceCandidate:
    place_id: str
    states: States
    pre: Dict[str, int]
    post: Dict[str, int]
    m0: int
    fm: int
    score: float
    origin: str   # "strict" | "approximate"

    def signature(self) -> PlaceSignature:
        return tuple(sorted(self.pre.items())), tuple(sorted(self.post.items())), self.m0

    def validate(self) -> None:
        if not self.place_id:
            raise ValueError("place_id must be non-empty")
        if self.m0 not in (0, 1):
            raise ValueError(f"m0 must be 0/1 for safe-net setting, got {self.m0}")
        if self.score < 0:
            raise ValueError(f"score must be >= 0, got {self.score}")
        if self.origin not in {"strict", "approximate"}:
            raise ValueError(f"origin must be 'strict' or 'approximate', got {self.origin}")


@dataclass(frozen=True, slots=True)
class Place:
    """
    One state-region -> one Petri net place.

    pre[label]  : Transition(label) -> Place
    post[label] : Place -> Transition(label)
    m0          : initial tokens in the place
    """
    place_id: str
    states: FrozenSet[str]
    m0: int
    fm: int
    pre: Dict[str, int] = field(default_factory=dict)
    post: Dict[str, int] = field(default_factory=dict)

    def labels(self) -> FrozenSet[str]:
        return frozenset(self.pre.keys()) | frozenset(self.post.keys())

    def signature(self) -> Tuple[Tuple[Tuple[str, int], ...], Tuple[Tuple[str, int], ...], int]:
        return tuple(sorted(self.pre.items())), tuple(sorted(self.post.items())), int(self.m0)

    def validate(self, allow_isolated: bool = False) -> None:
        if not self.place_id or not isinstance(self.place_id, str):
            raise ValueError("region_id must be a non-empty string")

        if not isinstance(self.states, frozenset):
            raise TypeError("states must be a frozenset[str]")

        for dname, d in (("pre", self.pre), ("post", self.post)):
            if not isinstance(d, dict):
                raise TypeError(f"{dname} must be a dict")
            for k, v in d.items():
                if not isinstance(k, str) or not k:
                    raise ValueError(f"{dname} keys must be non-empty strings")
                if not isinstance(v, int) or v < 0:
                    raise ValueError(f"{dname}[{k}] must be a non-negative integer")

        if not isinstance(self.m0, int) or self.m0 < 0:
            raise ValueError("m0 must be a non-negative integer")

        if not allow_isolated and len(self.pre) == 0 and len(self.post) == 0:
            raise ValueError(f"isolated region {self.place_id} is not allowed")