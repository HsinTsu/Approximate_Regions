from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal, Sequence, cast

from dotenv import load_dotenv

from src.transition_system_builder import ViewType, xes_to_ts
from src.regions_discovery import approximate_label_splitting_algo, classical_label_splitting_algo
from src.regions_selection import convert_regions_to_places
from src.petri_net_builder import places_to_petri_net
from src.visualization import visualize_petri_net, save_all_petri_net_outputs
from src.violation_cost_fn import *

from copy import deepcopy
from pm4py.objects.log.obj import Event
from pm4py import fitness_alignments, precision_alignments
from datetime import datetime, timedelta
from pm4py.objects.log.importer.xes import importer as xes_importer


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"


ViewChoice = Literal["sequence", "set", "multiset"]
DirectionChoice = Literal["forward", "backward"]


def _load_env() -> None:
    """Load .env from the project root, i.e. the folder containing main.py."""
    load_dotenv(dotenv_path=ENV_PATH)


def _env_required(name: str) -> str:
    import os

    value = os.getenv(name)
    if value is None or value.strip() == "":
        raise RuntimeError(
            f"Missing required environment variable: {name}\n"
            f"Please define it in {ENV_PATH}"
        )
    return value.strip()


def _env_str(name: str, default: str) -> str:
    import os

    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, str(default)).lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Environment variable {name} must be a boolean "
        f"(true/false, 1/0, yes/no), got {raw!r}"
    )


def _env_choice(name: str, default: str, choices: Sequence[str]) -> str:
    value = _env_str(name, default)
    if value not in choices:
        raise ValueError(f"Environment variable {name} must be one of {list(choices)}, got {value!r}")
    return value


def _resolve_path(path_value: str) -> Path:
    """Resolve absolute paths directly; resolve relative paths from the project root."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()

def _log_with_start_and_end_event(
    log,
    *,
    activity_key: str = "concept:name",
    timestamp_key: str = "time:timestamp",
    start_label: str = "__START__",
    end_label: str = "__END__",
    add_timestamp: bool = False,
    timestamp_delta: timedelta = timedelta(microseconds=1),
    skip_if_present: bool = True,
):
    """
    Return a copied EventLog with artificial __START__ and __END__ events.
    Original log is not modified.
    """
    log_copy = deepcopy(log)

    for trace in log_copy:
        has_start = (
            skip_if_present
            and len(trace) > 0
            and trace[0].get(activity_key) == start_label
        )

        has_end = (
            skip_if_present
            and len(trace) > 0
            and trace[-1].get(activity_key) == end_label
        )

        start_event_data = {
            activity_key: start_label
        }

        end_event_data = {
            activity_key: end_label
        }

        if add_timestamp and len(trace) > 0:
            first_ts = trace[0].get(timestamp_key)
            last_ts = trace[-1].get(timestamp_key)

            if first_ts is not None:
                start_event_data[timestamp_key] = first_ts - timestamp_delta

            if last_ts is not None:
                end_event_data[timestamp_key] = last_ts + timestamp_delta

        if not has_start:
            trace.insert(0, Event(start_event_data))

        if not has_end:
            trace.append(Event(end_event_data))

    return log_copy


def _as_serializable_metric(value):
    """Return a metric value that can be safely written to JSON/CSV."""
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _format_metric(value) -> str:
    """Format numeric metrics as decimals; keep string values such as 'Not sound'."""
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def _count_ts_states(ts) -> int:
    return len(getattr(ts, "states", []) or [])


def _count_ts_arcs(ts) -> int:
    # Your custom TransitionSystemData uses .arcs, while PM4Py TS uses .transitions.
    arcs = getattr(ts, "arcs", None)
    if arcs is not None:
        return len(arcs)
    transitions = getattr(ts, "transitions", None)
    if transitions is not None:
        return len(transitions)
    return 0


def _write_summary_outputs(
    *,
    output_dir: Path,
    base_name: str,
    dataset_name: str,
    xes_path: Path,
    view: str,
    window: int,
    activity_key: str,
    direction: str,
    min_self_loop_freq_ratio_to_state: float,
    min_self_loop_freq_amount_to_state: int,
    approx_cost_threshold: float,
    min_main_bahaviour_coverage_ratio: float,
    min_freq_to_trigger_label_splitting: int,
    min_crossing_freq_ratio: float,
    min_crossing_freq_amount: float,
    original_ts,
    final_ts,
    strict_regions,
    approx_regions,
    places,
    net,
    self_loop_arcs,
    fitness,
    precision,
) -> dict:
    """Save one detailed JSON summary and append one compact CSV row."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = {
        "view": view,
        "window": window,
        "activity_key": activity_key,
        "direction": direction,
        "min_self_loop_freq_ratio_to_state": min_self_loop_freq_ratio_to_state,
        "min_self_loop_freq_amount_to_state": min_self_loop_freq_amount_to_state,
        "approx_cost_threshold": approx_cost_threshold,
        "min_main_bahaviour_coverage_ratio": min_main_bahaviour_coverage_ratio,
        "min_freq_to_trigger_label_splitting": min_freq_to_trigger_label_splitting,
        "min_crossing_freq_ratio": min_crossing_freq_ratio,
        "min_crossing_freq_amount": min_crossing_freq_amount,
    }

    summary = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "xes_path": str(xes_path),
        "parameters": parameters,
        "transition_system": {
            "original_states": _count_ts_states(original_ts),
            "original_arcs": _count_ts_arcs(original_ts),
            "final_states": _count_ts_states(final_ts),
            "final_arcs": _count_ts_arcs(final_ts),
            "final_labels": len(final_ts.labels()) if hasattr(final_ts, "labels") else None,
            "self_loop_arcs": len(self_loop_arcs),
        },
        "regions": {
            "strict_regions": len(strict_regions),
            "approximate_regions": len(approx_regions),
        },
        "petri_net": {
            "place_candidates": len(places),
            "places": len(net.places),
            "transitions": len(net.transitions),
            "arcs": len(net.arcs),
        },
        "quality": {
            "fitness": _as_serializable_metric(fitness),
            "precision": _as_serializable_metric(precision),
        },
    }

    summary_json_path = output_dir / f"{base_name}__summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_row = {
        "run_timestamp": summary["run_timestamp"],
        "dataset_name": dataset_name,
        **parameters,
        "original_ts_states": summary["transition_system"]["original_states"],
        "original_ts_arcs": summary["transition_system"]["original_arcs"],
        "final_ts_states": summary["transition_system"]["final_states"],
        "final_ts_arcs": summary["transition_system"]["final_arcs"],
        "final_labels": summary["transition_system"]["final_labels"],
        "self_loop_arcs": summary["transition_system"]["self_loop_arcs"],
        "strict_regions": summary["regions"]["strict_regions"],
        "approximate_regions": summary["regions"]["approximate_regions"],
        "place_candidates": summary["petri_net"]["place_candidates"],
        "pn_places": summary["petri_net"]["places"],
        "pn_transitions": summary["petri_net"]["transitions"],
        "pn_arcs": summary["petri_net"]["arcs"],
        "fitness": summary["quality"]["fitness"],
        "precision": summary["quality"]["precision"],
    }

    summary_csv_path = output_dir / "evaluation_summary.csv"
    write_header = not summary_csv_path.exists()
    with summary_csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)

    summary["saved_files"] = {
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
    }
    return summary


def main() -> None:
    _load_env()

    # ----- XES -> Transition System -----
    xes_path = _resolve_path(_env_required("XES_PATH"))
    if not xes_path.exists():
        raise FileNotFoundError(f"XES file does not exist: {xes_path}")

    view = cast(ViewType, _env_choice("VIEW", "sequence", ["sequence", "set", "multiset"]))
    window = _env_int("WINDOW", 2)
    activity_key = _env_str("ACTIVITY_KEY", "concept:name")
    direction = cast(DirectionChoice, _env_choice("DIRECTION", "backward", ["forward", "backward"]))
    min_self_loop_freq_ratio_to_state = _env_float("MIN_SELF_LOOP_FREQ_RATIO_TO_STATE", 0.2)
    min_self_loop_freq_amount_to_state = _env_float("MIN_SELF_LOOP_FREQ_AMOUNT_TO_STATE", 10)

    print("=== Step 1: Build transition system ===")
    print(f"XES_PATH                           : {xes_path}")
    print(f"VIEW                               : {view}")
    print(f"WINDOW                             : {window}")
    print(f"ACTIVITY_KEY                       : {activity_key}")
    print(f"DIRECTION                          : {direction}")
    print(f"min_self_loop_freq_ratio_to_state  :{min_self_loop_freq_ratio_to_state}")
    print(f"min_self_loop_freq_amount_to_state :{min_self_loop_freq_amount_to_state}")

    log = xes_importer.apply(str(xes_path))
    log_copy = _log_with_start_and_end_event(log)

    ts, self_loop_arcs = xes_to_ts(
        str(xes_path),
        view=view,
        window=window,
        activity_key=activity_key,
        include_data=True,
        direction=direction,
        min_self_loop_freq_ratio_to_state=min_self_loop_freq_ratio_to_state,
        min_self_loop_freq_amount_to_state=min_self_loop_freq_amount_to_state
    )

    print("TS built.")
    print(f"Self-loop arcs: {len(self_loop_arcs)}")

    # ----- Approximate label splitting -----
    violation_cost_fn = get_violation_cost_fn(_env_str("VIOLATION_COST_FN", "violation_cost_violating_freq"))
    approx_cost_threshold = _env_float("APPROX_COST_THRESHOLD", 0)
    min_main_bahaviour_coverage_ratio = _env_float("MIN_MAIN_BEHAVIOUR_COVERAGE_RATIO", 1.0)
    min_freq_to_trigger_label_splitting = _env_int("MIN_FREQ_TO_TRIGGER_LABEL_SPLITTING", 10)
    if not _env_bool("APPROXIMATE_REGION_METHOD", True):
        approx_cost_threshold = 0
        min_main_bahaviour_coverage_ratio = 1
        min_freq_to_trigger_label_splitting = 0

    print("\n=== Step 2: Approximate label splitting ===")
    print(f"APPROX_COST_THRESHOLD              : {approx_cost_threshold}")
    print(f"MIN_ACCUMULATE_FREQUENCY_RATIO     : {min_main_bahaviour_coverage_ratio}")
    print(f"MIN_FREQ_TO_TRIGGER_LABEL_SPLITTING: {min_freq_to_trigger_label_splitting}")

    if _env_bool("APPROXIMATE_REGION_METHOD", True):
        result = approximate_label_splitting_algo(
            ts,
            violation_cost_fn,
            approx_cost_threshold=approx_cost_threshold,
            min_main_bahaviour_coverage_ratio=min_main_bahaviour_coverage_ratio,
            min_freq_to_trigger_label_splitting=min_freq_to_trigger_label_splitting,
        )
    else:
        result = classical_label_splitting_algo(
            ts
        )
        
    strict_regions = result.minimal_strict_regions
    approx_regions = result.approximate_regions
    final_ts = result.split_ts
    final_labels = final_ts.labels()

    print("Approximate splitting finished.")
    print(f"Final labels      : {len(final_labels)}")
    print(f"Strict regions    : {len(strict_regions)}")
    print(f"Approx. regions   : {len(approx_regions)}")

    # ----- Regions -> Places -----
    min_crossing_freq_ratio = _env_float("MIN_CROSSING_FREQ_RATIO", 0.05)
    min_crossing_freq_amount = _env_float("MIN_CROSSING_FREQ_AMOUNT", 10.0)

    if not _env_bool("APPROXIMATE_REGION_METHOD", True):
        min_crossing_freq_ratio = 0
        min_crossing_freq_amount = 0
        
    print("\n=== Step 3: Convert regions to places ===")
    print(f"MIN_CROSSING_FREQ_RATIO : {min_crossing_freq_ratio}")
    print(f"MIN_CROSSING_FREQ_AMOUNT: {min_crossing_freq_amount}")

    places = convert_regions_to_places(
        final_ts,
        strict_regions,
        approx_regions,
        min_crossing_freq_ratio=min_crossing_freq_ratio,
        min_crossing_freq_amount=min_crossing_freq_amount,
    )

    print("Places generated.")
    print(f"Places: {len(places)}")

    # ----- Places -> Petri net -----
    net_name = _env_str("NET_NAME", "Net")

    print("\n=== Step 4: Build Petri net ===")
    net, im, fm = places_to_petri_net(
        places,
        net_name=net_name,
        labels=final_labels,
        self_loop_arcs=self_loop_arcs,
    )

    try:
        fitness_temp = fitness_alignments(log_copy, net, im, fm)
        fitness = fitness_temp["average_trace_fitness"]
    except Exception as e:
        if "not a easy sound net" in str(e):
            fitness = 'Not sound'
        else:
            raise

    try:
        precision = precision_alignments(log_copy, net, im, fm)
    except Exception as e:
        if "not a easy sound net" in str(e):
            precision = 'Not sound'
        else:
            raise

    print("Petri net built.")
    print(f"Places     : {len(net.places)}")
    print(f"Transitions: {len(net.transitions)}")
    print(f"Arcs       : {len(net.arcs)}")
    print(f"fitness    : {_format_metric(fitness)}")
    print(f"precision  : {_format_metric(precision)}")

    # ----- Visualization -----
    print("\n=== Step 5: Visualize Petri net ===")
    gviz = visualize_petri_net(net, im, fm, view=False)

    if _env_bool("VIEW_PETRI_NET", True):
        gviz.view()
    else:
        print("VIEW_PETRI_NET=false, so the visualization was created but not opened.")

    # ----- Save Result -----
    if _env_bool("SAVE_RESULTS", True):
        print("\n=== Step 6: Save results ===")
        
        threshold_tag = f"{approx_cost_threshold:.2f}".replace(".", "p")
        dataset_name = xes_path.stem
        output_dir = PROJECT_ROOT / "out" / dataset_name
        base_name = f"ApproxRegion__thres_{threshold_tag}"
        pn_base_name = f"Petri_Net__thres_{threshold_tag}"
        ts_base_name = f"Splitted_TS__thres_{threshold_tag}"

        output_dir.mkdir(parents=True, exist_ok=True)

        save_all_petri_net_outputs(
            net,
            im,
            fm,
            output_dir=str(output_dir),
            base_name=pn_base_name,
        )

        final_ts.save_plot(
            path=str(output_dir),
            file_name=ts_base_name,
        )

        summary = _write_summary_outputs(
            output_dir=output_dir,
            base_name=base_name,
            dataset_name=dataset_name,
            xes_path=xes_path,
            view=view,
            window=window,
            activity_key=activity_key,
            direction=direction,
            min_self_loop_freq_ratio_to_state=min_self_loop_freq_ratio_to_state,
            min_self_loop_freq_amount_to_state=min_self_loop_freq_amount_to_state,
            approx_cost_threshold=approx_cost_threshold,
            min_main_bahaviour_coverage_ratio=min_main_bahaviour_coverage_ratio,
            min_freq_to_trigger_label_splitting=min_freq_to_trigger_label_splitting,
            min_crossing_freq_ratio=min_crossing_freq_ratio,
            min_crossing_freq_amount=min_crossing_freq_amount,
            original_ts=ts,
            final_ts=final_ts,
            strict_regions=strict_regions,
            approx_regions=approx_regions,
            places=places,
            net=net,
            self_loop_arcs=self_loop_arcs,
            fitness=fitness,
            precision=precision,
        )

        print(f"Output directory: {output_dir}")
        print(f"Summary JSON    : {summary['saved_files']['summary_json']}")
        print(f"Summary CSV     : {summary['saved_files']['summary_csv']}")
    else:
        print("\nSAVE_RESULTS=false, so PN/TS/summary outputs were not saved.")

    print("\nDone.")


if __name__ == "__main__":
    main()
