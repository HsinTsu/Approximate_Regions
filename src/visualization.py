from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.visualization.petri_net import visualizer as pn_visualizer

from .structures import Place

# ------------------------------------------------------------
# Visualization and export
# ------------------------------------------------------------
def export_pm4py_petri_net_to_pnml(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    pnml_path: str | Path,
) -> None:
    pnml_exporter.apply(net, initial_marking, str(pnml_path), final_marking=final_marking)

def visualize_petri_net(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Optional[Marking] = None,
    *,
    image_path: Optional[str | Path] = None,
    view: bool = False,
):
    if final_marking is None:
        final_marking = Marking()

    gviz = pn_visualizer.apply(net, initial_marking, final_marking)

    if image_path is not None:
        pn_visualizer.save(gviz, str(image_path))

    if view:
        pn_visualizer.view(gviz)

    return gviz


def save_all_petri_net_outputs(
    net: PetriNet,
    initial_marking: Marking,
    final_marking: Marking,
    *,
    output_dir: str | Path,
    base_name: str = "region_net",
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pnml_path = output_dir / f"{base_name}.pnml"
    png_path = output_dir / f"{base_name}.png"

    export_pm4py_petri_net_to_pnml(net, initial_marking, final_marking, pnml_path)
    visualize_petri_net(net, initial_marking, final_marking, image_path=png_path, view=False)

    return {
        "pnml": str(pnml_path),
        "image": str(png_path),
    }


# ------------------------------------------------------------
# Inspection / validation helpers
# ------------------------------------------------------------

def _arc_weight(arc) -> int:
    if hasattr(arc, "weight"):
        return int(arc.weight)
    if hasattr(arc, "properties") and isinstance(arc.properties, dict) and "weight" in arc.properties:
        return int(arc.properties["weight"])
    return 1


def summarize_regions(regions: Iterable[Place]) -> None:
    regions = list(regions)
    print("=== Regions summary ===")
    print(f"Regions: {len(regions)}")
    for r in regions:
        print(
            f"  {r.place_id}: "
            f"states={sorted(r.states)} "
            f"pre={dict(r.pre)} "
            f"post={dict(r.post)} "
            f"StartPlace={r.m0 > 0} "
            f"EndPlace={r.fm > 0}"
        )
    print("=======================")

def summarize_net(net: PetriNet, im: Marking) -> None:
    print("=== Net summary ===")
    print(f"Places: {len(net.places)}")
    print(f"Transitions: {len(net.transitions)}")
    print(f"Arcs: {len(net.arcs)}")
    print("Initial marking:")
    if len(im) == 0:
        print("  (empty)")
    else:
        for p, tok in im.items():
            print(f"  {p.name}: {tok}")

    print("\nArcs (source -> target, weight):")
    for a in net.arcs:
        print(f"  {a.source.name} -> {a.target.name} (w={_arc_weight(a)})")
    print("===================")