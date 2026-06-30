# Region → Petri Net Pipeline (Strict + Approximate Regions)

This pipeline describes the full transformation from state-based regions to a cleaned Petri net, including place selection and structural cleanup.

---

## Overview

```mermaid
flowchart TD
    A[Strict Regions and Approximate Regions] --> B[1. Regions -> Place Candidates<br/>Take the main crossing type as behavior<br/>and<br/>calculate again violation score with final TS]
    B --> C[2. Keep the minimum score for each group of Places]
    C --> D[3. Optional<br/>Score pruning with stricter threshold]
    D --> E[4. Remove Irrelevant Places<br/>pre = ∅ and post = ∅: <br/>provide no behavior restriction]
    E --> F[5. Dominance Redundancy Pruning<br/>pre1 ⊆ pre2<br/>and<br/>post1 ⊇ post2<br/>and<br/>set1⊆set2<br/>and<br/>m0_1 = m0_2]
    F --> G[6. Build Petri Net]
    G --> H[7. Cleanup Transitions<br/>Remove transitions with no preset or no postset]
    H --> I[8. Cleanup Places<br/>Remove places with no incoming arcs and m0 = 0 or no outgoing arcs but not include any sink states]
    I --> J{Fixed point reached}
    J -- No --> H
    J -- Yes --> K[Final Petri Net]
```