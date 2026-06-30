# Approximate Regions Thesis Code

This repository contains experimental code for discovering Petri nets from event logs using approximate regions and label splitting.

The code is still under active development. APIs, module names, parameters, and evaluation scripts may change.

## Current Pipeline

1. Build a transition system from an XES event log.
2. Discover strict and approximate regions.
3. Select region-derived place candidates.
4. Convert selected places into a Petri net.
5. Visualize and evaluate the resulting model.

## Project Structure

```text
Thesis/
├── main.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── transition_system_builder.py
│   ├── regions_discovery.py
│   ├── regions_selection.py
│   ├── petri_net_builder.py
│   ├── visualization.py
│   ├── violation_costs.py
│   └── structures.py
├── out/                 # Generated outputs, ignored by Git
└── data_example/        # Small toy logs included for testing
```

## Data

Large event logs and evaluation datasets are not included in this repository.

Small toy example logs are provided in:

```text
data_example/
```

These logs are intended for testing and illustrating the behavior of the discovery pipeline under simple control-flow patterns and controlled noise settings.

## Configuration

Runtime parameters can be configured through environment variables.

The repository includes:

```text
.env.example
```

Create a local `.env` file based on `.env.example` when running experiments locally.

The `.env` file is not committed to Git.

## Outputs

Generated files such as Petri net visualizations, evaluation results, and intermediate outputs are written to:

```text
out/
```

The `out/` directory is intended for local generated files and should not be used to store source code.