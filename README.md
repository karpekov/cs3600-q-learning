# MDP Q-Learning Visualization

A Q-learning implementation for Markov Decision Processes with interactive visualization, designed for educational purposes in CS3600.

## Overview

This project provides:
- Q-learning algorithm implementation for different graph environments
- Interactive visualization of learning progress
- Human play mode for manual exploration
- Multiple graph types (custom rooms, simple grid, complex maze)
- Comprehensive experiment running and comparison

## Quick Start

### Basic Training
```bash
python q_learning.py                    # Train with default settings
python q_learning.py -n 1000 -s 1      # 1000 episodes, moderate stochasticity
```

### Visualization
```bash
python mdp_viz.py                       # Interactive visualization
python mdp_viz.py --human-play          # Play manually
```

### Run Experiments
```bash
python q_learning.py --run-experiments --graph-type custom_rooms
python mdp_viz.py --experiments --experiments-graph custom_rooms
```

## Key Files

- **`q_learning.py`** - Main Q-learning implementation and training
- **`mdp_viz.py`** - Interactive visualization and human play mode
- **`graph_definitions.py`** - Different environment definitions
- **`run_commands.bash`** - Complete command examples

## Graph Types

- **`custom_rooms`** - Original room layout with goal and pit
- **`simple_grid`** - Simple 3x3 grid environment
- **`complex_maze`** - Complex maze with multiple traps

## Parameters

- `-e, --epsilon` - Exploration rate (default: 0.1)
- `-a, --alpha` - Learning rate (default: 0.1)
- `-g, --gamma` - Discount factor (default: 0.9)
- `-s, --stochasticity` - Environment randomness: 0=deterministic, 1=moderate, 2=high
- `-n, --episodes` - Number of training episodes

## Installation

Requires Python 3.7+ with:
```bash
pip install pygame matplotlib numpy networkx
```

Or use the provided conda environment:
```bash
conda env create -f environment.yml
conda activate cs3600-mdp
```

## Examples

See `run_commands.bash` for comprehensive usage examples.