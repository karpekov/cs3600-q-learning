# MDP Q-Learning Room Explorer

This project implements Q-learning for exploring room environments with different graph structures and stochasticity levels.

## Quick Start

1. **Install dependencies:**
   ```bash
   conda env create -f environment.yml
   conda activate cs3600-q-learning
   ```

2. **Run Q-learning experiments:**
   ```bash
   python q_learning.py --run-experiments --graph-type custom_rooms
   ```

3. **Visualize results:**
   ```bash
   python interactive_visualizer.py                       # Interactive visualization
   python interactive_visualizer.py --human-play          # Play manually
   ```

## Advanced Usage

4. **Analyze multiple experiments:**
   ```bash
   python interactive_visualizer.py --experiments --experiments-graph custom_rooms
   ```

## Files

- **`q_learning.py`** - Main Q-learning implementation and experiment runner
- **`interactive_visualizer.py`** - Interactive visualization and human play mode
- **`graph_definitions.py`** - Graph structures (custom_rooms, simple_grid, complex_maze)
- **`environment.yml`** - Conda environment specification

## Graph Types

- `custom_rooms` - 13-state room environment with goal and pit
- `simple_grid` - Grid-based navigation environment
- `complex_maze` - Large maze environment (400+ states)

## Controls

Interactive mode supports episode navigation, speed control, and toggles for policy arrows, Q-values, and agent paths.

## Experiments

Generated experiments are stored in `q_learning_experiments/` organized by graph type and stochasticity level.

Q-learning parameters can be adjusted including epsilon (exploration), alpha (learning rate), gamma (discount), and environment stochasticity.

For detailed parameter exploration, see the experiment configurations in `q_learning.py`.

## Human Play Mode

Experience the environment manually with:
```bash
python interactive_visualizer.py --human-play --graph-type custom_rooms
```

Use keyboard or mouse to select actions, with real-time Q-value learning and visualization.