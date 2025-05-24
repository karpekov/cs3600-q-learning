#!/bin/bash
#
# MDP Room Exploration - Command Examples
# =====================================
# This file contains common command examples for training and visualizing MDP Q-learning

echo "MDP Q-Learning Command Examples"
echo "=============================="

echo
echo "1. Basic Training (single run)"
python q_learning.py

echo
echo "2. Training with Different Stochasticity Levels"
python q_learning.py -s 0 -n 1000  # Deterministic
python q_learning.py -s 1 -n 1000  # Moderate stochastic
python q_learning.py -s 2 -n 1000  # High stochastic

echo
echo "3. Training on Different Graph Types"
python q_learning.py --graph-type custom_rooms -n 500  # Original room layout
python q_learning.py --graph-type simple_grid -n 500   # Simple 3x3 grid
python q_learning.py --graph-type complex_maze -n 500  # Complex maze

echo
echo "4. Custom Parameters"
python q_learning.py -e 0.2 -a 0.1 -g 0.9 -c -1.0 -n 500

echo
echo "5. Interactive Visualization"
python mdp_viz.py

echo
echo "6. Visualization with Episode Skipping"
python mdp_viz.py --episode-skip 10

echo
echo "7. Static Visualization"
python mdp_viz.py --static-step 100

echo
echo "8. Human Play Mode"
python mdp_viz.py --human-play
python mdp_viz.py --human-play --graph-type simple_grid
python mdp_viz.py --human-play --graph-type complex_maze
python mdp_viz.py --human-play --graph-type custom_rooms

echo
echo "9. Comprehensive Experiments"
# Run comprehensive experiments for custom_rooms (18 combinations)
python q_learning.py --run-experiments --graph-type custom_rooms

# Run experiments for other graph types
python q_learning.py --run-experiments --graph-type simple_grid
python q_learning.py --run-experiments --graph-type complex_maze

echo
echo "10. Experiment Visualization"
# View all experiments from all graph types
python mdp_viz.py --experiments --episode-skip 50

# View experiments from specific graph type
python mdp_viz.py --experiments --experiments-graph custom_rooms --episode-skip 50
python mdp_viz.py --experiments --experiments-graph simple_grid --episode-skip 50
python mdp_viz.py --experiments --experiments-graph complex_maze --episode-skip 50

echo
echo "11. Learning Rate Experiments"
python q_learning.py -a 0.1 -n 1000 -s 1  # Standard learning
python q_learning.py -a 0.5 -n 1000 -s 1  # Fast learning

echo
echo "12. Exploration Rate Experiments"
python q_learning.py -e 0.1 -n 1000 -s 1  # Low exploration
python q_learning.py -e 0.4 -n 1000 -s 1  # High exploration

echo
echo "Available Options:"
echo "Graph types: --graph-type custom_rooms/simple_grid/complex_maze"
echo "Stochasticity: -s 0 (deterministic), 1 (moderate), 2 (high)"
echo "Parameters: -e (epsilon), -a (alpha), -g (gamma), -c (step cost), -n (episodes)"

echo
echo "Generated Data Directories:"
echo "  q_learning_data/           - Single run data"
echo "  q_learning_experiments/custom_rooms/  - Original room layout experiments"
echo "  q_learning_experiments/simple_grid/   - Simple grid experiments"
echo "  q_learning_experiments/complex_maze/  - Complex maze experiments"
echo "  visualizations/            - Static visualization outputs"

echo
echo "Experiment Filtering:"
echo "  --experiments-graph custom_rooms      - View only custom_rooms experiments"
echo "  --experiments-graph simple_grid       - View only simple_grid experiments"
echo "  --experiments-graph complex_maze      - View only complex_maze experiments"