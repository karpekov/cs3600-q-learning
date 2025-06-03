#!/bin/bash
#
# MDP Q-Learning - Essential Commands
# ==================================
# Key commands for training and visualizing Q-learning with recent improvements

echo "MDP Q-Learning - Essential Commands"
echo "=================================="

echo
echo "1. Basic Training with Optimistic Initialization"
echo "   (Helps exploration in complex environments)"
python q_learning.py --graph-type complex_maze -n 5000 --optimistic-init 100.0 --epsilon 0.1 --alpha 0.2

echo
echo "2. Training with Epsilon and Alpha Decay"
echo "   (Better convergence through parameter decay)"
python q_learning.py --graph-type complex_maze -n 10000 --epsilon 0.8 --epsilon-decay 0.999 --alpha 0.2 --alpha-decay-rate 0.01 --optimistic-init 50.0

echo
echo "3. Run Comprehensive Experiments"
echo "   (Tests multiple parameter combinations, organized by stochasticity level)"
python q_learning.py --run-experiments --graph-type complex_maze

echo
echo "4. Visualize Experiments"
echo "   (Interactive visualization of all experiments with nested directory support)"
python mdp_viz.py --experiments --graph-type complex_maze
python mdp_viz.py --experiments --graph-type complex_maze --episode-skip 50

echo
echo "5. Human Play Mode"
echo "   (Play manually to understand the environment)"
python mdp_viz.py --human-play --graph-type complex_maze

echo
echo "6. 🎓 Debug Mode - Learn Q-Learning Step-by-Step"
echo "   (Interactive tutorial showing Q-value updates and epsilon-greedy decisions)"
python q_learning.py --debug

echo
echo "7. Advanced Debug with Custom Graph"
echo "   (Debug mode with different environment for comparison)"
python q_learning.py --debug --graph-type simple_grid

echo
echo "Generated Directory Structure:"
echo "  q_learning_experiments/complex_maze/s_0/  - Deterministic experiments"
echo "  q_learning_experiments/complex_maze/s_1/  - Stochastic experiments"

echo
echo "Key Parameters:"
echo "  --optimistic-init VALUE    - Initial Q-value (encourages exploration)"
echo "  --epsilon-decay RATE       - Exploration decay per episode"
echo "  --alpha-decay-rate RATE    - Learning rate decay"
echo "  --graph-type TYPE          - custom_rooms/simple_grid/complex_maze"