# Specific training:
python q_learning.py --graph-type complex_maze -n 10000 --epsilon 0.8 --epsilon-decay 0.999 --alpha 0.2 --alpha-decay-rate 0.01 --optimistic-init 100.0

# Experiments:
python q_learning.py --run-experiments --graph-type complex_maze
python interactive_visualizer.py --experiments --graph-type complex_maze

# Human play:
python interactive_visualizer.py --human-play --graph-type simple_grid
python interactive_visualizer.py --human-play --graph-type complex_maze

# Debug Mode
python q_learning.py --debug