"""
Simple graph visualization tool

Sample usage:
    >>> python viz_graph.py complex_maze
    >>> python viz_graph.py custom_rooms
    >>> python viz_graph.py simple_grid
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from graph_definitions import get_graph

def viz_graph(graph_type="complex_maze"):
    """Simple graph visualization"""

    # Create graphs directory if it doesn't exist
    graphs_dir = "graphs"
    os.makedirs(graphs_dir, exist_ok=True)

    # Get graph data
    adj, rewards, coords = get_graph(graph_type)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Draw connections
    for state, neighbors in adj.items():
        x1, y1 = coords[state]
        for neighbor in neighbors:
            x2, y2 = coords[neighbor]
            plt.plot([x1, x2], [y1, y2], 'b-', alpha=0.5, linewidth=1)

    # Draw nodes
    for state, (x, y) in coords.items():
        if state == "S":
            plt.scatter(x, y, c='orange', s=200, label='Start' if state == "S" else "")
        elif state == "GOAL":
            plt.scatter(x, y, c='green', s=200, label='Goal')
        elif state in rewards:
            plt.scatter(x, y, c='red', s=200, label='Trap' if len([s for s in rewards if s != "GOAL"]) == 1 else "")
        else:
            plt.scatter(x, y, c='lightblue', s=150)

        # Add labels
        plt.text(x, y+0.15, state, ha='center', fontsize=8, fontweight='bold')
        if state in rewards:
            plt.text(x, y-0.2, f'{rewards[state]:+}', ha='center', fontsize=7, color='black')

    plt.title(f'{graph_type.replace("_", " ").title()} Graph Structure')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save in graphs folder
    plt.tight_layout()
    output_path = os.path.join(graphs_dir, f'{graph_type}_viz.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Graph saved as {output_path}")
    plt.show()

if __name__ == "__main__":
    import sys
    graph_type = sys.argv[1] if len(sys.argv) > 1 else "complex_maze"
    viz_graph(graph_type)