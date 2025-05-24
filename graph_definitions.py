"""
Graph Definitions for MDP Room Exploration
==========================================

This module contains graph definitions for different room layouts that can be used
in the MDP visualization system. Each graph defines the room adjacency structure,
terminal rewards, and room coordinates for visualization.
"""

# Available graph types
AVAILABLE_GRAPHS = ["custom_rooms", "simple_grid", "complex_maze"]

def create_room_graph():
    """Create the default adjacency structure for the room graph"""
    adj = {
        "S": ["A"],
        "A": ["S", "L", "F", "K"],
        "L": ["A", "G"],
        "G": ["L", "W"],
        "W": ["G", "R"],
        "R": ["W", "J"],
        "F": ["A", "T", "J"],
        "K": ["A", "T", "D"],
        "T": ["F", "K", "H", "D", "Z"],  # pit   –10
        "D": ["K", "T", "Z"],
        "Z": ["D", "T", "H"],
        "H": ["T", "Z", "J"],
        "J": ["F", "R", "H"],            # goal  +10
    }
    terminal_rewards = {"J": 10.0, "T": -10.0}

    # Room coordinates for visualization
    room_coords = {
        "S": (0, 4),
        "A": (2, 4),
        "L": (2, 6),
        "G": (4, 6),
        "W": (6, 6),
        "R": (8, 6),
        "F": (5, 4),
        "K": (2, 2),
        "T": (5, 2),
        "D": (4, 0),
        "Z": (6, 0),
        "H": (8, 2),
        "J": (8, 4)
    }

    return adj, terminal_rewards, room_coords

def create_simple_grid():
    """Classic RL 4X3 grid layout"""
    adj = {
        "S": ["N5", "N8"],
        "N1": ["N5", "N2"],
        "N2": ["N1", "N3"],
        "N3": ["N2", "N4", "N6"],
        "N4": ["N3", "N7"],
        "N5": ["S", "N1"],
        "N6": ["N3", "N9", "N7"],
        "N7": ["N4", "N6", "N10"],
        "N8": ["S", "N9"],
        "N9": ["N6", "N8", "N10"],
        "N10": ["N7", "N9"],

    }
    terminal_rewards = {"N4": 1.0, "N7": -1.0}

    room_coords = {
        "S": (0, 0),
        "N1": (0, 6),
        "N2": (3, 6),
        "N3": (6, 6),
        "N4": (9, 6),
        "N5": (0, 3),
        "N6": (6, 3),
        "N7": (9, 3),
        "N8": (3, 0),
        "N9": (6, 0),
        "N10": (9, 0)
    }

    return adj, terminal_rewards, room_coords

def create_complex_maze():
    """Create a more complex maze-like structure"""
    adj = {
        "S": ["A"],                          # Start
        "A": ["S", "B", "C"],               # First junction
        "B": ["A", "D"],                    # Upper path
        "C": ["A", "E", "F"],               # Lower path branch
        "D": ["B", "G", "H"],               # Upper junction
        "E": ["C", "I"],                    # Lower branch
        "F": ["C", "J"],                    # Alternative lower path
        "G": ["D", "K"],                    # Path to reward
        "H": ["D", "L"],                    # Path to danger
        "I": ["E", "M"],                    # Lower maze part
        "J": ["F", "N"],                    # Alternative lower route
        "K": ["G", "GOAL"],                 # Near goal
        "L": ["H", "TRAP1"],                # Near trap
        "M": ["I", "TRAP2"],                # Lower trap
        "N": ["J", "GOAL"],                 # Alternative to goal
        "GOAL": ["K", "N"],                 # Victory!
        "TRAP1": ["L"],                     # Dead end trap
        "TRAP2": ["M"],                     # Another trap
    }
    terminal_rewards = {"GOAL": 15.0, "TRAP1": -8.0, "TRAP2": -6.0}

    room_coords = {
        "S": (0, 3),
        "A": (1, 3),
        "B": (2, 4),
        "C": (2, 2),
        "D": (3, 4),
        "E": (3, 1),
        "F": (3, 2),
        "G": (4, 4),
        "H": (4, 3),
        "I": (4, 1),
        "J": (4, 2),
        "K": (5, 4),
        "L": (5, 3),
        "M": (5, 1),
        "N": (5, 2),
        "GOAL": (6, 3),
        "TRAP1": (6, 4),
        "TRAP2": (6, 1)
    }

    return adj, terminal_rewards, room_coords

def get_graph(graph_type="custom_rooms"):
    """
    Get a graph definition by type

    Args:
        graph_type (str): Type of graph to create. Options:
            - "custom_rooms": Original room layout with pit and goal
            - "simple_grid": Simple 3x3 grid layout
            - "complex_maze": More complex maze-like structure

    Returns:
        tuple: (adjacency_dict, terminal_rewards_dict, room_coordinates_dict)
    """
    if graph_type == "custom_rooms":
        return create_room_graph()
    elif graph_type == "simple_grid":
        return create_simple_grid()
    elif graph_type == "complex_maze":
        return create_complex_maze()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}. Available types: {AVAILABLE_GRAPHS}")

def list_available_graphs():
    """List all available graph types with descriptions"""
    descriptions = {
        "custom_rooms": "Original room layout with goal (+10) and pit (-10)",
        "simple_grid": "Simple 3x3 grid with goal (+5) and trap (-5)",
        "complex_maze": "Complex maze with goal (+15) and multiple traps (-8, -6)"
    }

    print("Available graph types:")
    for graph_type in AVAILABLE_GRAPHS:
        print(f"  - {graph_type}: {descriptions[graph_type]}")

    return AVAILABLE_GRAPHS

if __name__ == "__main__":
    # Demo the available graphs
    list_available_graphs()

    print("\nTesting graph creation:")
    for graph_type in AVAILABLE_GRAPHS:
        adj, rewards, coords = get_graph(graph_type)
        print(f"\n{graph_type}:")
        print(f"  States: {len(adj)}")
        print(f"  Terminal states: {list(rewards.keys())}")
        print(f"  Rewards: {rewards}")