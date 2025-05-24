#!/usr/bin/env python3
"""
q_learning.py
Q-Learning implementation for MDP room exploration

usage:  python q_learning.py                   # uses defaults
        python q_learning.py -e 0.1 -a 0.1 -g 0.9 -n 1000  # custom parameters
        python q_learning.py --run-experiments  # run multiple parameter configurations
"""

import sys
import argparse
import random
import math
import json
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
from graph_definitions import get_graph, list_available_graphs, AVAILABLE_GRAPHS

# Helper functions
# create_room_graph() and ROOM_COORDS moved to graph_definitions.py

def build_transition_model(adj, terminal_rewards, stochasticity=1):
    """Build transition model with configurable stochasticity levels

    Args:
        stochasticity:
            0 = deterministic (1.0 intended)
            1 = moderate stochastic (4/6 intended, 1/6 left, 1/6 right)
            2 = high stochastic (2/6 intended, 2/6 left, 2/6 right)
    """
    model = defaultdict(dict)

    # Define stochasticity levels
    if stochasticity == 0:
        # Deterministic: always go intended direction
        probs = (1.0, 0.0, 0.0)
    elif stochasticity == 1:
        # Moderate: 4/6 intended, 1/6 left, 1/6 right
        probs = (4/6, 1/6, 1/6)
    elif stochasticity == 2:
        # High: 2/6 intended, 2/6 left, 2/6 right
        probs = (2/6, 2/6, 2/6)
    else:
        raise ValueError(f"Invalid stochasticity level: {stochasticity}. Must be 0, 1, or 2.")

    for s, nbrs in adj.items():
        if s in terminal_rewards:
            continue
        n = len(nbrs)
        for i, nxt in enumerate(nbrs):
            left = nbrs[(i - 1) % n]
            right = nbrs[(i + 1) % n]
            model[s][i] = [(probs[0], nxt), (probs[1], left), (probs[2], right)]
    return model

class RoomEnvironment:
    def __init__(self, adj, terminal_rewards, step_cost=-0.05, stochasticity=1):
        self.adj = adj
        self.terminal_rewards = terminal_rewards
        self.stochasticity = stochasticity
        self.transition_model = build_transition_model(adj, terminal_rewards, stochasticity)
        self.step_cost = step_cost
        self.reset()

    def reset(self):
        # Start at state S
        self.current_state = "S"
        return self.current_state

    def get_available_actions(self, state):
        if state in self.terminal_rewards:
            return []
        return list(range(len(self.adj[state])))

    def step(self, action):
        if self.current_state in self.terminal_rewards:
            return self.current_state, 0, True, {}

        # Sample from transition distribution
        transitions = self.transition_model[self.current_state][action]
        probs = [p for p, _ in transitions]
        next_states = [s for _, s in transitions]
        next_state = random.choices(next_states, weights=probs)[0]

        # Calculate reward
        reward = self.step_cost + self.terminal_rewards.get(next_state, 0.0)

        # Update current state
        self.current_state = next_state

        # Check if terminal
        done = next_state in self.terminal_rewards

        return next_state, reward, done, {}

class QLearningAgent:
    def __init__(self, env, adj, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.env = env
        self.adj = adj
        self.epsilon = epsilon  # exploration rate
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.episode_history = []
        self.current_episode = []
        self.total_reward_history = []
        self.episode_length_history = []

    def get_action(self, state):
        available_actions = self.env.get_available_actions(state)
        if not available_actions:
            return None

        # Epsilon-greedy policy
        if random.random() < self.epsilon:
            # Explore: random action
            return random.choice(available_actions)
        else:
            # Exploit: best action based on Q-values
            return self.get_best_action(state, available_actions)

    def get_best_action(self, state, available_actions=None):
        if available_actions is None:
            available_actions = self.env.get_available_actions(state)

        if not available_actions:
            return None

        # Find action with highest Q-value
        best_value = float('-inf')
        best_actions = []

        for action in available_actions:
            q_value = self.q_values[state][action]
            if q_value > best_value:
                best_value = q_value
                best_actions = [action]
            elif q_value == best_value:
                best_actions.append(action)

        # Break ties randomly
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        # Q-learning update
        best_next_value = 0
        next_actions = self.env.get_available_actions(next_state)
        if next_actions:
            best_next_action = self.get_best_action(next_state, next_actions)
            best_next_value = self.q_values[next_state][best_next_action]

        # Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        self.q_values[state][action] += self.alpha * (
            reward + self.gamma * best_next_value - self.q_values[state][action]
        )

    def train_episode(self):
        state = self.env.reset()
        self.current_episode = []
        total_reward = 0
        step_count = 0

        while True:
            # Select action
            action = self.get_action(state)
            if action is None:
                break

            # Take action
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward

            # Store transition for visualization
            intended_direction = self.adj[state][action]
            transition_data = {
                "step": step_count,
                "state": state,
                "action": action,
                "intended": intended_direction,
                "next_state": next_state,
                "reward": reward,
                "q_value": self.q_values[state][action]
            }
            self.current_episode.append(transition_data)

            # Update Q-values
            self.update(state, action, reward, next_state)

            # Move to next state
            state = next_state
            step_count += 1

            if done:
                break

        self.episode_history.append({
            "steps": self.current_episode,
            "total_reward": total_reward,
            "step_count": step_count
        })

        self.total_reward_history.append(total_reward)
        self.episode_length_history.append(step_count)

        return total_reward, step_count

    def train(self, num_episodes=1000):
        rewards = []
        for i in range(num_episodes):
            episode_reward, steps = self.train_episode()
            rewards.append(episode_reward)
            if (i+1) % 100 == 0:
                print(f"Episode {i+1}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {steps}")
        return rewards

    def get_policy(self):
        policy = {}
        for state in self.adj.keys():
            if state in self.env.terminal_rewards:
                policy[state] = "-"
                continue

            best_action = self.get_best_action(state)
            if best_action is None:
                policy[state] = "-"
            else:
                policy[state] = self.adj[state][best_action]  # Show intended direction

        return policy

    def get_visualization_data(self):
        # Prepare data for visualization
        vis_data = {
            "environment": {
                "adjacency": self.adj,
                "terminal_rewards": self.env.terminal_rewards,
                "step_cost": self.env.step_cost,
                "stochasticity": self.env.stochasticity
            },
            "agent": {
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma
            },
            "q_values": {s: dict(actions) for s, actions in self.q_values.items()},
            "policy": self.get_policy(),
            "episodes": self.episode_history,
            "total_reward_history": self.total_reward_history,
            "episode_length_history": self.episode_length_history
        }
        return vis_data

    def save_visualization_data(self, filename="q_learning_data.json"):
        """Save data for visualization"""
        data = self.get_visualization_data()
        os.makedirs('q_learning_data', exist_ok=True)

        # If filename doesn't include directory, put it in the data directory
        if '/' not in filename:
            filename = f'q_learning_data/{filename}'

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Exploration data saved to {filename}")
        return filename

def create_static_visualization(data_file, output_dir, room_coords=None):
    """Create static visualization of Q-learning progress and policy"""

    # Get room coordinates if not provided
    if room_coords is None:
        _, _, room_coords = get_graph()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    with open(data_file, 'r') as f:
        data = json.load(f)

    # Extract parameters for the title
    epsilon = data['agent']['epsilon']
    alpha = data['agent']['alpha']
    gamma = data['agent']['gamma']
    step_cost = data['environment']['step_cost']
    stochasticity = data['environment'].get('stochasticity', 1)  # Default to 1 for backward compatibility

    # Extract policy and Q-values
    policy = data['policy']
    q_values = data['q_values']
    adj = data['environment']['adjacency']
    terminal_rewards = data['environment']['terminal_rewards']

    # Color definitions - using more attractive colors
    colors = {
        'background': '#f8f9fa',
        'room': '#f0f0f080',
        'terminal_positive': '#a8e6cf',  # Light green
        'terminal_negative': '#ffaaa7',  # Light red
        'arrow': '#3d5a80',              # Dark blue
        'text': '#293241',               # Dark blue-gray
        'grid': '#e0e0e0',               # Light gray
        'best_q': '#2a9d8f',             # Teal for best Q-value
        'other_q': '#8a8a8a'             # Gray for other Q-values
    }

    # Create a new figure with a specific size and high DPI for clarity
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    # Set a nice background color
    fig.patch.set_facecolor(colors['background'])
    ax.set_facecolor(colors['background'])

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set up axes
    ax.set_xlim(-1, 9)
    ax.set_ylim(-1, 7)
    ax.set_aspect('equal')

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a subtle grid
    ax.grid(True, linestyle='--', alpha=0.3, color=colors['grid'])

    # Draw edges between rooms
    for room, neighbors in adj.items():
        x1, y1 = room_coords[room]
        for neighbor in neighbors:
            x2, y2 = room_coords[neighbor]
            ax.plot([x1, x2], [y1, y2], color=colors['grid'], linewidth=1, alpha=0.6, zorder=1)

    # Draw rooms
    room_size = 0.8
    for room, (x, y) in room_coords.items():
        # Choose color based on terminal state
        if room in terminal_rewards:
            if terminal_rewards[room] > 0:
                color = colors['terminal_positive']
                label = f"{room}\n+{terminal_rewards[room]}"
            else:
                color = colors['terminal_negative']
                label = f"{room}\n{terminal_rewards[room]}"
        else:
            color = colors['room']
            label = room

        # Draw room - removed unsupported boxstyle parameter
        rect = patches.Rectangle(
            (x - room_size/2, y - room_size/2),
            room_size, room_size,
            linewidth=1.5,
            edgecolor=colors['text'],
            facecolor=color,
            alpha=0.7,
            zorder=2
        )
        ax.add_patch(rect)

        # Add room label
        ax.text(
            x, y, label,
            ha='center', va='center',
            fontsize=10, fontweight='medium',
            color=colors['text'],
            zorder=3
        )

        # Add Q-values for non-terminal states
        if room in q_values and room not in terminal_rewards:
            values = q_values[room]
            if values:
                # Find max Q-value and action
                max_q_action = max(values.items(), key=lambda x: x[1])[0]

                # Calculate vertical offset for text
                y_offset = -0.2
                line_height = 0.16

                # Show all Q-values
                for action_str, q_val in sorted(values.items(), key=lambda x: int(x[0])):
                    # Get target room for this action
                    action_idx = int(action_str)
                    if action_idx < len(adj[room]):
                        target = adj[room][action_idx]

                        # Format text with action→target: value
                        q_text = f"{action_idx}→{target}: {q_val:.2f}"

                        # Choose color based on whether this is max Q-value
                        if action_str == max_q_action:
                            text_color = colors['best_q']
                            text_weight = 'bold'
                            text_alpha = 1.0
                        else:
                            text_color = colors['other_q']
                            text_weight = 'normal'
                            text_alpha = 0.7

                        # Add the Q-value text
                        ax.text(
                            x, y + y_offset,
                            q_text,
                            ha='center', va='center',
                            fontsize=8,
                            color=text_color,
                            alpha=text_alpha,
                            fontweight=text_weight,
                            zorder=3
                        )

                        # Update offset for next line
                        y_offset -= line_height

    # Draw policy arrows
    for state, target in policy.items():
        if target == "-":
            continue

        x1, y1 = room_coords[state]
        x2, y2 = room_coords[target]

        # Calculate the direction vector
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        if length > 0:
            # Normalize
            dx, dy = dx / length, dy / length

            # Draw arrow (not all the way to the target)
            arrow_length = 0.3
            ax.arrow(
                x1, y1,
                dx * arrow_length, dy * arrow_length,
                head_width=0.15, head_length=0.15,
                fc=colors['arrow'], ec=colors['arrow'],
                length_includes_head=True,
                zorder=4,
                alpha=0.9
            )

    # Add title and metadata
    title = f"Optimal Policy: ε={epsilon}, α={alpha}, γ={gamma}, step_cost={step_cost}, stochasticity={stochasticity}"
    ax.set_title(title, fontsize=14, fontweight='bold', color=colors['text'], pad=20)

    # Add metadata as text
    steps_taken = len(data['episodes'])
    last_rewards = data['total_reward_history'][-10:]
    avg_reward = sum(last_rewards) / len(last_rewards)

    metadata = f"Episodes: {steps_taken}  |  Avg Final Reward: {avg_reward:.2f}"
    fig.text(0.5, 0.02, metadata, ha='center', fontsize=12, color=colors['text'])

    # Add a subtle border
    fig.patch.set_linewidth(2)
    fig.patch.set_edgecolor(colors['grid'])

    # Save the visualization with transparent background
    output_file = os.path.join(output_dir, f"policy_e{epsilon}_a{alpha}_g{gamma}_c{step_cost}_s{stochasticity}.png")
    fig.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print(f"Static visualization saved to {output_file}")

    # Also create a reward/steps history plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=150, sharex=True)
    fig.patch.set_facecolor(colors['background'])

    # Plot reward history
    episodes = range(1, len(data['total_reward_history'])+1)

    # Calculate moving average
    window = min(50, len(data['total_reward_history']))
    if window > 0:
        reward_moving_avg = [sum(data['total_reward_history'][max(0,i-window):i])/min(i,window)
                            for i in range(1, len(data['total_reward_history'])+1)]
    else:
        reward_moving_avg = data['total_reward_history']

    ax1.plot(episodes, data['total_reward_history'], alpha=0.3, color=colors['arrow'], label='Reward')
    ax1.plot(episodes, reward_moving_avg, color=colors['arrow'], linewidth=2, label='Moving Avg')
    ax1.set_ylabel('Reward', color=colors['text'])
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_title(f'Training Progress: ε={epsilon}, α={alpha}, γ={gamma}, step_cost={step_cost}, stochasticity={stochasticity}',
                 fontsize=12, color=colors['text'])
    ax1.legend()

    # Plot episode length history
    if window > 0:
        steps_moving_avg = [sum(data['episode_length_history'][max(0,i-window):i])/min(i,window)
                           for i in range(1, len(data['episode_length_history'])+1)]
    else:
        steps_moving_avg = data['episode_length_history']

    ax2.plot(episodes, data['episode_length_history'], alpha=0.3, color='#ff7b00', label='Steps')
    ax2.plot(episodes, steps_moving_avg, color='#ff7b00', linewidth=2, label='Moving Avg')
    ax2.set_xlabel('Episode', color=colors['text'])
    ax2.set_ylabel('Steps', color=colors['text'])
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()

    # Set common style
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_alpha(0.3)
        ax.spines['left'].set_alpha(0.3)
        ax.tick_params(colors=colors['text'])

    plt.tight_layout()
    history_file = os.path.join(output_dir, f"history_e{epsilon}_a{alpha}_g{gamma}_c{step_cost}_s{stochasticity}.png")
    plt.savefig(history_file, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

    print(f"Training history visualization saved to {history_file}")

    return output_file, history_file

def run_experiments(graph_type="custom_rooms"):
    """Run multiple experiments with different parameter settings for a specific graph type"""
    # Create directory structure organized by graph type
    base_dir = "q_learning_experiments"
    graph_dir = os.path.join(base_dir, graph_type)
    os.makedirs(graph_dir, exist_ok=True)

    # Base parameter combinations to try
    base_configs = [
        {"step_cost": 0, "gamma": 0.9, "epsilon": 0.1, "alpha": 0.1},
        {"step_cost": -0.4, "gamma": 0.9, "epsilon": 0.1, "alpha": 0.1},
        {"step_cost": -1, "gamma": 0.9, "epsilon": 0.1, "alpha": 0.1},
        {"step_cost": -2, "gamma": 0.9, "epsilon": 0.1, "alpha": 0.1},
        {"step_cost": -2, "gamma": 1.0, "epsilon": 0.1, "alpha": 0.1},
        {"step_cost": -2, "gamma": 0.6, "epsilon": 0.1, "alpha": 0.1},
        {"step_cost": -2, "gamma": 0.9, "epsilon": 0.4, "alpha": 0.1},
    ]
    config = base_configs.copy()
    # Expand configs to include all stochasticity levels
    configs = []
    for base_config in base_configs:
        for stochasticity in [0, 1]:  # All three stochasticity levels
            config = base_config.copy()
            config["stochasticity"] = stochasticity
            config["graph_type"] = graph_type  # Add graph type to config
            configs.append(config)

    results = []

    print(f"\n{'='*60}")
    print(f"RUNNING EXPERIMENTS FOR GRAPH TYPE: {graph_type.upper()}")
    print(f"{'='*60}")

    for i, config in enumerate(configs):
        print(f"\nRunning experiment {i+1}/{len(configs)} for {graph_type}")
        print(f"Parameters: ε={config['epsilon']}, α={config['alpha']}, γ={config['gamma']}, step_cost={config['step_cost']}, stochasticity={config['stochasticity']}")

        # Create experiment directory
        exp_dir = os.path.join(graph_dir, f"exp_{i+1}_e{config['epsilon']}_g{config['gamma']}_c{config['step_cost']}_s{config['stochasticity']}")
        os.makedirs(exp_dir, exist_ok=True)

        adj, terminal_rewards, room_coords = get_graph(graph_type)
        env = RoomEnvironment(adj, terminal_rewards, step_cost=config['step_cost'],
                             stochasticity=config['stochasticity'])
        agent = QLearningAgent(
            env, adj,
            epsilon=config['epsilon'],
            alpha=config['alpha'],
            gamma=config['gamma']
        )

        # Train agent
        rewards = agent.train(num_episodes=1000)

        # Save data
        data_file = os.path.join(exp_dir, "exploration_data.json")
        data_file = agent.save_visualization_data(data_file)

        # Create static visualization
        viz_file, history_file = create_static_visualization(data_file, exp_dir, room_coords)

        # Record results
        results.append({
            "config": config,
            "data_file": data_file,
            "viz_file": viz_file,
            "history_file": history_file,
            "final_avg_reward": sum(rewards[-10:]) / 10
        })

    # Print summary
    print("\n" + "="*80)
    print(f"EXPERIMENT SUMMARY FOR {graph_type.upper()}")
    print("="*80)
    for i, result in enumerate(results):
        config = result["config"]
        print(f"\nExperiment {i+1}:")
        print(f"  Parameters: ε={config['epsilon']}, α={config['alpha']}, γ={config['gamma']}, step_cost={config['step_cost']}, stochasticity={config['stochasticity']}")
        print(f"  Final Avg Reward: {result['final_avg_reward']:.2f}")
        print(f"  Data: {result['data_file']}")
        print(f"  Visualization: {result['viz_file']}")
        print(f"  History: {result['history_file']}")

    print(f"\nAll {graph_type} experiments saved to: {graph_dir}")
    return results

def display_policy(agent, adj, terminal_rewards):
    """Display the policy and Q-values in a readable format"""
    policy = agent.get_policy()
    q_values = agent.q_values

    print("\nLearned policy:\n")
    print("{:<5} {:<15}   {}\n{}".format(
        "State", "Q-values", "Best action",
        "-" * 50))

    for s in sorted(adj.keys()):
        if s in terminal_rewards:
            q_str = "Terminal"
        else:
            available_actions = agent.env.get_available_actions(s)
            q_str = " ".join([f"{q_values[s][a]:.2f}" for a in available_actions])
        print(f"{s:<5} {q_str:<15}   {policy[s]}")

def main():
    parser = argparse.ArgumentParser(description='Room Exploration using Q-learning')
    parser.add_argument('-e', '--epsilon', type=float, default=0.1,
                        help='Exploration rate (default: 0.1)')
    parser.add_argument('-a', '--alpha', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('-g', '--gamma', type=float, default=0.9,
                        help='Discount factor (default: 0.9)')
    parser.add_argument('-c', '--cost', type=float, default=-0.05,
                        help='Step cost (default: -0.05)')
    parser.add_argument('-s', '--stochasticity', type=int, default=1, choices=[0, 1, 2],
                        help='Stochasticity level: 0=deterministic, 1=moderate (4/6,1/6,1/6), 2=high (2/6,2/6,2/6) (default: 1)')
    parser.add_argument('-n', '--episodes', type=int, default=1000,
                        help='Number of episodes (default: 1000)')
    parser.add_argument('--graph-type', type=str, default='custom_rooms', choices=AVAILABLE_GRAPHS,
                        help=f'Graph type to use (default: custom_rooms). Choices: {AVAILABLE_GRAPHS}')
    parser.add_argument('--run-experiments', action='store_true',
                        help='Run multiple Q-learning experiments with different parameters')
    args = parser.parse_args()

    if args.run_experiments:
        print("Running multiple Q-learning experiments with different parameters...")
        run_experiments(args.graph_type)
        return

    # Create environment and agent
    adj, terminal_rewards, room_coords = get_graph(args.graph_type)
    env = RoomEnvironment(adj, terminal_rewards, step_cost=args.cost, stochasticity=args.stochasticity)
    agent = QLearningAgent(env, adj, epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma)

    # Train the agent
    print(f"\nTraining Q-learning agent for {args.episodes} episodes...")
    print(f"Parameters: ε={args.epsilon}, α={args.alpha}, γ={args.gamma}, step cost={args.cost}, stochasticity={args.stochasticity}")
    rewards = agent.train(num_episodes=args.episodes)

    # Display the learned policy
    display_policy(agent, adj, terminal_rewards)

    # Save data for visualization
    data_file = agent.save_visualization_data('q_learning_data')

    # Create static visualization
    create_static_visualization(data_file, 'visualizations', room_coords)

if __name__ == "__main__":
    main()