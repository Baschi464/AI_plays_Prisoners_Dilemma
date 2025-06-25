import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import numpy as np

def game_viz(agent1_name, agent2_name, actions, rewards, interval=1000):
    """
    Animated version of game_viz using matplotlib.animation.FuncAnimation

    Parameters:
    - agent1_name, agent2_name: str
    - actions: list of (a1, a2) pairs
    - rewards: list of (r1, r2) pairs
    - interval: time between frames in ms
    """

    rounds = len(actions)
    fig, ax = plt.subplots(figsize=(1.5 * rounds, 3))
    ax.set_xlim(-0.5, rounds - 0.5)
    ax.set_ylim(-1.5, 0.5)
    ax.axis('off')

    circles = []
    for t in range(rounds):
        circle1 = plt.Circle((t, 0), 0.4, color='white', ec='black')
        circle2 = plt.Circle((t, -1), 0.4, color='white', ec='black')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        circles.append((circle1, circle2))

    # Static agent labels
    ax.text(-1, 0, agent1_name, fontsize=12, fontweight='bold', verticalalignment='center')
    ax.text(-1, -1, agent2_name, fontsize=12, fontweight='bold', verticalalignment='center')

    # Total rewards (static)
    total_r1 = sum(r[0] for r in rewards)
    total_r2 = sum(r[1] for r in rewards)
    ax.text(rounds + 0.5, 0, f"Total: {total_r1}", fontsize=12, verticalalignment='center')
    ax.text(rounds + 0.5, -1, f"Total: {total_r2}", fontsize=12, verticalalignment='center')

    # Legend
    green_patch = mpatches.Patch(color='green', label='Cooperate')
    red_patch = mpatches.Patch(color='red', label='Defect')
    ax.legend(handles=[green_patch, red_patch], loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)

    plt.title(f"{agent1_name} vs {agent2_name}", fontsize=14)

    def update(frame):
        a1, a2 = actions[frame]
        color1 = 'green' if a1 == 0 else 'red'
        color2 = 'green' if a2 == 0 else 'red'
        circles[frame][0].set_color(color1)
        circles[frame][1].set_color(color2)
        return circles[frame]

    anim = FuncAnimation(fig, update, frames=rounds, interval=interval, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()  


def game(agentA, agentB, visualize=False):

    # Payoff matrix: (self_action, opponent_action) → reward
    REWARD_TABLE = {
        (0, 0): 3,      
        (0, 1): 0,
        (1, 0): 5,
        (1, 1): 1
    }
    ROUNDS=20
    MEMORY_SIZE = 5 # number of rounds that an agent remembers
    scoreA = 0    # total reward for agent A
    scoreB = 0    # total reward for agent B
    actions = []   # list of tuples (actionA, actionB)
    rewards = []  # list of tuples (rewardA, rewardB)

    # Start with padded memory: [('C', 'C')] = (0, 0)
    history_A = [(0, 0)] * MEMORY_SIZE
    history_B = [(0, 0)] * MEMORY_SIZE

    for i in range(ROUNDS):
        
        # Actions: 0=defect; 1=cooperate
        actA = agentA.act(history_A, epsilon=0)   # epsilon=0, this is no training, its a tournament game!
        actB = agentB.act(history_B, epsilon=0)   # so it means no random exploration, just follow the Q-table

        # Rewards
        rewardA = REWARD_TABLE[(actA, actB)]
        rewardB = REWARD_TABLE[(actB, actA)]

        # Update memory
        history_A = history_A[1:] + [(actA, actB)]
        history_B = history_B[1:] + [(actA, actB)]

        # Update scores
        scoreA += rewardA
        scoreB += rewardB

        # Store actions and rewards
        actions.append((actA, actB))
        rewards.append((rewardA, rewardB))
        
        print("round number "+ str(i+1) + ":\n", rewardA, rewardB)
    
    if visualize:
        game_viz(agentA.get_name(), agentB.get_name(), actions, rewards)

    return (scoreA, scoreB, actions, rewards)  


def tournament_viz(names, scores):
    """
    Visualize tournament standings.

    Parameters:
    - names: list of agent names
    - scores: list of total reward scores (same length as names)
    """

    # Sort scores descending with associated names
    sorted_data = sorted(zip(scores, names), reverse=True)
    sorted_scores, sorted_names = zip(*sorted_data)
    ranks = list(range(1, len(sorted_scores) + 1))

    # Set up the plot
    fig, ax = plt.subplots(figsize=(6, len(names) * 0.6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    ax.barh(ranks, sorted_scores, color=colors, edgecolor='black')

    # Aesthetics
    ax.set_yticks(ranks)
    ax.set_yticklabels(sorted_names, fontweight='bold')
    ax.invert_yaxis()  # Rank 1 at top
    ax.set_xlabel("Total Reward", fontsize=12)
    ax.set_title("Tournament Standings", fontsize=14, pad=12)

    # Annotate scores next to bars
    for i, score in enumerate(sorted_scores):
        ax.text(score + 1, ranks[i], str(score), va='center', ha='left', fontsize=10)

    plt.tight_layout()
    plt.show()

