import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import numpy as np
from agent_class import Agent
from hardcoded_agent_class import HardcodedAgent

# -----------------------------------------------------------------

def game_viz(agent1_name, agent2_name, actions, rewards, interval=1000):
    rounds = len(actions)
    fig, ax = plt.subplots(figsize=(0.7 * rounds, 3))
    ax.set_xlim(-0.5, rounds - 0.5)
    ax.set_ylim(-1.5, 0.5)
    ax.axis('off')

    # Preallocate circle objects
    circles = []
    for t in range(rounds):
        circle1 = plt.Circle((t, 0), 0.2, color='white', ec='black')
        circle2 = plt.Circle((t, -1), 0.2, color='white', ec='black')
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        circles.append((circle1, circle2))

    # Agent names
    ax.text(-1, 0, agent1_name, fontsize=12, fontweight='bold', verticalalignment='center')
    ax.text(-1, -1, agent2_name, fontsize=12, fontweight='bold', verticalalignment='center')

    # Dynamic reward text (initialized as placeholders)
    reward_text1 = ax.text(rounds + 0.5, 0, "Total: 0", fontsize=12, verticalalignment='center')
    reward_text2 = ax.text(rounds + 0.5, -1, "Total: 0", fontsize=12, verticalalignment='center')

    # Legend
    green_patch = mpatches.Patch(color='green', label='Cooperate')
    red_patch = mpatches.Patch(color='red', label='Defect')
    ax.legend(handles=[green_patch, red_patch], loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2)

    plt.title(f"{agent1_name} vs {agent2_name}", fontsize=14)

    # Accumulated rewards
    cumulative_r1 = [0]
    cumulative_r2 = [0]
    for r1, r2 in rewards:
        cumulative_r1.append(cumulative_r1[-1] + r1)
        cumulative_r2.append(cumulative_r2[-1] + r2)

    def update(frame):
        a1, a2 = actions[frame]
        color1 = 'green' if a1 == 1 else 'red'
        color2 = 'green' if a2 == 1 else 'red'
        circles[frame][0].set_color(color1)
        circles[frame][1].set_color(color2)

        reward_text1.set_text(f"Total: {cumulative_r1[frame + 1]}")
        reward_text2.set_text(f"Total: {cumulative_r2[frame + 1]}")

        return list(circles[frame]) + [reward_text1, reward_text2]

    anim = FuncAnimation(fig, update, frames=rounds, interval=interval, blit=False, repeat=False)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------

def game(agentA, agentB, visualize=False):

    # Payoff matrix: (self_action, opponent_action) → reward
    REWARD_TABLE = {
        (1, 1): 3,     # both cooperate 
        (1, 0): 0,     # agent A cooperates, agent B defects
        (0, 1): 5,     # agent A defects, agent B cooperates
        (0, 0): 1      # both defect
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
        if isinstance(agentA, HardcodedAgent):
            actA = agentA.act(history_A, i)
        else:
            actA = agentA.act(history_A, epsilon=0) # epsilon=0, this is no training, its a tournament game! It means no exploration, just follow the Q-table

        if isinstance(agentB, HardcodedAgent):
            actB = agentB.act(history_B, i)
        else:
            actB = agentB.act(history_B, epsilon=0) # epsilon=0, this is no training, its a tournament game! It means no exploration, just follow the Q-table

        # Rewards
        rewardA = REWARD_TABLE[(actA, actB)]
        rewardB = REWARD_TABLE[(actB, actA)]

        # Update memory
        history_A = history_A[1:] + [(actA, actB)]
        history_B = history_B[1:] + [(actB, actA)]

        # Update scores
        scoreA += rewardA
        scoreB += rewardB

        # Store actions and rewards
        actions.append((actA, actB))
        rewards.append((rewardA, rewardB))
        
        #print("round number "+ str(i+1) + ":\n", rewardA, rewardB)
    
    if visualize:
        game_viz(agentA.get_name(), agentB.get_name(), actions, rewards)

    return (scoreA, scoreB, actions, rewards)  

# -----------------------------------------------------------------

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