from agent_class import Agent
from itertools import combinations
import pickle
import torch

# Constants
NUM_AGENTS = 10
NUM_EPOCHS = 10000
NUM_ROUNDS = 20
MEMORY_SIZE = 5

# Payoff matrix: (self_action, opponent_action) → reward
REWARD_TABLE = {
    (1, 1): 3,      # both cooperate   
    (1, 0): 0,      # agent A cooperates, agent B defects
    (0, 1): 5,      # agent A defects, agent B cooperates
    (0, 0): 1       # both defect
}

# Epsilon parameters (exploration probability)
EPSILON_START = 1.0    # epsilon >= 1 means agents acts fully random
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
# Alpha parameters (learning rate)
ALPHA_START = 0.3      # large enough for early learning
ALPHA_MIN = 0.02       # small enough for stable late-stage convergence
ALPHA_DECAY = 0.9995   # slower than epsilon decay

# Initialize agents  (gamma = discount_factor on future rewards)
agent0 = Agent(name="Pitson", gamma=0.95)
agent1 = Agent(name="Futson", gamma=0.95)
agent2 = Agent(name="Nicson", gamma=0.95)
agent3 = Agent(name="Gecson", gamma=0.95)
agent4 = Agent(name="Alecson", gamma=0.95)
agent5 = Agent(name="Mikeson", gamma=0.95)
agent6 = Agent(name="Gionson", gamma=0.95)
agent7 = Agent(name="Laurson", gamma=0.95)
agent8 = Agent(name="Carlson", gamma=0.95)
agent9 = Agent(name="Tomson", gamma=0.95)
agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8, agent9]
epsilon = EPSILON_START
alpha = ALPHA_START

# ----- helper functions for learning tracking ---------

def mean_q_change(Q_old: torch.Tensor, Q_new: torch.Tensor) -> float:
    """
    Computes the mean absolute change in Q-values between two Q-tables.

    Parameters:
    - Q_old: torch.Tensor of shape (num_states, num_actions)
    - Q_new: torch.Tensor of same shape

    Returns:
    - Mean absolute change across all Q-values
    """
    if Q_old.shape != Q_new.shape:
        raise ValueError("Q-tables must have the same shape")

    delta = torch.abs(Q_new - Q_old)
    return delta.mean().item()

def policy_vector(agent):
    return [torch.argmax(agent.Q[idx]).item() for idx in range(1024)]

def hamming_distance(policy1, policy2):
    return sum(a != b for a, b in zip(policy1, policy2)) / len(policy1)

# -------- TRAINING -------------

for epoch in range(NUM_EPOCHS):
    # For every pair of agents
    for i, j in combinations(range(NUM_AGENTS), 2):
        agent_i = agents[i]
        agent_j = agents[j]
        score_i = 0
        score_j = 0
        #print(f"Playing {agent_i.get_name()} vs {agent_j.get_name()}")

        # Start with padded memory: [('C', 'C')] = (0, 0)
        history_i = [(0, 0)] * MEMORY_SIZE
        history_j = [(0, 0)] * MEMORY_SIZE

        # ------ GAME --------
        for _ in range(NUM_ROUNDS):
            a_i = agent_i.act(history_i, epsilon)
            a_j = agent_j.act(history_j, epsilon)

            r_i = REWARD_TABLE[(a_i, a_j)]
            r_j = REWARD_TABLE[(a_j, a_i)]

            # Update memory
            new_history_i = history_i[1:] + [(a_i, a_j)]
            new_history_j = history_j[1:] + [(a_j, a_i)]

            # Q-learning update
            agent_i.update(history_i, a_i, r_i, new_history_i, alpha=alpha)
            agent_j.update(history_j, a_j, r_j, new_history_j, alpha=alpha)

            # Advance memory
            history_i = new_history_i
            history_j = new_history_j

            # Update scores
            score_i += r_i      
            score_j += r_j

        # if score_i > score_j:
        #     print(f"{agent_i.get_name()} WINS THE MATCH!")
        # elif score_j > score_i:
        #     print(f"{agent_j.get_name()} WINS THE MATCH!")
        # else:
        #     print("It's a TIE!")

    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
    # Decay alpha
    alpha = max(ALPHA_MIN, alpha * ALPHA_DECAY)

    # print training progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, epsilon = {epsilon:.4f}, alpha = {alpha:.4f}")

        delta_qs = []
        for agent in agents:
            delta_q = mean_q_change(agent.prev_Q, agent.Q)
            delta_qs.append(delta_q)
        
        average_delta_q = sum(delta_qs) / NUM_AGENTS
        print(f"Average Q-value change: {average_delta_q:.6f}") # If it drops and stabilizes near zero, learning has plateaued.

        # Entropy tracking
        total_entropy = 0
        for agent in agents:
            entropy = 0
            for q_row in agent.Q:
                probs = torch.zeros(2)
                best_action = torch.argmax(q_row)
                probs[best_action] = 1.0
                entropy += -torch.sum(probs * torch.log2(probs + 1e-8))
            avg_entropy = entropy / 1024
            total_entropy += avg_entropy
        mean_entropy = total_entropy / NUM_AGENTS
        print(f"Average policy entropy: {mean_entropy:.6f}") # Low entropy indicates convergence to a stable policy.

        vectors = [policy_vector(agent) for agent in agents]
        divergences = []

        for i in range(len(agents)):
            for j in range(i+1, len(agents)):                                # 0 = agents act the same
                d = hamming_distance(vectors[i], vectors[j])                 # 0.5 = agents act diversly
                divergences.append(d)                                        # 1 = agents act in opposite ways

        mean_divergence = sum(divergences) / len(divergences)
        print(f"Average pairwise policy divergence: {mean_divergence:.4f}") 

        # Update prev_Q 
        for agent in agents:
            agent.prev_Q = agent.Q.clone()

        print("------------------------------------------------------------------------")

# Store agents
with open("trained_agents.pkl", "wb") as f:
    pickle.dump(agents, f)
