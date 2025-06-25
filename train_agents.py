from agent_class import Agent
from itertools import combinations
import pickle

# Constants
NUM_AGENTS = 10
NUM_EPOCHS = 1000
NUM_ROUNDS = 20
MEMORY_SIZE = 5

# Payoff matrix: (self_action, opponent_action) → reward
REWARD_TABLE = {
    (0, 0): 3,      
    (0, 1): 0,
    (1, 0): 5,
    (1, 1): 1
}

# Epsilon parameters
EPSILON_START = 0.2
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Initialize agents  (alpha=learning_rate; gamma=discount_factor)
agent0 = Agent(name="Pitson", alpha=0.1, gamma=0.95)
agent1 = Agent(name="Futson", alpha=0.1, gamma=0.95)
agent2 = Agent(name="Nicson", alpha=0.1, gamma=0.95)
agent3 = Agent(name="Gecson", alpha=0.1, gamma=0.95)
agent4 = Agent(name="Alecson", alpha=0.1, gamma=0.95)
agent5 = Agent(name="Mikeson", alpha=0.1, gamma=0.95)
agent6 = Agent(name="Gionson", alpha=0.1, gamma=0.95)
agent7 = Agent(name="Laurson", alpha=0.1, gamma=0.95)
agent8 = Agent(name="Carlson", alpha=0.1, gamma=0.95)
agent9 = Agent(name="Tomson", alpha=0.1, gamma=0.95)
agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8, agent9]
epsilon = EPSILON_START

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
            agent_i.update(history_i, a_i, r_i, new_history_i)
            agent_j.update(history_j, a_j, r_j, new_history_j)

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

    # print training progress
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, epsilon = {epsilon:.4f}")
        #print("------------------------------------------------------------------------")

# Store agents
with open("trained_agents.pkl", "wb") as f:
    pickle.dump(agents, f)
