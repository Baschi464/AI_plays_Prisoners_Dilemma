import pickle

## AI agents, trained with Q-learning, play the Prisoner's Dilemma game

# Load trained agents
with open("trained_agents.pkl", "rb") as f:
    agents = pickle.load(f)

