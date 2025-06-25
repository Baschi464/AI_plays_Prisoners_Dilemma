import pickle
from itertools import combinations
from game import game, tournament_viz

## AI agents, trained with Q-learning, play the Prisoner's Dilemma game

# Load trained agents
with open("trained_agents.pkl", "rb") as f:
    agents = pickle.load(f)
agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7, agent8, agent9 = agents

# Run Tournament
TOURNAMENT_ROUNDS = 10   # each agent plays n-1 matches per round (n = number of agents)
NUM_AGENTS = len(agents)
results = {agent.get_name(): 0 for agent in agents}

for i in range(TOURNAMENT_ROUNDS):
    print(f"Round {i + 1} of {TOURNAMENT_ROUNDS}")

    for i, j in combinations(range(NUM_AGENTS), 2):
        agent_i = agents[i]
        agent_j = agents[j]
        print(f"Playing {agent_i.get_name()} vs {agent_j.get_name()}")
        score_i, score_j, actions, rewards = game(agent_i, agent_j)
        results[agent_i.get_name()] += score_i
        results[agent_j.get_name()] += score_j
        if score_i > score_j:
            print(f"{agent_i.get_name()} WINS THE MATCH!")
        elif score_j > score_i:
            print(f"{agent_j.get_name()} WINS THE MATCH!")
        else:
            print("It's a TIE!")

tournament_viz(list(results.keys()), list(results.values()))



