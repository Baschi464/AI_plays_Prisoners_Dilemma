import pickle
from itertools import combinations
from game import game, tournament_viz

## AI agents, trained with Q-learning, play the Prisoner's Dilemma game

def tournament(tournament_rounds=10, visualize_matches=False):
    # Load trained agents
    with open("trained_agents.pkl", "rb") as f:
        agents = pickle.load(f)

    # Run Tournament
    NUM_AGENTS = len(agents)
    results = {agent.get_name(): 0 for agent in agents}

    for i in range(tournament_rounds):
        print(f"Round {i + 1} of {tournament_rounds}")

        for i, j in combinations(range(NUM_AGENTS), 2):
            agent_i = agents[i]
            agent_j = agents[j]
            print(f"Playing {agent_i.get_name()} vs {agent_j.get_name()}")
            score_i, score_j, actions, rewards = game(agent_i, agent_j, visualize=visualize_matches)
            results[agent_i.get_name()] += score_i
            results[agent_j.get_name()] += score_j
            if score_i > score_j:
                print(f"{agent_i.get_name()} WINS THE MATCH!")
            elif score_j > score_i:
                print(f"{agent_j.get_name()} WINS THE MATCH!")
            else:
                print("It's a TIE!")
    return results

if __name__ == "__main__":
    results = tournament(tournament_rounds=10, visualize_matches=False)
    tournament_viz(list(results.keys()), list(results.values()))



