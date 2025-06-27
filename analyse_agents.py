import pickle
import torch
from collections import defaultdict
from agent_class import Agent
from hardcoded_agent_class import HardcodedAgent
from main import tournament
from matplotlib import pyplot as plt
from utils import decode_state

# Load trained agents
with open("trained_agents.pkl", "rb") as f:
    agents = pickle.load(f)

def compute_agent_characteristics(agent_policy, opponent_policy):
    """
    Compute 4 characteristics of an agent given its decoded Q-policy and an opponent's.

    Inputs:
    - agent_policy: list of (history, action) from decode_q_policy(agent)
    - opponent_policy: list of (history, action) from decode_q_policy(opponent)

    Returns:
    - dict with normalized scores for:
        - niceness: avoids defecting first
        - retaliatory: punishes after opponent defects
        - forgiving: returns to cooperation after opponent does
        - clarity: acts consistently across similar histories
    """

    assert len(agent_policy) == 1024 and len(opponent_policy) == 1024

    retaliate_count = 0
    retaliate_total = 0

    forgive_count = 0
    forgive_total = 0

    first_defect_count = 0
    total_defection_opportunities = 0

    response_map = defaultdict(list)

    for idx in range(1024):
        history, agent_action = agent_policy[idx]
        last_round = history[-1]  # (agent_prev, opp_prev)

        # 1. Niceness: defecting first against always-cooperative opponent
        if all(opp == 1 for _, opp in history):
            total_defection_opportunities += 1
            if agent_action == 0:
                first_defect_count += 1

        # 2. Retaliatory: defecting after opponent defected last round
        if last_round[1] == 0:
            retaliate_total += 1
            if agent_action == 0:
                retaliate_count += 1

        # 3. Forgiving: cooperating after opponent defected and then cooperated
        if len(history) >= 2:
            prev_round = history[-2]
            if prev_round[1] == 0 and last_round[1] == 1:
                forgive_total += 1
                if agent_action == 1:
                    forgive_count += 1

        # 4. Clarity: consistent responses to last opponent move
        last_opp = last_round[1]
        response_map[last_opp].append(agent_action)

    # Clarity score: average consistency for each group
    clarity_score = 0
    for responses in response_map.values():
        most_common = max(set(responses), key=responses.count)
        consistency = responses.count(most_common) / len(responses)
        clarity_score += consistency
    clarity_score /= len(response_map)

    return {
        "niceness": 1 - (first_defect_count / total_defection_opportunities) if total_defection_opportunities > 0 else 1.0,
        "retaliatory": retaliate_count / retaliate_total if retaliate_total > 0 else 0.0,
        "forgiving": forgive_count / forgive_total if forgive_total > 0 else 0.0,
        "clarity": clarity_score
    }

def decode_q_policy(agent):
    """
    Given an Agent with a Q-table, return a list of (history, action) tuples,
    where action is the greedy action under Q(s).
    """
    decoded = []
    for idx in range(1024):
        history = decode_state(idx)
        action = torch.argmax(agent.Q[idx]).item()
        decoded.append((history, action))
    return decoded


if __name__ == "__main__":
    # Load agents
    with open("trained_agents.pkl", "rb") as f:
        agents = pickle.load(f)

    # Decode all policies once
    all_decoded = {}

    for agent in agents:
        if isinstance(agent, HardcodedAgent):
            # Hardcoded agents store their policy directly as a lookup table
            decoded = [(decode_state(idx), agent.hardcoded_policy[idx]) for idx in range(1024)]
        else:
            # Learned agents use argmax over Q-table rows
            decoded = decode_q_policy(agent)  # your function for standard agents
        all_decoded[agent.get_name()] = decoded

    # Initialize a dictionary to hold traits for each agent
    traits_by_agent = {}

    # Compute characteristics per agent vs others
    for agent in agents:
        name = agent.get_name()
        policy = all_decoded[name]

        trait_sums = {"niceness": 0.0, "retaliatory": 0.0, "forgiving": 0.0, "clarity": 0.0}
        count = 0

        for other_agent in agents:
            if other_agent.get_name() == name:
                continue
            other_policy = all_decoded[other_agent.get_name()]
            traits = compute_agent_characteristics(policy, other_policy)
            for key in trait_sums:
                trait_sums[key] += traits[key]
            count += 1

        traits_by_agent[name] = {key: trait_sums[key] / count for key in trait_sums}

    # Run tournament and collect scores
    results = tournament(tournament_rounds=10, visualize_matches=False)

    # Create 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()  # flatten to 1D list for easy indexing

    traits_list = ["niceness", "retaliatory", "forgiving", "clarity"]

    for i, trait in enumerate(traits_list):
        ax = axs[i]
        x = [traits_by_agent[agent.get_name()][trait] for agent in agents]
        y = [results[agent.get_name()] for agent in agents]
        names = [agent.get_name() for agent in agents]

        ax.scatter(x, y)
        for j, name in enumerate(names):
            ax.text(x[j], y[j], name, fontsize=8, ha='right')

        ax.set_xlabel(trait.capitalize())
        ax.set_ylabel("Tournament Score")
        ax.set_title(f"{trait.capitalize()} vs Tournament Score")
        ax.grid(True)

    # Create histogram for average agent traits
    fig_hist, ax_hist = plt.subplots(figsize=(12, 6))

    # Prepare data for histogram
    x = range(len(agents))  # Agent indices
    averages = [
        sum(traits_by_agent[agent.get_name()][trait] for trait in ["niceness", "retaliatory", "forgiving", "clarity"]) / 4
        for agent in agents
    ]

    # Create bars for average scores
    ax_hist.bar(x, averages, color='skyblue')

    # Add labels and title
    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels([agent.get_name() for agent in agents], rotation=45, ha='right')
    ax_hist.set_ylabel("Average Trait Score")
    ax_hist.set_title("Average Trait Scores per Agent")
    ax_hist.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()




