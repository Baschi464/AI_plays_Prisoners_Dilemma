import random
import pickle
from utils import encode_state, decode_state

class HardcodedAgent:
    def __init__(self, name="random", strategy="random"):
        self.name = name
        self.hardcoded_Q = None  # Not used, but kept for interface consistency
        self.strategy = strategy
        self.hardcoded_policy = self._build_policy()

    def get_name(self):
        return self.name

    def act(self, history, round):
        """
        Return action based on hardcoded policy.
        history: list of (a1, a2) tuples from previous rounds.
        """

        if self.strategy == "tit_for_tat" and round == 0:
            return 1

        state = encode_state(history)
        return self.hardcoded_policy[state]

    def _build_policy(self):
        """
        Builds a lookup table of size 1024 implementing Selected Strategy (str).
        Tit for Tat: cooperate (0) on first round, then copy opponent's last move.
        """
        policy = {}
        for idx in range(1024):
            history = decode_state(idx)

            # ---------- TIT FOR TAT --------------
            if self.strategy == "tit_for_tat":
                last_opponent_move = history[-1][1]
                policy[idx] = last_opponent_move  # Mirror opponent

            # ---------- ALWAYS DEFECT ------------
            elif self.strategy == "always_defect":
                policy[idx] = 0
            
            # ---------- ALWAYS COOPERATE ------------
            elif self.strategy == "always_cooperate":
                policy[idx] = 1

            # ---------- RANDOM --------------
            else:
                policy[idx] = random.choice([0, 1])

        return policy


# ------------- ADD HARDCODED AGENTS TO "trained_agents.pkl" ---------------------
if __name__ == "__main__":
    # Load trained agents
    with open("trained_agents.pkl", "rb") as f:
        agents = pickle.load(f)

    # Add hardcoded agents
    agents.append(HardcodedAgent(name="tit_for_tat", strategy="tit_for_tat"))
    agents.append(HardcodedAgent(name="always_defect", strategy="always_defect"))
    agents.append(HardcodedAgent(name="always_cooperate", strategy="always_cooperate"))  

    # Save updated agent list
    with open("trained_agents.pkl", "wb") as f:
        pickle.dump(agents, f)

    print("Hardcoded agents added.")