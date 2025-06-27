from hardcoded_agent_class import HardcodedAgent
from game import game

agentA = HardcodedAgent(name="tit_for_tat", strategy="tit_for_tat")
agentB = HardcodedAgent(name="always_cooperate", strategy="always_cooperate")

scoreA, scoreB, actions, rewards = game(agentA, agentB, visualize=True)

if scoreA > scoreB:
    print(f"{agentA.get_name()} WINS THE MATCH!")
elif scoreB> scoreA:
    print(f"{agentB.get_name()} WINS THE MATCH!")
else:
    print("It's a TIE!")