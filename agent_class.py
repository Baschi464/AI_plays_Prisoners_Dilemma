import torch
import random
from utils import encode_state, decode_state

class Agent:
    NUM_STATES = 1024   # 4^5
    NUM_ACTIONS = 2     # 0 = Cooperate, 1 = Defect

    def __init__(self, name="agent", gamma=0.95):
        self.name = name
        self.Q = torch.zeros(self.NUM_STATES, self.NUM_ACTIONS)
        self.gamma = gamma
        self.prev_Q = self.Q.clone()  # For tracking changes in Q-values during training

    def get_name(self):
        return self.name

    def act(self, history, epsilon=0.1):
        """Return action (0 or 1) given history and exploration probability"""
        state = encode_state(history)
        if random.random() < epsilon:
            return random.randint(0, 1)
        return torch.argmax(self.Q[state]).item()

    def update(self, history, action, reward, next_history, alpha=0.1):
        """Update Q-table based on transition"""
        s = encode_state(history)
        s_next = encode_state(next_history)
        td_target = reward + self.gamma * torch.max(self.Q[s_next])
        self.Q[s, action] += alpha * (td_target - self.Q[s, action])