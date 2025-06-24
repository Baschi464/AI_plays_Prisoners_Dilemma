import torch
import random

class Agent:
    NUM_STATES = 1024   # 4^5
    NUM_ACTIONS = 2     # 0 = Cooperate, 1 = Defect

    def __init__(self, alpha=0.1, gamma=0.95):
        self.Q = torch.zeros(self.NUM_STATES, self.NUM_ACTIONS)
        self.alpha = alpha
        self.gamma = gamma

    def encode_state(self, history):     # history is a list of 5 tuples: (self_action, opp_action)
        """Encode last 5 (self, opp) moves as integer index ∈ [0, 1023]"""
        idx = 0
        for (a, b) in history:
            code = a * 2 + b
            idx = (idx << 2) | code
        return idx

    def act(self, history, epsilon=0.1):
        """Return action (0 or 1) given history and exploration probability"""
        state = self.encode_state(history)
        if random.random() < epsilon:
            return random.randint(0, 1)
        return torch.argmax(self.Q[state]).item()

    def update(self, history, action, reward, next_history):
        """Update Q-table based on transition"""
        s = self.encode_state(history)
        s_next = self.encode_state(next_history)
        td_target = reward + self.gamma * torch.max(self.Q[s_next])
        self.Q[s, action] += self.alpha * (td_target - self.Q[s, action])
