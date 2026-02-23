import random as rd
import numpy as np
from collections import defaultdict

class Q_learning_Agent:
    def __init__(
            self,
            action_size=4,
            learning_rate=0.1,
            gamma=0.9,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            ):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_table = defaultdict(lambda: np.zeros(self.action_size, dtype=np.float32))

    def choose_action(self, state):
        if rd.random() < self.epsilon:
            return rd.randrange(self.action_size)
        return int(np.argmax(self.q_table[state]))
    
    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]
        max_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        target_q = reward + self.gamma * max_next_q
        self.q_table[state][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)