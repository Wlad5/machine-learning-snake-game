import numpy as np
import random as rd

class LinearQLearningAgent:
    def __init__(
            self,
            action_size=4,
            feature_size=12,
            alfa=0.1,
            gamma=0.9,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            ):
        self.action_size = action_size
        self.feature_size = feature_size
        self.alfa = alfa
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.weights = np.zeros(self.action_size * self.feature_size, dtype=np.float32)

    def choose_action(self, state):
        if rd.random() < self.epsilon:
            return rd.randrange(self.action_size)
        else:
            state_array = np.array(state, dtype=np.float32)
            weights_matrix = self.weights.reshape(self.action_size, self.feature_size)
            q_values = weights_matrix @ state_array
            return int(np.argmax(q_values))

    def learn(self, state, action, reward, next_state, done):
        state_array = np.array(state, dtype=np.float32)
        next_state_array = np.array(next_state, dtype=np.float32)
        
        weights_matrix = self.weights.reshape(self.action_size, self.feature_size)
        
        current_q = weights_matrix[action] @ state_array
        
        if done:
            target = reward
        else:
            next_q_values = weights_matrix @ next_state_array
            max_next_q = np.max(next_q_values)
            target = reward + self.gamma * max_next_q
        
        td_error = target - current_q
        
        weights_matrix[action] += self.alfa * td_error * state_array
        
        self.weights = weights_matrix.reshape(-1)

    def decay_epsilon(self):
        """Decay the exploration rate after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min