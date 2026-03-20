import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import pickle
import os


class DQNNetwork(nn.Module):
    def __init__(self, state_size=12, action_size=4, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
    
    def forward(self, state):
        return self.net(state)


class DQNAgent:
    def __init__(
        self,
        state_size=12,
        action_size=4,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=10000,
        hidden_size=256,
        update_frequency=100,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.step_count = 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = deque(maxlen=memory_size)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(q_values.argmax(dim=1).cpu().numpy()[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Calculate loss
        loss = self.criterion(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_mode(self):
        self.q_network.train()

    def eval_mode(self):
        self.q_network.eval()

    def save(self, filepath):
        """Save trained model and hyperparameters to pickle file."""
        # Properly extract hidden_size from first layer
        first_layer = self.q_network.net[0]
        if isinstance(first_layer, nn.Linear):
            saved_hidden_size = first_layer.out_features
        else:
            saved_hidden_size = 256
            
        save_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_size': saved_hidden_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'update_frequency': self.update_frequency,
        }
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Model saved to {filepath} (hidden_size={saved_hidden_size})")

    @staticmethod
    def load(filepath, epsilon=0.0):
        """Load trained model from pickle file."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Always infer hidden_size from the actual network weights (authoritative source)
        hidden_size = None
        if 'q_network_state_dict' in save_data:
            state_dict = save_data['q_network_state_dict']
            # The first layer weight shape is [hidden_size, state_size]
            # So shape[0] gives us the hidden_size
            if 'net.0.weight' in state_dict:
                hidden_size = state_dict['net.0.weight'].shape[0]
                print(f"Inferred hidden_size from checkpoint: {hidden_size}")
        
        # Fallback to saved data if inference failed
        if hidden_size is None:
            hidden_size = save_data.get('hidden_size', 256)
            if 'hidden_size' not in save_data:
                print(f"Using default hidden_size: {hidden_size}")
        
        print(f"Loading model with hidden_size={hidden_size}")
        
        # Create agent with loaded hyperparameters
        agent = DQNAgent(
            state_size=save_data['state_size'],
            action_size=save_data['action_size'],
            learning_rate=save_data['learning_rate'],
            gamma=save_data['gamma'],
            epsilon=epsilon,
            epsilon_min=save_data['epsilon_min'],
            epsilon_decay=save_data['epsilon_decay'],
            batch_size=save_data['batch_size'],
            update_frequency=save_data['update_frequency'],
            hidden_size=hidden_size,
        )
        
        # Load network weights
        agent.q_network.load_state_dict(save_data['q_network_state_dict'])
        agent.target_network.load_state_dict(save_data['q_network_state_dict'])
        agent.eval_mode()
        
        print(f"Model loaded from {filepath}")
        return agent
