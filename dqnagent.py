import numpy as np
from qnetwork import QNetwork
from replaybuffer import ReplayBuffer
import torch
import torch.nn as nn


class DQNAgent:
    def __init__(self, state_size, action_size, 
                hidden_layers=[64, 64],
                gamma = 0.99,
                learning_rate = 0.001,
                buffer_capacity=10000,
                batch_size=32,
                target_update_freq=100,
                min_buffer_size=1000
                ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_size = min_buffer_size

        self.epsilon = 1

        self.q_network = QNetwork(state_size, action_size, hidden_layers, learning_rate)
        self.target_network = QNetwork(state_size, action_size, hidden_layers, learning_rate)

        self.update_target_network()

        self.memory = ReplayBuffer(buffer_capacity)

        self.steps = 0

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def state_to_one_hot(self, state):
        one_hot = np.zeros(self.state_size)

        one_hot[state] = 1
        return one_hot
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        state_one_hot = self.state_to_one_hot(state)
        state_tensor = torch.FloatTensor(state_one_hot).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.min_buffer_size:
            return None
    
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states_one_hot = np.array([self.state_to_one_hot(s) for s in states])
        next_states_one_hot = np.array([self.state_to_one_hot(s) for s in next_states])
        
        states = torch.FloatTensor(states_one_hot)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states_one_hot)
        dones = torch.FloatTensor(dones)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()
    
        return loss.item()

    def decay_epsilon(self):

        self.epsilon = max(0.001, self.epsilon * 0.995) 
    