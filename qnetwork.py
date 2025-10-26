import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers=[64,64], learning_rate=0.001):
        super(QNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_layers[0])) 

        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        self.layers.append(nn.Linear(hidden_layers[-1], action_size))
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = state
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))

        x = self.layers[-1](x)

        return x