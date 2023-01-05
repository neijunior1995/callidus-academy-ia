"""
Inteligência artificial para deep-Qlearning que pode ser aplicado
a problemas de estados discretos
"""
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


"""
Network classe utilizada para armazenar a rede neural utilizada para o
desenvolvimento da rede de aprendizagem por reforço

*input_size*: uma variável inteira que deve armazenar a quantidade de estados
que são passados como entradas para rede

*nb_action*: uma variável que armazena o número de ações que podem ser tomadas
pela rede de aprendizagem por reforço.

*hidden_layer*: é o número de neurônios na camada interna da rede neural
"""
class Network(nn.Module):
    """
    Função de inicializacao de rede neural
    """
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_action)
    """
    forward de rede neural onde as saida sao os Q-values
    """
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

"""

Implementacao da memoria utilizada no treinamento

"""

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    """
    Metodo que implementa o salvamento de dados na memoria
    """
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    """
    Metodo que escolhe dados aleatorios da memoria
    """
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
"""
Classe que implementa a metodologia de deep-Qlearning
"""
class Dqn(object):
    """
    Metodo de inicializacao da DQN
    """
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity = 100000)
        self.optimizer = optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    """
    Metodo que seleciona a acao que sera tomada a partir da probilidade obtida
    por meio da DQN.
    """
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state))*100)
        action = probs.multinomial(len(probs))
        return action.data[0,0]
    """
    Metodo utilizado para implementar o aprendizado da rede
    """
    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
        batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]
        batch_targets = batch_rewards + self.gamma * batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()
    """
    Metodo utilizado para realizar o treinamento e atualizacao dos pesos da rede
    """
    def update(self, new_state, new_reward):
        new_state = torch.Tensor(new_state).float().unsqueeze(0)
        self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state))
        new_action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(100)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        return new_action
    """
    Metodo que salva a rede treinada
    """
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    """
    Metodo que carrega uma rede treinada
    """
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")