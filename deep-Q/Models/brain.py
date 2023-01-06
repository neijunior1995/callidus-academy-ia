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

from Utils.utils import *


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
    def __init__(self, input_size, nb_action,hidden_layer):
        self.hidden_layer = hidden_layer
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, nb_action)
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
        self.memory = push(event,self.memory, self.capacity)
    
    """
    Metodo que escolhe dados aleatorios da memoria
    """
    
    def sample(self, batch_size):
        return sample(self.memory,batch_size)
    
"""
Classe que implementa a metodologia de deep-Qlearning
"""
class Dqn(object):
    """
    Metodo de inicializacao da DQN
    """
    def __init__(self, input_size, nb_action,
                 gamma, hidden_layer = 45,
                 capacity = 100000, batch_size = 100):
        self.gamma = gamma
        self.batch_size = batch_size
        self.hidden_layer = hidden_layer
        self.capacity = capacity
        self.model = Network(input_size, nb_action, self.hidden_layer)
        self.memory = ReplayMemory(self.capacity)
        self.optimizer = optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.loss = nn.SmoothL1Loss()
        self.last_action = 0
        self.last_reward = 0
    """
    Metodo que seleciona a acao que sera tomada a partir da probilidade obtida
    por meio da DQN.
    """
    def select_action(self, state):
        return select_action(state, self.model)
    """
    Metodo utilizado para implementar o aprendizado da rede
    """
    def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        learn(self, batch_states, batch_actions, batch_rewards, batch_next_states)
    """
    Metodo utilizado para realizar o treinamento e atualizacao dos pesos da rede
    """
    def update(self, new_state, new_reward):
        return update(self, new_state, new_reward)
    """
    Metodo que salva a rede treinada
    """
    def save(self):
        save(self)
    """
    Metodo que carrega uma rede treinada
    """
    def load(self):
        load(self)