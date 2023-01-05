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
    def __init__(self, input_size, nb_action, hidden_layer = 45):
        super(Network, self).__init__()
        self.input_size = input_size # número de entradas
        self.nb_action = nb_action   # número de ações
        self.hidden_layer = hidden_layer # números de neurônios na camada escondida
        self.fc1 = nn.Linear(self.input_size, 45) # Primeira camada com neurônios lineares
        self.fc2 = nn.Linear(45, self.nb_action) # Segunda camada com neurônios lineares
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
        self.capacity = capacity # capacidade de armazenamento de estados
        self.memory = []         # memoria armazenada
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
        print(self.memory[0])
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
"""
Classe que implementa a metodologia de deep-Qlearning
"""

class Dqn(object):
    """
    Metodo de inicializacao da DQN
    """
    def __init__(self, input_size, nb_action,
                 gamma, capacity = 100000, bacth_size = 100 ):
        self.gamma = gamma
        self.capacity = capacity
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity = self.capacity)
        self.batch_size = bacth_size
        self.optimizer = optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    """
    Metodo que seleciona a acao que sera tomada a partir da probilidade obtida
    por meio da DQN.
    """
    def select_action(self, state):
        with torch.no_grad():
            teste = self.model(state)
            probabilities = F.softmax(teste, dim=-1)*100
            action = torch.multinomial(probabilities[0,0],num_samples=1)
            return action.item()
    """
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state))*100,dim=2)
        
        print("Tamanho: ",len(probs[0,0,0]))
        print("Prob: ",probs[0,0,0])
        action = probs.multinomial(num_samples=1)
        return probs #action.item()
    """
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

        new_state = np.array([new_state[0]], np.float32)
        new_state = torch.Tensor([new_state]).float().unsqueeze(0)
        self.memory.push((self.last_state[0], torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state[0,0,0]))
        new_action = self.select_action(new_state)
        if len(self.memory.memory) > self.batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(self.batch_size)
            self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
        self.last_state = new_state
        self.last_action = new_action
        self.last_reward = new_reward
        return new_action
    """
    Metodo que salva a rede treinada
    """
    def save(self, name = 'last_brain.pth'):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, name)
    """
    Metodo que carrega uma rede treinada
    """
    def load(self,name = 'last_brain.pth' ):
        if os.path.isfile(name):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")