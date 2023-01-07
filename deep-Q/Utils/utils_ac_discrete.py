import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from Models.Pongs.recompensas import discount_rewards


def push( event, memory, capacity):
    """
    Método utilizado para atualizar e adicionar novos eventos a memoria
    """
    memory.append(event)
    if len(memory) > capacity:
        del memory[0]
    return memory
def sample(memory, batch_size):
    """
    Método utilizado para realizar a escolha aleatória de dados do dataset
    e assim realizar um novo treinamento
    """
    samples = zip(*random.sample(memory, batch_size))
    return map(lambda x: Variable(torch.cat(x, 0)), samples)

def pull(memory):
    states, actions, rewards = [],[],[]
    for data in memory:
        state, action, reward = data
        states.append(state)
        actions.append(action)
        rewards.append(reward)
    return states, actions, rewards

def select_action(state, model):
    """
    Método que implementa a seleção da ação da saida do modelo
    """
    probs = F.softmax(model(Variable(state))*100)
    action = probs.multinomial(len(probs))
    return action.data[0,0]


def remember(self,state,action,reward):
    """
    Método utilizado para armazenar em forma de tensores os dados utilizados
    """
    state = torch.Tensor(state).float().unsqueeze(0)
    self.memory.push((state, torch.LongTensor([int(action)]), torch.Tensor([reward])))

def discount_rewards(self):
    """Peso utilizado para aprender as ações apresentadas"""
    _, _, rewards = self.replay()
    flag = []
    for reward in rewards:
        flag.append(reward.item())
    return discount_rewards(self,flag)


def learn(self):
    "Implementação do treinamento da rede de aprendizagem profunda utilizada"
    states, actions, rewards = self.memory.pull()
    dis_reward = discount_rewards()
    td_loss = self.loss(batch_outputs, batch_targets, dis_reward)
    self.optimizer.zero_grad()
    td_loss.backward()
    self.optimizer.step()

def save(self,name = 'last_brain.pth'):
    torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, name)
def load(self, name = 'last_brain.pth'):
    if os.path.isfile(name):
        print("=> Carregando IA... ")
        checkpoint = torch.load(name)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("carregado !")
    else:
        print("Arquivo não encontrado...")

class ReplayMemory(object):
    """
    Objeto utilizado para armazenar os dados obtidos durante a realização do treinamento do sistema
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    def push(self, event):
        """
        Metodo que implementa o salvamento de dados na memoria
        """
        self.memory = push(event,self.memory, self.capacity)
    def pull(self):
        """
        Metodo utilizado para acessar os dados da memoria
        """
        return pull(self.memory)
    def sample(self):
        """
        Metodos utilizados para carregar uma quantidade desejada de dados aletorios
        """
        return sample(self.memory)
    def erase(self):
        """
        Metodo utilizado para apagar a memoria
        """
        self.memory = []

class CrossEntropyLossW(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLossW, self).__init__()

    def forward(self, inputs, targets, weight, smooth=1):

        inputs = F.cross_entropy(inputs,targets)

        loss = (inputs*weight).sum()

        return loss