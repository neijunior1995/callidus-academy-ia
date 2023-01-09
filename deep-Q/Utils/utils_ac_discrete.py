import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader

from Models.Pongs.recompensas import discount_rewards as dr


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
    """
    Método utilizado para adicionar dados a rede treinada
    """
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
    probs = F.softmax(model(Variable(state))[0]*100)
    action = probs.multinomial(len(probs))
    return action.data[0,0].item()


def remember(self,state,action,reward):
    """
    Método utilizado para armazenar em forma de tensores os dados utilizados
    """
    state = torch.Tensor(state).float().unsqueeze(0)
    flag = []
    for out in range(self.output_size):
        flag.append(0)
    flag[int(action)] = 1
    flag = torch.Tensor(flag).float().unsqueeze(0)
    self.memory.push((state, flag, torch.Tensor([reward])))

def discount_rewards(self):
    """Peso utilizado para aprender as ações apresentadas"""
    _, _, rewards = self.replay()
    flag = []
    for reward in rewards:
        flag.append(reward.item())
    return dr(self,flag)


def learn(self):
    "Implementação do treinamento da rede de aprendizagem profunda utilizada"
    states, actions, rewards = self.memory.pull()
    junta_tensor = lambda x,y: Variable(torch.cat(x*y, 0))
    state = Variable(torch.cat(states, 0))
    dis_reward = discount_rewards(self)
    pred_action = self.model(state)
    td_loss = self.loss(pred_action,state,dis_reward)
    self.optimizer.zero_grad()
    td_loss.backward()
    self.optimizer.step()

def save(self,name = 'last_brain.pth'):
    """
    Método utilizado para realizar o salvamento de uma rede que foi treinada
    """
    torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, name)
def load(self, name = 'last_brain.pth'):
    """
    Método que realiza o carregamento de uma rede treinada
    """
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
    """
    Desenvolvimento da função de perta utilizada para realizar o treina de rede do deep-Qlearning com PGR
    """
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLossW, self).__init__()

    def forward(self, inputs, targets,weight):

        entropy  = F.cross_entropy(inputs,targets,reduction='none')
        print(entropy)
        loss = (weight*entropy).mean()
        
        return loss