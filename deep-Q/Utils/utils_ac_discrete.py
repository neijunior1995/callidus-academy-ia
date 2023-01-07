import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

def push( event, memory, capacity):
    """
    Método utilizado para atualizar e adicionar novos eventos a memoria
    """
    
    memory.append(event)
    if len(memory) > capacity:
        del memory[0]
    return memory
def sample(memory):
    """
    Método utilizado para realizar a escolha aleatória de dados do dataset
    e assim realizar um novo treinamento
    """
    samples = zip(*random.sample(memory, 2))
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

def learn(self, batch_states, batch_actions, batch_rewards, batch_next_states):
    "Implementação do treinamento da rede de aprendizagem profunda utilizada"
    batch_outputs = self.model(batch_states).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
    batch_next_outputs = self.model(batch_next_states).detach().max(1)[0]
    batch_targets = batch_rewards + self.gamma * batch_next_outputs
    td_loss = self.loss(batch_outputs, batch_targets)
    self.optimizer.zero_grad()
    td_loss.backward()
    self.optimizer.step()
def update(self, new_state, new_reward):
    """
    Implementação do método que realiza a ação e atualiza os pesos da rede
    """
    new_state = torch.Tensor(new_state).float().unsqueeze(0)
    self.memory.push((self.last_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward]), new_state))
    new_action = self.select_action(new_state)
    if len(self.memory.memory) > self.batch_size:
        batch_states, batch_actions, batch_rewards, batch_next_states = self.memory.sample(self.batch_size)
        self.learn(batch_states, batch_actions, batch_rewards, batch_next_states)
    self.last_state = new_state
    self.last_action = new_action
    self.last_reward = new_reward
    return new_action.item()
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