import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

def push(self, event, memory, capacity):
    """
    Método utilizado para atualizar e adicionar novos eventos a memoria
    """
    memory.append(event)
    if len(memory) > capacity:
        del memory[0]
    return memory
def sample(memory,batch_size)
    """
    Método utilizado para realizar a escolha aleatória de dados do dataset
    e assim realizar um novo treinamento
    """
    samples = zip(*random.sample(memory, batch_size))
    return map(lambda x: Variable(torch.cat(x, 0)), samples)