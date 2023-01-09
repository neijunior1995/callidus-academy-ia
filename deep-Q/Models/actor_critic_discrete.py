import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Utils.utils_ac_discrete import *


class PolicyActor(nn.Module):
    """
    Impementação utilizada pelo autor contínuo
    """
    def __init__(self, input_shape,output_shape,):
        super(PolicyActor,self).__init__()
        self.input_size = input_shape
        self.output_size = output_shape
        self.layer1 = nn.Linear(self.input_size, 512)
        self.layer2 = nn.Linear(512, self.output_size)
    def forward(self, input):
        x = F.elu(self.layer1(input))
        x = F.relu(self.layer2(x))
        return x
    
class Agent():
    """
    # Algoritmo de otimização principal do gradiente de política.
    """
    """
    Metodo de inicializacao da DQN
    """
    def __init__(self, input_size, nb_action,
                 gamma =0.99 , hidden_layer = 45,
                 capacity = 100000, batch_size = 100,
                 lr = 0.000025):
        self.gamma = gamma
        self.lr = lr
        self.input_size = input_size
        self.output_size = nb_action
        print(self.output_size)
        self.batch_size = batch_size
        self.capacity = capacity
        self.model = PolicyActor(self.input_size,self.output_size)
        self.memory = ReplayMemory(self.capacity)
        self.optimizer = optim.Adam(params = self.model.parameters())
        self.last_state = torch.Tensor(self.input_size).unsqueeze(0)
        self.loss = CrossEntropyLossW()
        self.last_action = 0
        self.last_reward = 0

    def remember(self,state,action,reward):
        """
        Método utilizado para salvar os estados, ações e recompensas da rede
        """
        remember(self,state,action,reward)
   
    def replay(self):
        """
        Método utilizado para 
        """
        return self.memory.pull()
   
    def discount_rewards(self):
        """
        Método que implementa a politica de recompensa da rede
        """
        return discount_rewards(self)
    def select_action(self, state):
        """
        Metodo que seleciona a acao que sera tomada a partir da probilidade obtida
        por meio da DQN.
        """
        return select_action(state, self.model)

    def discount_rewards(self, reward):
        """
        Método de desconto de recompensas que pode ser adaptado para cada caso
        """
        discount_rewards(self, reward)

    def learn(self):
        """
        Metodo utilizado para implementar o aprendizado da rede
        """
        learn(self)

    def save(self):
        """
        Metodo que salva a rede treinada
        """
        save(self)

    def load(self):
        """
        Metodo que carrega uma rede treinada
        """
        load(self)