import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyActor(nn.Module):
    """
    Impementação utilizada pelo autor contínuo
    """
    def __init__(self, input_shape):
        super(PolicyActor,self).__init__()
        self.input_shape = input_shape
        self.layer1 = nn.Linear(self.input_shape, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.output_layer = nn.Tanh()
    def forward(self, input):
        x = F.relu(self.layer1(input))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        output = self.output_layer(x)
        return output

#class PpoLossContinuous(nn.Module):
    
class Actor():
    """
    Implementação da classe do ator utilizado no deep-Qlearning crítico ator.
    """
    def __init__(self,input_shape,action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.Model = PolicyActor(self.input_shape)


