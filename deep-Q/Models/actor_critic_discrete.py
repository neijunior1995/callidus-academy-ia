import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Policy(nn.Module):
    """
    Impementação utilizada pelo autor contínuo
    """
    def __init__(self, input_shape,output_shape,):
        super(Policy,self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.layer1 = nn.Linear(self.input_shape, 512)
        self.layer2(512, self.output_shape)
        self.output_layer = nn.Softmax(dim = -1)
    def forward(self, input):
        x = F.elu(self.layer1(input))
        x = F.relu(self.layer2(x))
        output = self.output_layer(x)
        return output
    
class Agent():
    """
    # Algoritmo de otimização principal do gradiente de política.
    """
    def __init__(self,input_shape,action_space):
        self.input_shape = input_shape
        self.action_space = action_space
        self.Model = PolicyActor(self.input_shape)


