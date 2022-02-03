import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import sys
sys.path.append('../layers/')
import equivariant_linear_pytorch as eq

class invariant_basic(nn.Module):
    def __init__(self, config, data, device='cpu'):
        super(invariant_basic, self).__init__()
        self.config = config
        self.data = data
        self.device = device
        self.build_model()

    def build_model(self):
        # build network architecture using config file
        self.equi_layers = nn.ModuleList()
        self.equi_layers.append(eq.layer_2_to_2(self.data.train_graphs[0].shape[0], self.config.architecture[0], device=self.device))
        L = len(self.config.architecture)
        for layer in range(1, L):
            self.equi_layers.append(eq.layer_2_to_2(self.config.architecture[layer - 1], self.config.architecture[layer], device=self.device))
        out_dim = 8 # 32 # 128 # 1024
        self.equi_layers.append(eq.layer_2_to_1(self.config.architecture[L - 1], out_dim, device=self.device))
        self.fully1 = nn.Linear(out_dim, out_dim) # nn.Linear(1024, 512)
        self.fully2 = nn.Linear(out_dim, out_dim) # nn.Linear(512, 256)
        self.fully3 = nn.Linear(out_dim, self.config.num_classes) # nn.Linear(256, self.config.num_classes)

    def forward(self, inputs):
        outputs = inputs
        for layer in range(len(self.equi_layers)):
            outputs = self.equi_layers[layer](outputs)
            outputs = F.relu(outputs) # important: added relu by Chen
        # outputs = torch.sum(outputs, dim = 2) # important: changed by Chen;
        outputs = torch.mean(outputs, dim = 2)
        # outputs = torch.mean(outputs) # would it help?
        return outputs
        outputs = F.relu(self.fully1(outputs))
        outputs = F.relu(self.fully2(outputs))
        return F.log_softmax(self.fully3(outputs))
        # return self.fully3(outputs) # todo: try other layers
