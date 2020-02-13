import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from typing import Tuple

class NetWrapper(object):
    def __init__(self):
        super(NetWrapper, self).__init__()

    def build(self, input_planes, board_dim, action_size, output_planes, res_layer_number):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn = AlphaZeroNet(input_planes = input_planes, 
                              board_dim = board_dim, 
                              action_size = action_size, 
                              output_planes = output_planes, 
                              res_layer_number = res_layer_number
                              ).to(self.device)

    def build_optim(self, lr = 0.01, wd = 0.05, momentum=0.9, scheduler_params = None):
        self.optimizer = optim.SGD(self.nn.parameters(), lr = lr, weight_decay = wd)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.1)
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = scheduler_params['milestones'], gamma = scheduler_params['gamma'])

    def train(self, data):
        self.nn.train()

        board, policy, value = data
        board, policy, value = torch.Tensor(board).to(self.device), torch.Tensor(policy).to(self.device), torch.Tensor(value).to(self.device)

        self.optimizer.zero_grad()
        
        v, p = self.nn(board)
        loss, v_loss, p_loss = self.nn.loss((v, p), (value, policy))
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), v_loss.item(), p_loss.item()
    
    def predict(self, board):
        self.nn.eval()
        board = torch.Tensor(board)
        with torch.no_grad():
            v, p = self.nn(board)

        p = p.detach().numpy()
        return v, p

    def save_model(self, folder = "models", model_name = "model.pt"):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        torch.save({
            'model_state_dict': self.nn.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, "{}/{}".format(folder, model_name))

    def save_traced_model(self, folder = "models", model_name = "model.pt"):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        self.nn.eval()
        self.nn.to(self.device)

        model_loc = "{}/{}".format(folder, model_name)
        traced_model = torch.jit.script(self.nn)
        traced_model.save(model_loc)

        return model_loc

    def load_model(self, path = "models/fdsmodel.pt", load_optim = False):
        cp = torch.load(path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn.load_state_dict(cp['model_state_dict'])
        if load_optim:   
            self.optimizer = optim.Adam(self.nn.parameters(), lr = 0.1, weight_decay = 0.005)
            self.optimizer.load_state_dict(cp['optimizer_state_dict'])
        
        print("Netwrapper: model loaded")

        return self.nn
    
    def load_traced_model(self, path = "models/traced_model_new.pt"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.nn = torch.jit.load(path)
        self.nn.to(self.device)

        print("Netwrapper: Traced model loaded")
        return self.nn

class AlphaZeroNet(nn.Module):
    def __init__(self, input_planes, board_dim, action_size, output_planes, res_layer_number):
        super(AlphaZeroNet, self).__init__()
        self.conv = ConvLayer(board_dim = board_dim, inplanes = input_planes)
        self.res_layers = torch.nn.ModuleList([ResLayer() for i in range(res_layer_number)])
        self.valueHead = ValueHead(board_dim = board_dim)
        self.policyHead = PolicyHead(board_dim = board_dim, action_size = action_size, output_planes = output_planes)
        
    def forward(self,s):
        s = self.conv(s)

        for res_layer in self.res_layers:
            s = res_layer(s)

        v = self.valueHead(s)
        p = self.policyHead(s)

        return v, p
    
    @torch.jit.export
    def loss(self, predicted : Tuple[torch.Tensor, torch.Tensor], label: Tuple[torch.Tensor, torch.Tensor]):
        (v, p) = predicted
        (z, pi) = label
        
        value_error = (z.float() - torch.transpose(v,0,1))**2
        policy_error = (pi.float()*p.log()).sum(1)

        return value_error.mean() - policy_error.mean(), value_error.mean(), - policy_error.mean() #no need to add the l2 regularization term as it is done in the optimizer

class ConvLayer(nn.Module):
    def __init__(self, board_dim = (), inplanes = 1, planes=128, stride=1):
        super(ConvLayer, self).__init__()
        self.inplanes = inplanes
        self.board_dim = board_dim
        print(board_dim)
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, s):
        s = s.view(-1, self.inplanes, self.board_dim[0], self.board_dim[1])  # batch_size x planes x board_x x board_y
        s = F.relu(self.bn(self.conv(s)))

        return s

class ResLayer(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1):
        super(ResLayer, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        
        return out


class PolicyHead(nn.Module):
    def __init__(self, board_dim = (), action_size = -1, output_planes = -1):
        super(PolicyHead, self).__init__()
        self.board_dim = board_dim
        self.action_size = action_size
        self.output_planes = output_planes

        self.conv1 = nn.Conv2d(128, 128, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(128)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        #if self.output_planes > 1:
        self.conv2 = nn.Conv2d(128, self.output_planes, kernel_size=1) # policy head
        #else:
        self.fc = nn.Linear(self.board_dim[0]*self.board_dim[1]*128, self.action_size)

    def forward(self,s):
        p = F.relu(self.bn1(self.conv1(s))) # policy head

        if self.output_planes > 1:
            p = self.conv2(p)
        else:
            p = p.view(-1, self.board_dim[0]*self.board_dim[1]*128)
            p = self.fc(p)
            
        p = self.logsoftmax(p).exp()

        return p


class ValueHead(nn.Module):
    def __init__(self, board_dim = (3,3)):
        super(ValueHead, self).__init__()
        self.board_dim = board_dim
        self.conv = nn.Conv2d(128, 1, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(self.board_dim[0]*self.board_dim[1], 128) 
        self.fc2 = nn.Linear(128, 1)

    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, self.board_dim[0]*self.board_dim[1])  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        return v   

