import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from typing import Tuple
from dataclasses import dataclass
import math

@dataclass
class Stats:
    loss: float = 0
    value_loss: float = 0
    policy_loss: float = 0
    accuracy: float = 0
    n_batch: int = 0

    def __add__(self, other): 
        self.loss += other.loss
        self.value_loss += other.value_loss
        self.policy_loss += other.policy_loss
        self.accuracy += other.accuracy

        return self

    def log(self, batch, loss_log):
        self.n_batch = batch
        self.loss /= loss_log
        self.value_loss /= loss_log
        self.policy_loss /= loss_log
        self.accuracy /= loss_log

        print("Batch: {}, \
            loss: {}, \
            value_loss: {},\
            policy_loss: {} \
            accuracy: {}".format(self.n_batch, 
                            self.loss,  
                            self.value_loss,
                            self.policy_loss,
                            self.accuracy))
            

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
        #self.optimizer = optim.Adam(self.nn.parameters(), lr = lr, weight_decay = wd)
        #self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=lr, max_lr=0.1)
        #self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = scheduler_params['milestones'], gamma = scheduler_params['gamma'])

    def count_actions(self, predicted, label):
        best_actions = torch.eq(label, torch.max(label, dim=1, keepdim=True).values)
        best_predicted_actions = torch.eq(predicted, torch.max(predicted, dim=1, keepdim=True).values)

        matching_actions = (best_actions*best_predicted_actions).any(dim=1)
        return matching_actions.sum().item()/matching_actions.size(0)

    def train(self, data):
        self.nn.train()

        board, policy, value = data
        #board, policy, value = board.to(self.device),policy.to(self.device),value.to(self.device)
        board, policy, value = torch.Tensor(board).to(self.device), torch.Tensor(policy).to(self.device), torch.Tensor(value).to(self.device)
        self.optimizer.zero_grad()
        
        v, p = self.nn(board)
        loss, v_loss, p_loss = self.nn.loss((v, p), (value, policy))
        loss.backward()
        
        count = self.count_actions(p, policy)

        self.optimizer.step()
        #self.scheduler.step()

        return Stats(loss = loss.item(), 
                    value_loss = v_loss.item(), 
                    policy_loss = p_loss.item(),
                    accuracy = count)
    
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

    def load_model(self, path = "temp/models/model_new.pt", load_optim = False):
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

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    

class ConvLayer(nn.Module):
    def __init__(self, board_dim = (), inplanes = 1, planes=128, stride=1):
        super(ConvLayer, self).__init__()
        self.inplanes = inplanes
        self.board_dim = board_dim
        self.conv = Conv2dSamePadding(inplanes, planes, kernel_size=4, stride=stride)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, s):
        s = s.view(-1, self.inplanes, self.board_dim[0], self.board_dim[1])  # batch_size x planes x board_x x board_y
        s = F.relu(self.bn(self.conv(s)))

        return s

class ResLayer(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1):
        super(ResLayer, self).__init__()
        self.conv1 = Conv2dSamePadding(inplanes, planes, kernel_size=4, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dSamePadding(planes, planes, kernel_size=4, stride=stride)
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

        self.conv1 = Conv2dSamePadding(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        
        self.softmax = nn.Softmax(dim=1)
        
        #if self.output_planes > 1:
        self.conv2 = Conv2dSamePadding(32, self.output_planes, kernel_size=1) # policy head
        #else:
        self.fc = nn.Linear(self.board_dim[0]*self.board_dim[1]*32, self.action_size)

    def forward(self,s):
        p = F.relu(self.bn1(self.conv1(s))) # policy head

        if self.output_planes > 1:
            p = self.conv2(p)
        else:
            p = p.view(-1, self.board_dim[0]*self.board_dim[1]*32)
            p = self.fc(p)
            
        p = self.softmax(p)
        return p


class ValueHead(nn.Module):
    def __init__(self, board_dim = (3,3)):
        super(ValueHead, self).__init__()
        self.board_dim = board_dim
        self.conv = Conv2dSamePadding(128, 32, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(self.board_dim[0]*self.board_dim[1]*32, 1) 

    def forward(self,s):
        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, self.board_dim[0]*self.board_dim[1]*32)  # batch_size X channel X height X width
        v = torch.tanh(self.fc1(v))

        return v   

