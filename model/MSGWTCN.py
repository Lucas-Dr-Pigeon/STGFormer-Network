import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, BatchNorm2d, ModuleList, Parameter
import torch.nn.functional as F
import math
from tqdm import trange
import time
from torch.autograd import Variable
import numpy as np

def nconv(x, phi):
    return torch.einsum('ncvl, vw->ncwl', (x, phi)).contiguous()

class GraphWaveletLayer(torch.nn.Module):
    """docstring for GraphWaveletLayer"""
    def __init__(self, in_channels, out_channels, num_nodes, device):
        super(GraphWaveletLayer, self).__init__()
        self.in_channels = num_nodes
        self.out_channels = num_nodes
        self.device = device
        self.final_conv = Conv2d(in_channels, out_channels, (1,1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = 0.3
        self.define_parameters()
        self.init_parameters()


    def define_parameters(self):
        self.diag_weight_L = torch.nn.Parameter(torch.Tensor(self.in_channels).to(self.device))
        self.diag_weight_H = torch.nn.Parameter(torch.Tensor(self.in_channels).to(self.device))
        self.bias = torch.nn.Parameter(torch.FloatTensor(self.in_channels).to(self.device))

    def init_parameters(self): 
        stdv = 1. / math.sqrt(self.in_channels)
        self.diag_weight_L.data.uniform_(-stdv,stdv)
        self.diag_weight_H.data.uniform_(-stdv,stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, phi_matrices,phi_inverse_matrices):
        phi_product_matrices_L = phi_matrices[0].matmul(torch.diag(self.diag_weight_L)).matmul(phi_inverse_matrices[0])
        phi_product_matrices_M = phi_matrices[1].matmul(torch.diag(self.diag_weight_L)).matmul(phi_inverse_matrices[1])
        phi_product_matrices_H = phi_matrices[2].matmul(torch.diag(self.diag_weight_H)).matmul(phi_inverse_matrices[2])

        x = input.permute(0,1,3,2)
        x_L = torch.matmul(x, phi_product_matrices_L)
        x_M = torch.matmul(x, phi_product_matrices_M)
        x_H = torch.matmul(x, phi_product_matrices_H)
        #h = F.linear(x, phi_product_matrices, self.bias)
        #h = (x_L+x_H)/2
        #h = torch.max(x_L, x_H)
        h = x_L+x_M+x_H
        h = h.permute(0,1,3,2)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)

        return h
    
class GraphConvLayer(torch.nn.Module):
    """docstring for GraphWaveletLayer"""
    def __init__(self, in_channels, out_channels, device, adj):
        super(GraphConvLayer, self).__init__()
        self.in_channels = 323
        self.out_channels = 323
        self.device = device
        self.final_conv = Conv2d(in_channels, out_channels, (1,1), padding=(0,0), stride=(1,1), bias=True)
        self.dropout = 0.3
        self.adj = adj.to(device)
        self.define_parameters()
        self.init_parameters()


    def define_parameters(self):
        self.diag_weight = torch.nn.Parameter(torch.Tensor(self.in_channels).to(self.device))
        self.bias = torch.nn.Parameter(torch.FloatTensor(self.in_channels).to(self.device))

    def init_parameters(self): 
        stdv = 1. / math.sqrt(self.in_channels)
        self.diag_weight.data.uniform_(-stdv,stdv)
        self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input, phi_matrices,phi_inverse_matrices):
        x = input.permute(0,1,3,2)
        h = F.linear(x, self.adj, self.bias)
        h = h.permute(0,1,3,2)
        h = self.final_conv(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h        

class GWTCN(torch.nn.Module):
    """docstring for TCNLayer"""
    def __init__(self, in_channels=2, out_channels=12, residual_channels=32, 
        dilation_channels=32, skip_channels=256, end_channels=512,
        kernel_size=2, blocks=4, layers=2, num_nodes=0, wavelet_phi_matrices=None, wavelet_inverse_phi_matrices=None):
        super(GWTCN, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blocks = blocks
        self.layers = layers
        self.phi_matrices = wavelet_phi_matrices 
        self.phi_inverse_matrices = wavelet_inverse_phi_matrices

        self.start_conv = Conv2d(in_channels = 1,
                                    out_channels = residual_channels,
                                    kernel_size = (1,1))
    
        #w_in = 323
        #self.diag_weight = torch.nn.Parameter(torch.Tensor(w_in).to(device))
        #stdv = 1. / math.sqrt(w_in)
        #self.diag_weight.data.uniform_(-stdv,stdv)

        receptive_field = 1
        depth = list(range(blocks*layers))

        self.residual_convs = ModuleList([Conv1d(dilation_channels, residual_channels, (1,1)) for _ in depth])
        self.skip_convs = ModuleList([Conv1d(dilation_channels, skip_channels, (1, 1)) for _ in depth])
        self.bn = ModuleList([BatchNorm2d(residual_channels) for _ in depth])
        self.graph_convs = ModuleList([GraphWaveletLayer(dilation_channels, residual_channels, num_nodes, device) for _ in depth])
        #self.graph_convs = ModuleList([GraphConvLayer(dilation_channels, residual_channels, device, adj) for _ in depth])
        
        self.filter_convs = ModuleList()
        self.gate_convs = ModuleList()

        for b in range(blocks):
            additional_scope = kernel_size - 1
            D = 1 #dilation
            for i in range(layers):
                #dilated convolutions
                self.filter_convs.append(Conv2d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                self.gate_convs.append(Conv1d(residual_channels, dilation_channels, (1, kernel_size), dilation=D))
                D *= 2
                receptive_field += additional_scope
                additional_scope *= 2
            self.receptive_field = receptive_field

            self.end_conv_1 = Conv2d(skip_channels, end_channels, (1,1), bias=True)
            self.end_conv_2 = Conv2d(end_channels, out_channels, (1,1), bias=True)

    def forward(self, x):
        #Update: Input shape is (batch_size, n_timesteps, n_nodes, features)
        #Input shape is (batch_size, features, n_nodes, n_timesteps)
        # x = x.unsqueeze(3)
        # x = x.permute(0,3,2,1)
        # we want input shape (batch_size, n_timestamps, n_nodes, features)
        # print (f'-- Debug input shape: {x.shape} --')
        x = x.permute(0,3,2,1)
        # print (f'-- Debug x shape before padding: {x.shape} --')
        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0,0,0))
        # print (f'-- Debug x shape after padding: {x.shape} --')
        x = self.start_conv(x)
        skip = 0
        #TCN Layers
        for i in range(self.blocks*self.layers):
            #Each block
            residual = x
            #dilated convolution
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter*gate
            #parameterized skip connection
            s = self.skip_convs[i](x)
            try:
                skip = skip[:,:,:, -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            if i == (self.blocks*self.layers - 1): #last X getting ignored anyway
                break
            #x = x.permute(0,3,2,1).squeeze(-1)
            x = self.graph_convs[i](x, self.phi_matrices, self.phi_inverse_matrices)
            #x = self.residual_convs[i](x)
            #x = x.unsqueeze(-1)
            x = x + residual[:,:,:, -x.size(3):]
            x = self.bn[i](x)
        # print (f"--debug skip after GWTCN: {skip.shape}--")
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        # print (f"--debug x after end_conv1: {x.shape}--")
        x = self.end_conv_2(x) #downsample to (bs, seq_lenth, n_nodes, n_features)
        # 
        # print (f"--debug x after end_conv2: {x.shape}--")
        output = x
        return output

