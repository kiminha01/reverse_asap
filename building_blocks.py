import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.cpp_extension import load
import numpy as np
import os
from random import *

cudnn_convolution = load(name="doing_convolution_layer_", sources=["cudnn_conv.cpp"], verbose=True)

#%%
"""
Feedback Alignment

Instead of using transposed weight of forward path,
we use weight_fa as random fixed weight for making grad_input.
The weight_fa is fixed because grad_weight_fa = 0
"""

class linear_fa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_fa):
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input,  bias, weight, weight_fa)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weight,weight_fa = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_weight_fa = None
       
        grad_weight = F.linear(input.t(), grad_output.t()).t()
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = F.linear(grad_output, weight_fa.t())
    
        return grad_input, grad_weight, grad_bias, grad_weight_fa

class Linear_FA(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear_FA, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).zero_())
        else:
            self.register_parameter('bias', None)
        self.weight_fa = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-1,1), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, input):
        return linear_fa.apply(input, self.weight, self.bias, self.weight_fa)

class linear_KP(linear_fa):
    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_bias, grad_weight_fa = linear_fa.backward(ctx, grad_output)
        grad_weight_fa=grad_weight
        return grad_input, grad_weight, grad_bias, grad_weight_fa
class Linear_KP(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        #super().__init__(in_features, out_features, bias)
        super(Linear_KP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).zero_())
        else:
            self.register_parameter('bias', None)
        self.weight_fa = nn.Parameter(torch.FloatTensor(out_features, in_features).uniform_(-1,1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, input):
        return linear_KP.apply(input, self.weight, self.bias, self.weight_fa )


class conv2d_fa(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, weight_fa, stride=1, padding=0, groups=1):
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.save_for_backward(input, bias, weight, weight_fa)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weight, weight_fa = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = grad_weight_fa = None
        grad_weight = cudnn_convolution.convolution_backward_weight(input, weight_fa.shape, grad_output, stride, padding, (1, 1), groups, False, False)
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(input.shape, weight_fa, grad_output, stride, padding, (1, 1), groups, False, False)
        return grad_input, grad_weight, grad_bias, grad_weight_fa, None, None, None, None, None, None

class Conv2d_FA(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(Conv2d_FA, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        
        self.weight_fa = nn.Parameter(self.weight, requires_grad=True)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        
    def forward(self, input):
        return conv2d_fa.apply(input, self.weight, self.bias, self.weight_fa, self.stride, self.padding, self.groups)

class conv2d_KP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias,weight_fa, stride=1, padding=0, dilation=1, groups=1):
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, dilation, groups, False, False)
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.dilation = dilation
        ctx.save_for_backward(input, bias, weight, weight_fa )
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weight, weight_fa = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        dilation = ctx.dilation
        grad_input = grad_weight = grad_bias = grad_weight_fa = None
        grad_weight = cudnn_convolution.convolution_backward_weight(input, weight_fa.shape, grad_output, stride, padding, dilation, groups, False, False)
        import pdb
        pdb.set_trace()
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        grad_input = cudnn_convolution.convolution_backward_input(input.shape, weight_fa, grad_output, stride, padding, dilation, groups, False, False)
        grad_weight_fa = grad_weight
        return grad_input, grad_weight, grad_bias, grad_weight_fa, None, None, None, None

class Conv2d_KP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(Conv2d_KP, self).__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,padding=padding, groups=groups)
        
        self.weight_fa = nn.Parameter(self.weight, requires_grad=True)
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))
        
    def forward(self, input):
        return conv2d_KP.apply(input, self.weight, self.bias, self.weight_fa, self.stride, self.padding, self.dilation, self.groups)
#%%
""""
Direct Feedback alignment

Feedback_Receiver module receives top error and transforms the top error through random fixed weights.
First, it makes dummy data and sends it to Top_Gradient module 
which distributes top error in forward prop.
And then, top error from Top_Gradient module is transformed by weight_fb in backward prop 

Top_Gradient module sends top error to lower layers which is made by loss function.
First, it receives dummy data from layers that will receive errors in forward prop.
And then, top error is sent to the layers that gave the dummy data in backward prop.

So, the Feedback_Receiver module is located behind the layer that wants to receive the error, 
and the Top_Gradient module is located at the end of the architecture. 
The dummy created in Feedback_Receiver must be accepted in Top_Gradient.
"""
class feedback_receiver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_fb):
        output = input.clone()
        dummy = torch.Tensor(input.size()[0],weight_fb.size()[0]).zero_().to(input.device)
        ctx.save_for_backward(weight_fb,)
        ctx.shape = input.shape
        return output, dummy
    
    @staticmethod
    def backward(ctx, grad_output, grad_dummy):
        weight_fb, = ctx.saved_tensors
        input_size = ctx.shape
        grad_weight_fb = None
        
        grad_input = torch.mm(grad_dummy.view(grad_dummy.size()[0],-1), weight_fb).view(input_size) # Batch_size, input
        return grad_input, grad_weight_fb


class Feedback_Receiver(nn.Module):
    def __init__(self, connect_features):
        super(Feedback_Receiver, self).__init__()
        self.connect_features = connect_features
        self.weight_fb = None
    
    def forward(self, input):
        if self.weight_fb is None:
            self.weight_fb = nn.Parameter(torch.Tensor(self.connect_features, *input.size()[1:]).view(self.connect_features, -1)).to(input.device)
            nn.init.normal_(self.weight_fb, std = math.sqrt(1./self.connect_features))
        return feedback_receiver.apply(input, self.weight_fb)
   
class top_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, *dummies):
        output = input.clone()
        ctx.save_for_backward(output ,*dummies)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, *dummies = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_dummies = [grad_output.clone() for dummy in dummies]
        return tuple([grad_input, *grad_dummies])

class Top_Gradient(nn.Module):
    def __init__(self):
        super(Top_Gradient, self).__init__()
    
    def forward(self, input, *dummies):
        return top_gradient.apply(input, *dummies)
#%%
'''Local Activity makes local classifier for immediate activations'''

class Local_Activity(nn.Module):
    def __init__(self, connect_features=10):
        super(Local_Activity, self).__init__()
        self.local_classifier = None
        self.connect_features = connect_features
        
    def forward(self, x):
        # first, there is no local classifier module(None)
        # When learning is start, local classifier is made using input size. 
        # And then, it is fixed
        if self.local_classifier == None:    
            if len(x.size()) == 4: # for immediate activation after conv
                self.local_classifier = nn.Sequential(
                            nn.Flatten(),
                            Linear_Fixed(x.size(1)*x.size(2)*x.size(3),self.connect_features)).to(x.device)
            elif len(x.size()) == 2: # for immediate activation after linear 
                self.local_classifier = nn.Sequential(
                            nn.Flatten(),
                            Linear_Fixed(x.size(1),self.connect_features)).to(x.device)
                
        self.local_activity = self.local_classifier(x)
        
        return x.detach() # x.detach() cuts back-propagation of global error. Therefore, it makes local learning.
    
class linear_fixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        output = F.linear(input, weight, bias)
        ctx.save_for_backward(input,  bias, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias= None
       
        grad_weight = None
        if bias is not None:
            grad_bias = grad_output.sum(0).squeeze(0)
        grad_input = F.linear(grad_output, weight.t())
        return grad_input, grad_weight, grad_bias

class Linear_Fixed(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(Linear_Fixed, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).zero_())
        else:
            self.register_parameter('bias', None)
        nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, input):
        return linear_fixed.apply(input, self.weight, self.bias)
#%%
class conv2d_fa_asap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, shared, weight, weight_fa, bias, stride=1, padding=0, groups=1, wt = False):
        if shared == None:
            output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
            return output

        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        shared = shared.detach().clone()
        shared_channel_ratio = int(input.size(1)/shared.size(1))#for matching shared activation with actual activation
        shared_filter_ratio = int(((shared.size(2)/input.size(2)))) #for matching shared activation with actual activation
        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.shared_channel_ratio = shared_channel_ratio 
        ctx.shared_filter_ratio = shared_filter_ratio
        ctx.wt = wt
        ctx.save_for_backward(input, weight, weight_fa, bias, shared)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_fa, bias, shared = ctx.saved_tensors
        if shared == None:
            return None, None, None, None, None, None, None, None, None
        else:
            print(str(shared.size(1)) + ':' + str(torch.norm(shared)))
        
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        wt = ctx.wt
        shared_channel_ratio = ctx.shared_channel_ratio
        shared_filter_ratio = ctx.shared_filter_ratio
        grad_input = grad_weight = grad_bias  = None
        
        # Matching shared activation with actual activation by concatenation and maxpool.
        
        shared = torch.cat([shared] * shared_channel_ratio, 1)
        shared = F.max_pool2d(shared, shared_filter_ratio)       
        
        if wt:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is weight transport. 
            grad_weight = cudnn_convolution.convolution_backward_weight(shared, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = None
            grad_input = cudnn_convolution.convolution_backward_input(shared.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False)
        else:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is no weight transport. 
            # we traind weight_fa by grad_weight_fa = grad_weight
            grad_weight = cudnn_convolution.convolution_backward_weight(shared, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = grad_weight
            grad_input = cudnn_convolution.convolution_backward_input(shared.shape, weight_fa, grad_output, stride, padding, (1, 1), groups, False, False)
            
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        
        return grad_input, None, grad_weight, grad_weight_fa, grad_bias, None, None, None, None

class Conv2d_FA_ASAP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, wt = False):
        super(Conv2d_FA_ASAP, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.weight_fa = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.wt = wt # by using wt, Activation Saring with weight transport is possible.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))      
        
    def forward(self, input, shared):
        return conv2d_fa_asap.apply(input, shared, self.weight, self.weight_fa, self.bias, self.stride, self.padding, self.groups, self.wt) 
class ASAP_Conv_Block(nn.Module): # for ASAP in ConvNet
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, wt = False):
        super(ASAP_Conv_Block, self).__init__()
        
        self.conv = Conv2d_FA_ASAP(in_channels, out_channels, kernel_size, stride, padding, wt = wt)
        self.bn = nn.BatchNorm2d(out_channels)
        self.max = nn.MaxPool2d(1,1, ceil_mode = True)
        
    def forward(self, x, save):
        x = self.conv(x, save)
        x = self.max(F.relu(self.bn(x)))
        return x
class ASAP_Reverse_Block(nn.Module): # for ASAP in ConvNet
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, wt = False):
        super(ASAP_Reverse_Block, self).__init__()
        
        self.conv = Conv2d_FA_ASAP(in_channels, out_channels, kernel_size, stride, padding, wt = wt)
        self.bn = nn.BatchNorm2d(out_channels)
        
        self.max = nn.MaxPool2d(1,1, ceil_mode = True)
        
    def forward(self, x, save):
        x = self.conv(x, save)
        x = self.max(F.relu(self.bn(x)))
        return x
#%%
class conv2d_fa_asap0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, shared, weight, weight_fa, bias, stride=1, padding=0, groups=1, wt = False):
        
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        shared = shared.detach().clone()
        shared_channel_ratio = int(input.size(1)/shared.size(1)) #for matching shared activation with actual activation
        shared_filter_ratio = int(shared.size(2)/input.size(2)) #for matching shared activation with actual activation
        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.shared_channel_ratio = shared_channel_ratio 
        ctx.shared_filter_ratio = shared_filter_ratio
        ctx.wt = wt
        ctx.save_for_backward(input, weight, weight_fa, bias, shared)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_fa, bias, shared = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        wt = ctx.wt
        
        shared_channel_ratio = ctx.shared_channel_ratio
        shared_filter_ratio = ctx.shared_filter_ratio
        grad_input = grad_weight = grad_bias  = None
        
        # Matching shared activation with actual activation by concatenation and maxpool.
        shared = torch.cat([shared] * shared_channel_ratio, 1)
        shared = F.max_pool2d(shared, shared_filter_ratio)       
        
        if wt:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is weight transport. 
            grad_weight = cudnn_convolution.convolution_backward_weight(shared, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = None
            grad_input = cudnn_convolution.convolution_backward_input(shared.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False)
        else:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is no weight transport. 
            # we traind weight_fa by grad_weight_fa = grad_weight
            grad_weight = cudnn_convolution.convolution_backward_weight(shared, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = grad_weight
            grad_input = cudnn_convolution.convolution_backward_input(shared.shape, weight_fa, grad_output, stride, padding, (1, 1), groups, False, False)
            
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        
        return grad_input, None, grad_weight, grad_weight_fa, grad_bias, None, None, None, None
class Conv2d_FA_ASAP0(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, wt = False):
        super(Conv2d_FA_ASAP0, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.weight_fa = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.wt = wt # by using wt, Activation Saring with weight transport is possible.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))      
        
    def forward(self, input, shared):
        return conv2d_fa_asap0.apply(input, shared, self.weight, self.weight_fa, self.bias, self.stride, self.padding, self.groups, self.wt) 

class ASAP_Conv_Block0(nn.Module): # for ASAP in ConvNet
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, wt = False):
        super(ASAP_Conv_Block0, self).__init__()
        
        self.conv = Conv2d_FA_ASAP0(in_channels, out_channels, kernel_size, stride, padding, wt = wt)
        self.bn = nn.BatchNorm2d(out_channels)
        self.max = nn.MaxPool2d(1,1, ceil_mode = True)
        
    def forward(self, x, save):
        x = self.conv(x, save)
        x = self.max(F.relu(self.bn(x)))
        return x
    
#%%
class rfeedback_receiver(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight_fb):
        output = input.clone()
        dummy = torch.Tensor(input.size()[0],weight_fb.size()[0]).zero_().to(input.device)
        ctx.save_for_backward(weight_fb,)
        ctx.shape = input.shape
        return output, dummy
    
    @staticmethod
    def backward(ctx, grad_output, grad_dummy):
        weight_fb, = ctx.saved_tensors
        input_size = ctx.shape
        grad_weight_fb = None
        
        grad_input = torch.mm(grad_dummy.view(grad_dummy.size()[0],-1), weight_fb).view(input_size) # Batch_size, input
        return grad_input, grad_weight_fb


class RFeedback_Receiver(nn.Module):
    def __init__(self, connect_features):
        super(RFeedback_Receiver, self).__init__()
        self.connect_features = connect_features
        self.weight_fb = None
    
    def forward(self, input):
        if self.weight_fb is None:
            self.weight_fb = nn.Parameter(torch.Tensor(self.connect_features, *input.size()[1:]).view(self.connect_features, -1)).to(input.device)
            nn.init.normal_(self.weight_fb, std = math.sqrt(1./self.connect_features))
        return rfeedback_receiver.apply(input, self.weight_fb)
   
class rtop_gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, *dummies):
        output = input.clone()
        ctx.save_for_backward(output ,*dummies)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, *dummies = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_dummies = [grad_output.clone() for dummy in dummies]
        return tuple([grad_input, *grad_dummies])
    
class conv2d_fa_asap0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, shared, weight, weight_fa, bias, stride=1, padding=0, groups=1, wt = False):
        
        dum = torch.zeros_like(input).to(input.device)
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        shared = shared.detach().clone()
        shared_channel_ratio = int(input.size(1)/shared.size(1)) #for matching shared activation with actual activation
        shared_filter_ratio = int(shared.size(2)/input.size(2)) #for matching shared activation with actual activation
        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.shared_channel_ratio = shared_channel_ratio 
        ctx.shared_filter_ratio = shared_filter_ratio
        ctx.wt = wt
        ctx.save_for_backward(input, weight, weight_fa, bias, shared)
        
        return output, dum
    
    @staticmethod
    def backward(ctx, grad_output, grad_dum):
        input, weight, weight_fa, bias, shared = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        wt = ctx.wt
        
        shared_channel_ratio = ctx.shared_channel_ratio
        shared_filter_ratio = ctx.shared_filter_ratio
        grad_input = grad_weight = grad_bias  = None
        
        # Matching shared activation with actual activation by concatenation and maxpool.
        shared = torch.cat([shared] * shared_channel_ratio, 1)
        shared = F.max_pool2d(shared, shared_filter_ratio)       
        
        if wt:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is weight transport. 
            grad_weight = cudnn_convolution.convolution_backward_weight(shared, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = None
            grad_input = cudnn_convolution.convolution_backward_input(shared.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False)
        else:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is no weight transport. 
            # we traind weight_fa by grad_weight_fa = grad_weight
            grad_weight = cudnn_convolution.convolution_backward_weight(grad_dum, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = grad_weight
            grad_input = cudnn_convolution.convolution_backward_input(grad_dum.shape, weight_fa, grad_output, stride, padding, (1, 1), groups, False, False)
            
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        
        return grad_input, None, grad_weight, grad_weight_fa, grad_bias, None, None, None, None
class Conv2d_FA_ASAP0(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True, wt = False):
        super(Conv2d_FA_ASAP0, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.weight_fa = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        self.wt = wt # by using wt, Activation Saring with weight transport is possible.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fa, a=math.sqrt(5))      
        
    def forward(self, input, shared):
        return conv2d_fa_asap0.apply(input, shared, self.weight, self.weight_fa, self.bias, self.stride, self.padding, self.groups, self.wt) 

class ASAP_Conv_Block0(nn.Module): # for ASAP in ConvNet
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, wt = False):
        super(ASAP_Conv_Block0, self).__init__()
        
        self.conv = Conv2d_FA_ASAP0(in_channels, out_channels, kernel_size, stride, padding, wt = wt)
        self.bn = nn.BatchNorm2d(out_channels)
        self.max = nn.MaxPool2d(1,1, ceil_mode = True)
        
    def forward(self, x, save):
        x = self.conv(x, save)
        x = self.max(F.relu(self.bn(x)))
        return x
class conv2d_fa_reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, weight_fa, bias, stride=1, padding=0, groups=1, wt = False):
        
        dum = torch.zeros_like(input).to(input.device)
        output = cudnn_convolution.convolution(input, weight, bias, stride, padding, (1, 1), groups, False, False)
        #shared = shared.detach().clone()
        dum_channel_ratio = int(input.size(1)/dum.size(1)) #for matching shared activation with actual activation
        dum_filter_ratio = int(dum.size(2)/input.size(2)) #for matching shared activation with actual activation
        
        ctx.stride = stride
        ctx.padding = padding
        ctx.groups = groups
        ctx.dum_channel_ratio = dum_channel_ratio 
        ctx.dum_filter_ratio = dum_filter_ratio
        ctx.wt = wt
        ctx.save_for_backward(input, weight, weight_fa, bias, dum)
        
        return output, dum
    
    @staticmethod
    def backward(ctx, grad_output, grad_dum):
        input, weight, weight_fa, bias, dum = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        groups = ctx.groups
        wt = ctx.wt
        
        dum_channel_ratio = ctx.dum_channel_ratio
        dum_filter_ratio = ctx.dum_filter_ratio
        grad_input = grad_weight = grad_bias  = None
        
        # Matching shared activation with actual activation by concatenation and maxpool.
        dum = torch.cat([dum] * dum_channel_ratio, 1)
        dum = F.max_pool2d(dum, dum_filter_ratio)       
        
        if wt:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is weight transport. 
            grad_weight = cudnn_convolution.convolution_backward_weight(grad_dum, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = None
            grad_input = cudnn_convolution.convolution_backward_input(grad_dum.shape, weight, grad_output, stride, padding, (1, 1), groups, False, False)
        else:
            # by using shared in grad_weight, activation sharing is done.
            # by using weight in grad_input, there is no weight transport. 
            # we traind weight_fa by grad_weight_fa = grad_weight
            grad_weight = cudnn_convolution.convolution_backward_weight(grad_dum, weight.shape, grad_output, stride, padding, (1, 1), groups, False, False)
            grad_weight_fa = grad_weight
            grad_input = cudnn_convolution.convolution_backward_input(grad_dum.shape, weight_fa, grad_output, stride, padding, (1, 1), groups, False, False)
            
        if bias is not None:
            grad_bias = grad_output.sum(dim=[0,2,3])
        
        return grad_input, grad_weight, grad_weight_fa, grad_bias, None, None, None, None
