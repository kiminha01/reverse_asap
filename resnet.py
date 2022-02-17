

import torch
import torch.nn as nn
import torch.nn.functional as F
from building_blocks import *

class Dfa_Block(nn.Module): # for DFA in ResNet
    def __init__(self, connect_channels):
        super(Dfa_Block, self).__init__()
        self.dfa = Feedback_Receiver(connect_channels)
    
    def forward(self, x):
        x, dm1 = self.dfa(x)
        self.dummy = dm1
        return 

class resnet(nn.Module):
    def __init__(self, dataset='cifar10', feedback='bp', model = 'resenet18', wt = False):
        super(resnet, self).__init__()
        self.feedback = feedback
        self.not_shortcut = False # Decide whether we use shortcut or not
        block = BasicBlock
        
        # Choose dimension of architecture by dataset
        if dataset == 'mnist':
            width = 28
            dim = 1
            num_classes = 10
        elif dataset == 'cifar10' or dataset == 'svhn':
            width = 32
            dim = 3
            num_classes = 10
        else:
            width = 32
            dim = 3
            num_classes = 100
        self.num_classes = num_classes
        
        # Decide number of layer for resnet
        if model == 'resnet34':
            num_blocks = [3,4,6,3]
        else:
            num_blocks = [2,2,2,2]
            if model == 'resnet18_not':
                self.not_shortcut == True
        
        # Make architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        if feedback == 'fa' or feedback == 'dfa':
                self.conv1 = Conv2d_FA(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        if feedback == 'dfa':
            self.feedback1 = Feedback_Receiver(num_classes)
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, feedback = feedback)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, feedback = feedback)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, feedback = feedback)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, feedback = feedback)
        self.classifier = nn.Sequential(
                            nn.Linear(512*block.expansion, num_classes),
                            nn.BatchNorm1d(num_classes)
                            )
        if feedback == 'fa' or feedback == 'dfa':
            self.classifier = nn.Sequential(
                            Linear_FA(512*block.expansion, num_classes),
                            nn.BatchNorm1d(num_classes)
                            )
            if feedback == 'dfa':
                self.top = Top_Gradient()
        
        # For DFA, make a list of dfa layers to extract dummy
        self.dfa_layers = [layer for layer in self.modules() if isinstance(layer, BasicBlock)]
        
        
    def _make_layer(self, block, planes, num_blocks, stride, feedback = 'bp'):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        if feedback =='bp':
            mode = 'bp'
        elif feedback == 'dfa':
            mode = 'dfa'
        else:
            mode = 'fa'
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, mode = mode, connect_features = self.num_classes, not_shortcut = self.not_shortcut))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, target):
        if self.feedback != 'dfa':
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out,dm1 = self.feedback1(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            dummies = [x.dummy for x in self.dfa_layers] # extract dummy for sending top gradient to layers
            dummies = sum(dummies,())
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            out = self.top(out, *dummies,dm1) # distribute top error to layers
        return out
    

class resnet_asap0(nn.Module):
    def __init__(self, dataset='cifar10', feedback='asap0', model = 'resnet18', wt = False):
        super(resnet_asap0, self).__init__()
        self.not_shortcut = False
        self.use_previous = False
        self.wt = wt
        block = ASBasicBlock
        
        # Choose dimension of architecture by dataset
        if dataset == 'mnist':
            width = 28
            dim = 1
            num_classes = 10
        elif dataset == 'cifar10' or dataset == 'svhn':
            width = 32
            dim = 3
            num_classes = 10
        else:
            width = 32
            dim = 3
            num_classes = 100
        
        # for asap0 k=4
        self.k_4 = False
        if feedback == 'asap0_k4':
            self.k_4 = True
       
        # Decide number of layer for resnet 
        # While Basic Block includes stride =2 first order,
        # ASBasick block includes stride = 2 last order
        # so num_blocks =[4,4,6,2] is equal to [3,4,6,3] in resnet for ResNet-34
        if model == 'resnet34':
            num_blocks = [4,4,6,2] 
        else:
            num_blocks = [3,2,2,1]
            if model == 'resnet18_not':
                self.not_shortcut = True
        
        # Make architecture
        self.conv0 = Conv2d_FA(3, 64,3,1,1)
        self.bn0 = nn.BatchNorm2d(64)
       
        self.in_planes = 64 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1)
        self.classifier = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        Linear_FA(512, num_classes),
                                        nn.BatchNorm1d(num_classes)
	)
        self.block = [layer for layer in self.modules() if isinstance(layer, Conv2d_FA_ASAP0)]
        
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [1]*(num_blocks-1) + [2] # ASBasick block includes stride = 2 last order
        layers = []
        for i, stride in enumerate(strides):
            if i != num_blocks-1:
                layers.append(block(self.in_planes, planes, stride, use_previous = self.use_previous, not_shortcut = self.not_shortcut, wt = self.wt))
                self.in_planes = planes * block.expansion
            else:
                if planes != 512:
                    layers.append(block(self.in_planes, 2 * planes, stride, use_previous = self.use_previous, not_shortcut = self.not_shortcut, wt = self.wt))
                    self.in_planes = 2 * planes * block.expansion
                else:
                    layers.append(block(self.in_planes, planes, 1, use_previous = self.use_previous, not_shortcut = self.not_shortcut, wt = self.wt))
                    self.in_planes = planes * block.expansion
            if self.k_4:
                # use_previous make ASBasick block to use previous_shared_activation
                # Therefore, k=4 can be implemented  
                # when use_previous is turned on and off alternately for each layer.
                self.use_previous = not(self.use_previous)
                
        return nn.Sequential(*layers)
        
    def forward(self, x, target):
        
        x = F.relu(self.bn0(self.conv0(x).detach()))
        shared = x.clone()
        x = x,shared # decided shared activation
        out, shared = self.layer1(x)
        
        out = out,shared
        out, shared = self.layer2(out)
        
        out = out,shared
        out, shared = self.layer3(out)
        
        out = out,shared
        out, save = self.layer4(out)
        
        out = self.classifier(out)
        
        return out
class resnet_asap(nn.Module):
    def __init__(self, dataset='cifar10', feedback='asap', model = 'resnet18', wt = False):
        super(resnet_asap, self).__init__()
        self.not_shortcut = False
        self.use_previous = False
        self.wt = wt
        block = ASBasicBlock
        
        # Choose dimension of architecture by dataset
        if dataset == 'mnist':
            width = 28
            dim = 1
            num_classes = 10
        elif dataset == 'cifar10' or dataset == 'svhn':
            width = 32
            dim = 3
            num_classes = 10
        else:
            width = 32
            dim = 3
            num_classes = 100
        
        # for asap0 k=4
        self.k_4 = False
        if feedback == 'asap_k4':
            self.k_4 = True
       
        # Decide number of layer for resnet 
        # While Basic Block includes stride =2 first order,
        # ASBasick block includes stride = 2 last order
        # so num_blocks =[4,4,6,2] is equal to [3,4,6,3] in resnet for ResNet-34
        if model == 'resnet34':
            num_blocks = [4,4,6,2] 
        else:
            num_blocks = [3,2,2,1]
            if model == 'resnet18_not':
                self.not_shortcut = True
        
        # Make architecture
        self.conv0 = Conv2d_FA(3, 64,3,1,1)
        self.bn0 = nn.BatchNorm2d(64)
       
        self.in_planes = 64 
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=1)
        self.classifier = nn.Sequential(
                                        nn.AdaptiveAvgPool2d((1,1)),
                                        nn.Flatten(),
                                        Linear_FA(64, num_classes),
                                        nn.BatchNorm1d(num_classes)
	)
        self.block = [layer for layer in self.modules() if isinstance(layer, Conv2d_FA_ASAP)]
        
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [1]*(num_blocks-1) + [1] # ASBasick block includes stride = 2 last order
        layers = []
        for i, stride in enumerate(strides):
            if i != num_blocks-1:
                layers.append(block(self.in_planes, planes, stride, use_previous = self.use_previous, not_shortcut = self.not_shortcut, wt = self.wt))
                self.in_planes = planes * block.expansion
            else:
                if planes != 512:
                    layers.append(block(self.in_planes, 2 * planes, stride, use_previous = self.use_previous, not_shortcut = self.not_shortcut, wt = self.wt))
                    self.in_planes = 2 * planes * block.expansion
                else:
                    layers.append(block(self.in_planes, planes, 1, use_previous = self.use_previous, not_shortcut = self.not_shortcut, wt = self.wt))
                    self.in_planes = planes * block.expansion
            if self.k_4:
                # use_previous make ASBasick block to use previous_shared_activation
                # Therefore, k=4 can be implemented  
                # when use_previous is turned on and off alternately for each layer.
                self.use_previous = not(self.use_previous)
                
        return nn.Sequential(*layers)
        
    def forward(self, x, target):
        
        x = F.relu(self.bn0(self.conv0(x).detach()))
        shared = x.clone()
        x = x,shared # decided shared activation
        out, shared = self.layer1(x)
        
        out = out,shared
        out, shared = self.layer2(out)
        
        out = out,shared
        out, shared = self.layer3(out)
        
        out = out,shared
        out, save = self.layer4(out)
        
        out = self.classifier(out)
        
        return out
    
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, mode='bp', connect_features = 10, not_shortcut = False):
        super(BasicBlock, self).__init__()
        self.mode = mode
        self.not_shortcut = not_shortcut
        if mode == 'bp':
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
        
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
        elif mode == 'dfa':
            self.conv1 = Conv2d_FA(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.feedback1 = Feedback_Receiver(connect_features)
            self.conv2 = Conv2d_FA(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.feedback2 = Feedback_Receiver(connect_features)
            
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    Conv2d_FA(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
        else:
            self.conv1 = Conv2d_FA(
                in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = Conv2d_FA(planes, planes, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
        
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion*planes:
                self.shortcut = nn.Sequential(
                    Conv2d_FA(in_planes, self.expansion*planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
                
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.mode == 'dfa':
            out, dm1 = self.feedback1(out)
        out = self.bn2(self.conv2(out))
        if not(self.not_shortcut):
            out += self.shortcut(x)
        out = F.relu(out)
        if self.mode == 'dfa':
            out, dm2 = self.feedback2(out)
            self.dummy = dm1, dm2
        return out

class ASBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, wt = False, use_previous = False, not_shortcut = False):
        super(ASBasicBlock, self).__init__()
        self.not_shortcut = not_shortcut
        
        self.conv1 = Conv2d_FA_ASAP0(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,wt=wt)
        self.bn1 = nn.BatchNorm2d(planes, affine = True)
        self.conv2 = Conv2d_FA_ASAP0(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, wt = wt)
        
        if in_planes != self.expansion*planes:
            self.short_conv = Conv2d_FA_ASAP0(in_planes, self.expansion*planes, 
                                           kernel_size=1, stride=stride, bias=False, wt =wt)
            self.short_bn = nn.BatchNorm2d(self.expansion*planes, affine = True)
                
        self.bn2 = nn.BatchNorm2d(planes, affine = True)  
        self.in_planes = in_planes
        self.planes = planes
        self.use_previous = use_previous
        
    def forward(self, x):
        '''
        In first layer of resnet block, it is updated with shared activation 
        determined in the previous block.
        
        and then we determined new shared activation in present block.
        
        this shared activation is used for second layer of present resnet block and
        first layer of next resnet block.
        
        However, when k=4, we do not determine new shared activation 
        and just use previous value. And then, next block will determine new shared activation.
        By determining shared activation by skipping resnet blocks one by one, we implemented k=4.
        '''
        x, shared_previous = x # decide present shared activation
        shared_present = x.clone()
        if self.use_previous:
            shared_present = shared_previous 
        out = F.relu(self.bn1(self.conv1(x, shared_previous))) # use previous shared activation
        out = self.bn2(self.conv2(out, shared_present)) # use present shared activation
        
        if not(self.not_shortcut):
            if self.in_planes == self.expansion*self.planes:
                out += x
            else:
                out += self.short_bn(self.short_conv(x, shared_previous))
            
        out = F.relu(out)
        shared = shared_present
        out = out, shared
        
        return out
class ASBasicBlock_F(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, wt = False, use_previous = False, not_shortcut = False):
        super(ASBasicBlock_F, self).__init__()
        self.not_shortcut = not_shortcut
        
        self.conv1 = Conv2d_FA_ASAP(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,wt=wt)
        self.bn1 = nn.BatchNorm2d(planes, affine = True)
        self.conv2 = Conv2d_FA_ASAP(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, wt = wt)
        
        if in_planes != self.expansion*planes:
            self.short_conv = Conv2d_FA_ASAP(in_planes, self.expansion*planes, 
                                           kernel_size=1, stride=stride, bias=False, wt =wt)
            self.short_bn = nn.BatchNorm2d(self.expansion*planes, affine = True)
                
        self.bn2 = nn.BatchNorm2d(planes, affine = True)  
        self.in_planes = in_planes
        self.planes = planes
        self.use_previous = use_previous
        
    def forward(self, x):
        '''
        In first layer of resnet block, it is updated with shared activation 
        determined in the previous block.
        
        and then we determined new shared activation in present block.
        
        this shared activation is used for second layer of present resnet block and
        first layer of next resnet block.
        
        However, when k=4, we do not determine new shared activation 
        and just use previous value. And then, next block will determine new shared activation.
        By determining shared activation by skipping resnet blocks one by one, we implemented k=4.
        '''
        x, shared_previous = x # decide present shared activation
        shared_present = x.clone()
        if self.use_previous:
            shared_present = shared_previous 
        out = F.relu(self.bn1(self.conv1(x, shared_previous))) # use previous shared activation
        out = self.bn2(self.conv2(out, shared_present)) # use present shared activation
        
        if not(self.not_shortcut):
            if self.in_planes == self.expansion*self.planes:
                out += x
            else:
                out += self.short_bn(self.short_conv(x, shared_previous))
            
        out = F.relu(out)
        shared = shared_present
        out = out, shared
        
        return out