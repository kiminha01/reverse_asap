import torch
import torch.nn as nn
import torch.nn.functional as F
from building_blocks import *

class convnet(nn.Module):
    def __init__(self, dataset='cifar10', feedback='bp'):
        super(convnet, self).__init__()
        
        self.feedback = feedback
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
        
        if self.feedback == 'bp':
            self.layers = nn.Sequential(
                    nn.Conv2d(dim, 64, 5, 1, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    nn.Conv2d(64, 128, 5, 1, 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    nn.Conv2d(128, 256, 5, 1, 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    nn.Flatten(),
                    nn.Linear(4096,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    nn.Linear(1024,num_classes),
                    nn.BatchNorm1d(num_classes),
                    )
        elif self.feedback == 'fa':
            self.layers = nn.Sequential(
                    Conv2d_FA(dim, 64, 5, 1, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    Conv2d_FA(64, 128, 5, 1, 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    Conv2d_FA(128, 256, 5, 1, 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    nn.Flatten(),
                    Linear_FA(4096,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,num_classes),
                    nn.BatchNorm1d(num_classes),
                    )

        elif self.feedback =='kp':
            self.layers = nn.Sequential(
                Conv2d_KP(dim, 64, 5, 1, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    Conv2d_KP(64, 128, 5, 1, 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    Conv2d_KP(128, 256, 5, 1, 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True),
                    nn.Flatten(),
                    Linear_KP(4096,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_KP(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_KP(1024,num_classes),
                    nn.BatchNorm1d(num_classes),
                    )
        elif self.feedback == 'dfa':
            self.layer1 = nn.Sequential(
                    Conv2d_FA(dim, 64, 5, 1, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.feedback1 = Feedback_Receiver(num_classes)
            
            self.layer2 = nn.Sequential(
                    Conv2d_FA(64, 128, 5, 1, 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.feedback2 = Feedback_Receiver(num_classes)
            
            self.layer3 = nn.Sequential(
                    Conv2d_FA(128, 256, 5, 1, 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.feedback3 = Feedback_Receiver(num_classes)
            
            self.layer4 = nn.Sequential(nn.Flatten(),
                    Linear_FA(4096,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                    )
            self.feedback4 = Feedback_Receiver(num_classes)
            
            self.layer5 = nn.Sequential(nn.Flatten(),
                    Linear_FA(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                    )
            self.feedback5 = Feedback_Receiver(num_classes)
            
            self.layer6 = nn.Sequential(nn.Flatten(),
                    Linear_FA(1024,num_classes),
                    nn.BatchNorm1d(num_classes)
                    )
            self.top = Top_Gradient()

        elif self.feedback == 'll':
            self.layer1 = nn.Sequential(
                    Conv2d_FA(dim, 64, 5, 1, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.lc1 = Local_Activity(num_classes)
            
            self.layer2 = nn.Sequential(
                    Conv2d_FA(64, 128, 5, 1, 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.lc2 = Local_Activity(num_classes)
            
            self.layer3 = nn.Sequential(
                    Conv2d_FA(128, 256, 5, 1, 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.lc3 = Local_Activity(num_classes)
            
            self.layer4 = nn.Sequential(nn.Flatten(),
                    Linear_FA(4096,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                    )
            self.lc4 = Local_Activity(num_classes)
            
            self.layer5 = nn.Sequential(nn.Flatten(),
                    Linear_FA(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                    )
            self.lc5 = Local_Activity(num_classes)
            
            self.layer6 = nn.Sequential(nn.Flatten(),
                    Linear_FA(1024,num_classes),
                    nn.BatchNorm1d(num_classes)
                    )
            
            self.local_layers = [layer for layer in self.modules() if isinstance(layer, Local_Activity)]
        
        elif self.feedback == 'drtp':
            self.layer1 = nn.Sequential(
                    Conv2d_FA(dim, 64, 5, 1, 2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.tp1 = Target_Projection()
            
            self.layer2 = nn.Sequential(
                    Conv2d_FA(64, 128, 5, 1, 2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.tp2 = Target_Projection()
            
            self.layer3 = nn.Sequential(
                    Conv2d_FA(128, 256, 5, 1, 2),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(2,2, ceil_mode = True))
            self.tp3 = Target_Projection()
            
            self.layer4 = nn.Sequential(nn.Flatten(),
                    Linear_FA(4096,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                    )
            self.tp4 = Target_Projection()
            
            self.layer5 = nn.Sequential(nn.Flatten(),
                    Linear_FA(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                    )
            self.tp5 = Target_Projection()
            
            self.layer6 = nn.Sequential(nn.Flatten(),
                    Linear_FA(1024,num_classes),
                    nn.BatchNorm1d(num_classes)
                    )
        elif self.feedback == 'asap':
            self.conv0 = Conv2d_FA(dim,64,5,1,2)
            self.bn0 = nn.BatchNorm2d(64)        
            
            self.layer1 = ASAP_Conv_Block(64,64,5,1,2, )
            self.layer2 = ASAP_Conv_Block(64,64,5,1,2, )
            
            self.classifier = nn.Sequential(
                    nn.Flatten(),
                    Linear_FA(16384,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,num_classes),
                    nn.BatchNorm1d(num_classes)
                    )
        elif self.feedback=='reverse':
            self.conv0 = Conv2d_FA(dim,64,5,1,2)
            self.bn0 = nn.BatchNorm2d(64)        
            
            self.layer1 = ASAP_Reverse_Block(64,64,5,1,padding=2 )
            self.layer2 = ASAP_Reverse_Block(64,64,5,1,padding=2 )
            
            self.classifier = nn.Sequential(
                    nn.Flatten(),
                    Linear_FA(16384,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,num_classes),
                    nn.BatchNorm1d(num_classes)
                    )
        elif self.feedback == 'asap0':
            self.conv0 = Conv2d_FA(dim,64,5,1,2)
            self.bn0 = nn.BatchNorm2d(64)        
            
            self.layer1 = ASAP_Conv_Block0(64,128,5,1,2, wt = False)
            self.layer2 = ASAP_Conv_Block0(128,256,5,1,2, wt = False)
            
            self.classifier = nn.Sequential(
                    nn.Flatten(),
                    Linear_FA(4096,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(),
                    Linear_FA(1024,num_classes),
                    nn.BatchNorm1d(num_classes)
                    )    
            
    def forward(self,x,target):
        if self.feedback == 'bp' or self.feedback == 'fa' or self.feedback=='kp':
            x = self.layers(x)
       
        elif self.feedback == 'dfa':
            x = self.layer1(x)
            x,dm1 = self.feedback1(x)
            
            x = self.layer2(x)
            x,dm2 = self.feedback2(x)
            
            x = self.layer3(x)
            x,dm3 = self.feedback3(x)
            
            x = self.layer4(x)
            x,dm4 = self.feedback4(x)
            
            x = self.layer5(x)
            x,dm5 = self.feedback5(x)
            
            x = self.layer6(x)
            x = self.top(x,dm1,dm2,dm3,dm4,dm5)

        elif self.feedback == 'll':
            x = self.layer1(x)
            x = self.lc1(x)
            
            x = self.layer2(x)
            x = self.lc2(x)
            
            x = self.layer3(x)
            x = self.lc3(x)
            
            x = self.layer4(x)
            x = self.lc4(x)
            
            x = self.layer5(x)
            x = self.lc5(x)
            
            x = self.layer6(x)
            
            self.local_activities = [x.local_activity for x in self.local_layers]
        
        elif self.feedback == 'drtp':
            x = self.layer1(x)
            x = self.tp1(x, target)
            
            x = self.layer2(x)
            x = self.tp2(x, target)
            
            x = self.layer3(x)
            x = self.tp3(x, target)
            
            x = self.layer4(x)
            x = self.tp4(x, target)
            
            x = self.layer5(x)
            x = self.tp5(x, target)
            
            x = self.layer6(x)
            
        
        elif self.feedback == 'asap':
            x = F.max_pool2d(F.relu(self.bn0(self.conv0(x).detach())),2)
            shared = x.clone()
            
            x = self.layer1(x,shared)
            x = self.layer2(x,shared.detach().clone())
            x = self.classifier(x)
        elif self.feedback == 'reverse':
            x = F.max_pool2d(F.relu(self.bn0(self.conv0(x).detach())),2)
            y=x.detach().clone()
            shared=self.layer2(self.layer1(y,None).detach().clone(),None).detach().clone()
            x = self.layer1(x,shared)
            x = self.layer2(x,shared.detach().clone())
            x = self.classifier(x)
        elif self.feedback == 'asap0':
            x = F.max_pool2d(F.relu(self.bn0(self.conv0(x).detach())),2)
            shared = x.clone()
            
            x = self.layer1(x,shared)
            x = self.layer2(x,shared.detach().clone())
            x = self.classifier(x)
        return x
#%%



    

    

