import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import json
import os
import argparse

from convnet import *
#from resnet import *
from utils import progress_bar
import numpy as np
import math
#%%
parser = argparse.ArgumentParser(description='Activation Sharing with Asymmetric Paths')
parser.add_argument('--test-batch-size', type=int, 
                    default=100, help='input batch size for test (default: 100)')
parser.add_argument('--dataset', type= str ,choices = ['mnist', 'cifar10', 'cifar100', 'svhn', 'tiny-imagenet'], 
                    default='cifar10', help='choose dataset (default: cifar10)')
parser.add_argument('--model', type= str ,choices = ['convnet', 'resnet18_not','resnet18' ,'resnet34','resnet68','resnet50','resnet101'], 
                    default='convnet', help='choose architecture (default: convnet)')
parser.add_argument('--feedback', type=str, 
                    choices = ['bp', 'fa', 'dfa', 'asap','asap_k4'],
                    default='bp',  help='feedback to use (default: bp)')
parser.add_argument('--aug', action = 'store_true', default = False, 
                    help = 'load pretrained model with augmentation(default : False)')
parser.add_argument('--wt', action = 'store_true', default = False
                    , help = 'activation sharing with transposed weight (default : False)')
parser.add_argument('--model_path', type=str, help='The path to the saved model file')
parser.add_argument('--optimizer', type=str, choices = ['sgd', 'adam'], default = 'sgd'
                    , help = 'choose optimizer (default : sgd)')
parser.add_argument('--device', type= int, default = 0, help='device_num')


def main():
    args = parser.parse_args()
    device = args.device

    # Data
    # All data licensed under CC-BY-SA.
    dataset = args.dataset
    if dataset == 'mnist':
        testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=50, shuffle=False)
    
    elif dataset == 'svhn':
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        testloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'test',download=True, transform=transform),
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False)
        
    elif dataset == 'cifar10':        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
                
    elif dataset == 'cifar100':
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
        
    elif dataset == 'tiny-imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        
        num_workers = 16
        testset = torchvision.datasets.ImageFolder('/datasets/tiny_imagenet/val', transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=num_workers)
    
    
    # load the pretrained model                 
    learning_kwargs = {'dataset' : args.dataset,
                    'feedback' : args.feedback}
   
    resnet_learning_kwargs = {'dataset' : args.dataset,
                               'feedback' : args.feedback,
                               'model' : args.model,
                               'wt' : args.wt}

    #load_file_name = args.dataset + '_' + args.model + '_' + args.feedback + '_' + args.optimizer + '_wt : ' + str(args.wt)
    #
    #if args.aug:
    #    load_file_name = args.dataset + '_' + args.model + '_' + args.feedback + '_' + 'augmentation' + '_wt : ' + str(args.wt)
    
    if args.model == 'convnet':
        net = convnet(**learning_kwargs)
    else:
        if args.feedback == 'asap' or args.feedback == 'asap_k4':
            net = resnet_asap(**resnet_learning_kwargs)
        else:
            net = resnet(**resnet_learning_kwargs)
        
    net = net.to(device)
    print(net)
    
    #PATH = './checkpoint/' + load_file_name + '.pth'
    PATH = args.model_path
    checkpoint = torch.load(PATH)
    state_dict = checkpoint['net']
    
    net.load_state_dict({k.replace('module.',''):v for k,v in state_dict.items()}, strict = False)

    
    #Only eval
    criterion = nn.CrossEntropyLoss()
    def test():
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        acc = 100.*correct/total

        return acc, test_loss
        

    test_acc, test_loss = test()
    
    print('test accuracy : ' + str(round(test_acc, 2)))
    
if __name__ == '__main__':
    main()      

#%%
