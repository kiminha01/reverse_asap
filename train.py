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
from resnet import *
from utils import progress_bar
import numpy as np
import math

#%%
parser = argparse.ArgumentParser(description='Activation Sharing with Asymmetric Paths')
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')
parser.add_argument('--batch-size', type=int, 
                    default=128, help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, 
                    default=100, help='input batch size for teset (default: 100)')
parser.add_argument('--epochs', type=int, default=200,  
                    help='number of epochs to train (default: 200)')
parser.add_argument('--dataset', type= str ,choices = ['mnist', 'cifar10', 'cifar100', 'svhn', 'tiny-imagenet'], 
                    default='cifar10', help='choose dataset (default: cifar10)')
parser.add_argument('--model', type= str ,choices = ['convnet', 'resnet18_not','resnet18' ,'resnet34'], 
                    default='convnet', help='choose architecture (default: convnet)')
parser.add_argument('--feedback', type=str, 
                    choices = ['bp', 'fa', 'dfa', 'll', 'drtp','kp','asap','reverse','asap0'],
                    default='bp',  help='feedback to use (default: bp)')
parser.add_argument('--aug', action = 'store_true', default = False, 
                    help = 'data augmentataion with random crop, horizontalflip (default : False)')
parser.add_argument('--optimizer', type=str, choices = ['sgd', 'adam'], default = 'sgd'
                    , help = 'choose optimizer (default : sgd)')
parser.add_argument('--device', type= int, default = 0, help='device_num')

def main():
    args = parser.parse_args()
    device = args.device
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    # Data
    # All data licensed under CC-BY-SA.
    dataset = args.dataset
    if dataset == 'mnist':
        n_classes = 10
        trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),batch_size=args.test_batch_size, shuffle=False)
    
    elif dataset == 'svhn':
        n_classes = 10
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


        trainloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'train',download=True, transform=transform),
                                                  batch_size=args.batch_size,
                                                  shuffle=True)

        testloader = torch.utils.data.DataLoader(torchvision.datasets.SVHN('./data', split = 'test',download=True, transform=transform),
                                                  batch_size=args.test_batch_size,
                                                  shuffle=False)
        
    elif dataset == 'cifar10':
        n_classes = 10
        if args.aug :
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else: 
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
                
    elif dataset == 'cifar100':
        n_classes = 100
        if args.aug:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2)
    
    elif dataset == 'tiny-imagenet':
        n_classes = 200
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])
        
        num_workers = 16
        trainset = torchvision.datasets.ImageFolder('/datasets/tiny_imagenet/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder('/datasets/tiny_imagenet/val', transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                 num_workers=num_workers)
        
    def make_labels_one_hot(y, n_classes):
        """
        Take a vector of labels and make it one-hot labels.
        :param y:   torch.Tensor with 1 dimension, each entry is a class from 0 to n-1
        :return:    torch.Tensor, y_onehot such that y_onehot.shape = (y.shape[0], n) and y_onehot[i][y[i]] == 1
        https://github.com/romanpogodin/plausible-kernelized-bottleneck
        """
        y_onehot = torch.zeros(y.shape[0], n_classes, dtype=torch.float, device=y.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot.detach()
    
    # Set the model               
    learning_kwargs = {'dataset' : args.dataset,
                    'feedback' : args.feedback}
    resnet_learning_kwargs = {'dataset' : args.dataset,
                               'feedback' : args.feedback,
                               'model' : args.model}
    
    save_file_name = args.dataset + '_' + args.model + '_' + args.feedback 
    print(learning_kwargs)
    if args.model == 'convnet':
        net = convnet(**learning_kwargs)
    else:
        if args.feedback == 'asap0' or args.feedback == 'asap0_k4':
            net = resnet_asap0(**resnet_learning_kwargs)
        elif args.feedback =='asap' or args.feedback == 'asap_k4' or args.feedback == 'reverse':
            net = resnet_asap(**resnet_learning_kwargs)
        else:
            net = resnet(**resnet_learning_kwargs)
    net = net.to(device)
    #net = nn.DataParallel(net, device_ids =[0,1,2])
    print(net)
    
    # Decide optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr,  weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        loss = 0
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            one_hot = make_labels_one_hot(targets,n_classes)
            outputs = net(inputs, one_hot)
            if args.feedback == 'll':
                for local_output in net.local_activities:
                    local_loss = F.cross_entropy(local_output, targets)
                    local_loss.backward(retain_graph=True)
            
            
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
        return 100.*correct/total, train_loss
        
    def test(epoch, best_acc):
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs, targets)
                loss = criterion(outputs, targets)
    
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        acc = 100.*correct/total

        return acc, test_loss
    
    result = []
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_acc, train_loss = train(epoch)
        test_acc, test_loss = test(epoch, best_acc)
        scheduler.step() 
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            if not os.path.isdir('results'):
                os.mkdir('results')
            torch.save(state, './checkpoint/' + save_file_name + '.pth')
            best_acc = test_acc
            print('best accuracy : ' + str(round(best_acc, 2)), 'epoch : ' + str(epoch) + '\n')
        result.append([train_acc, test_acc, train_loss, test_loss])
        np.save("results/" + save_file_name, result)
        
if __name__ == '__main__':
    main()      

#%%
