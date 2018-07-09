# coding: utf-8
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import csv
import sys
import time
import os

from network import MobileNetV2
from util import test



if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,4'

    start_time = time.time()
    max_val_acc = 0

    # load data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.53129727, 0.5259391, 0.52069134), (0.28938246, 0.28505746, 0.27971658))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.53129727, 0.5259391, 0.52069134), (0.28938246, 0.28505746, 0.27971658))])

    trainset = CIFAR10("/home/tinyalpha/dataset/cifar10", transform = transform_train, train = True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 2)

    testset = CIFAR10("/home/tinyalpha/dataset/cifar10", transform = transform_test, train = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 2)

    # write header
    with open('log.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "train_loss", "val_loss", "acc", "val_acc"])

    device_ids = [0,1]
    # build model and optimizer
    model = nn.DataParallel(MobileNetV2(10, alpha = 1), device_ids = device_ids)
    # model = MobileNetV2(10, alpha = 1)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    # model.load_state_dict(torch.load("weights.pkl"))


    # train
    i = 0
    correct, total = 0, 0
    train_loss, counter = 0, 0

    for epoch in range(300):
        epoch_start_time = time.time()

        # update lr
        if epoch == 0:
            optimizer = optim.SGD(model.parameters(), lr = 1e-1, weight_decay = 4e-5, momentum = 0.9)
        elif epoch == 150:
            optimizer = optim.SGD(model.parameters(), lr = 1e-2, weight_decay = 4e-5, momentum = 0.9)
        elif epoch == 225:
            optimizer = optim.SGD(model.parameters(), lr = 1e-3, weight_decay = 4e-5, momentum = 0.9)

        # iteration over all train data
        for data in trainloader:
            # shift to train mode
            model.train()
            
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # count acc,loss on trainset
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()        
            train_loss += loss.item()
            counter += 1

            if i % 100 == 0:
                # get acc,loss on trainset
                acc = correct / total
                train_loss /= counter
                
                # test
                val_loss, val_acc = test(model, testloader, criterion)

                print('iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f' 
                      %(i, epoch, train_loss, val_loss, acc, val_acc))
                
                # save logs and weights
                with open('log.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, train_loss, val_loss, acc, val_acc])
                if val_acc > max_val_acc:
                    torch.save(model.state_dict(), 'weights.pkl')
                    max_val_acc = val_acc
                    
                # reset counters
                correct, total = 0, 0
                train_loss, counter = 0, 0

            i += 1
        print("epoch time %.4f min" %((time.time() - epoch_start_time)/60))

    end_time = time.time()
    print("total time %.1f h" %((end_time - start_time)/3600))
