# coding: utf-8
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch
import tqdm
import os

# DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from network import MobileNetV2
from util import test

def get_dataloader():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    trainset = CIFAR10("/home/benchmark/cifar10", train = True, transform = transform_train, download = True)
    testset = CIFAR10("/home/benchmark/cifar10", train = False, transform = transform_test, download = False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = False, num_workers = 10, sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 10)
    return trainloader, testloader

def main(local_rank, world_size):
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size)
    if local_rank == 0:
        print("Training start")
        import torch.utils.tensorboard as tb
        writer = tb.SummaryWriter("log")
    
    trainloader, testloader = get_dataloader()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(MobileNetV2(10, alpha = 1).cuda())
    model.cuda()
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.1, weight_decay = 4e-5, momentum = 0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 300, eta_min=1e-6)
    
    correct, total = 0, 0
    train_loss, counter = 0, 0
    max_val_acc = 0

    tbar = tqdm.tqdm(range(300)) if local_rank == 0 else range(300)

    for epoch in tbar:
        trainloader.sampler.set_epoch(epoch)
        for i, (input, target) in enumerate(trainloader):
            model.train()
            input, target = input.cuda(), target.cuda()
            optimizer.zero_grad()
            outputs = model(input)
            loss = criterion(outputs, target)

            loss.backward()
            optimizer.step()
            
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()        
            train_loss += loss.item()
            counter += 1

            if i % 50 == 0 and local_rank == 0:
                 # get acc,loss on trainset
                acc = correct / total
                train_loss /= counter
                
                # test
                val_loss, val_acc = test(model, testloader, criterion)

                print('iteration %d , epoch %d:  loss: %.4f  val_loss: %.4f  acc: %.4f  val_acc: %.4f' 
                      %(i, epoch, train_loss, val_loss, acc, val_acc))
                
                # save logs and weights
                writer.add_scalars('loss', {'train_loss':train_loss}, epoch * len(trainloader) + i)
                writer.add_scalars('loss', {'test_loss':val_loss}, epoch * len(trainloader) + i)
                writer.add_scalars('acc', {'train_acc':acc}, epoch * len(trainloader) + i)
                writer.add_scalars('acc', {'test_acc':val_acc}, epoch * len(trainloader) + i)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch * len(trainloader) + i)
                if val_acc > max_val_acc:
                    # torch.save(model.state_dict(), 'weights.pkl')
                    max_val_acc = val_acc
                    
                # reset counters
                correct, total = 0, 0
                train_loss, counter = 0, 0
        scheduler.step()

    if local_rank == 0:

        torch.save(model.state_dict(), 'model.pth')
        print(f"Training end with acc {max_val_acc:.2f}")

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '23333'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
    world_size = NUM_GPUS = 2
    mp.spawn(main, nprocs=NUM_GPUS, args=(world_size,), join=True)
    