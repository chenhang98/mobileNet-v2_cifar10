import torch


def test(model, testloader, criterion):
    # test model on testloader
    # return val_loss, val_acc
    
    model.eval()
    correct, total = 0, 0
    loss, counter = 0, 0
    
    with torch.no_grad():
        for (images, labels) in testloader:

            images = images.cuda()
            labels = labels.cuda()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            counter += 1
    
    return loss / counter, correct / total