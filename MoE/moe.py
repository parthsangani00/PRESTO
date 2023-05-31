# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

from moe-utils import MoE
import sys

from myDatasets import *

from sklearn.metrics import f1_score
from arguments import myArgParse

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_torch(0)
    args = myArgParse()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
    #                                           shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=64,
    #                                          shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataset_map = {
        'cifar10-rotate': CIFAR10_ROTATE,
        'cifar100-rotate': CIFAR100_ROTATE,
        'stl-rotate': STL10_ROTATE
    }

    trainset = dataset_map[args.dataset](train = True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle = True)
    testset = dataset_map[args.dataset](train = False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle = False)


    if torch.cuda.is_available():
        device = torch.device('cuda:4')
    else:
        device = torch.device('cpu')

    net = MoE(input_size=8192, output_size=trainset.num_classes, num_experts=args.n_bins, hidden_size=256, noisy_gating=True, k=args.n_bins)
    # net = MoE(input_size=3072, output_size=10, num_experts=3, hidden_size=256, noisy_gating=True, k=3)
    net = net.to(device)

    print("Number of params in MoE: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net.train()
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (index, inputs, labels) in enumerate(trainloader):

        # for i, data in enumerate(trainloader):

        #     inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # print("first", inputs.shape, labels.shape)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.view(inputs.shape[0], -1)
            # print("second", inputs.shape)
            outputs, aux_loss = net(inputs)
            # print("third", outputs.shape)
            loss = criterion(outputs, labels)
            total_loss = loss + aux_loss
            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        print(f"Epoch {epoch + 1} took {time.time() - start} seconds time")

    print('Finished Training')


    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for i, (index, images, labels) in enumerate(testloader):
        # for data in testloader:
        #     images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, _ = net(images.view(images.shape[0], -1))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i == 0:
                pred = predicted.clone()
                actual = labels.clone()
            else:
                pred = torch.cat((pred,predicted),dim =0)
                actual = torch.cat((actual,labels),dim =0)

    print('Accuracy of the network on the 10000 test images:',(100 * correct / total))
    micro =  f1_score(actual.cpu().numpy(), pred.cpu().numpy(), average='micro')
    macro =  f1_score(actual.cpu().numpy(), pred.cpu().numpy(), average='macro')
    print("Micro : ",micro)
    print("Macro : ",macro)

    # yields a test accuracy of around 34 %
