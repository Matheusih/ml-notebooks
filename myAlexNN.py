import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.models import alexnet
from myutils import *

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class myAlexNet:
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = alexnet()
        self.model = self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.kernel_optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

    def learn(self, X, y):
        self.model.train()

        output = self.model(X)
        loss = self.criterion(output,y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss, output

    def validate(self, test_loader):
        self.model = self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        return 100 * correct / total
