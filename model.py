from torch import nn
import torch
from torchvision import models
class siamese(nn.Module):
    def __init__(self):
        super(siamese, self).__init__()
        
        self.base_model = models.resnet101(weights=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.fc = nn.Sequential(
            nn.Linear(2048, 128)
        )
        
    def forward_once(self, x):
        output = self.base_model(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class contrastive_loss(nn.Module):

    def __init__(self, margin=1.0):
        super(contrastive_loss, self).__init__()
        self.margin = margin

    def diff(self, x0, x1):
        return x0 -x1

    def accuracy(self, x0, x1, y):
        diff = self.diff(x0, x1)
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        res = torch.Tensor([1 if dist_ < 0.4 else 0 for dist_ in dist]).cuda()
        acc = torch.sum(res == dist) / len(y)

        return acc
    def forward(self, x0, x1, y):
        
        diff = self.diff(x0, x1)
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
    
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size(0)

        return loss
